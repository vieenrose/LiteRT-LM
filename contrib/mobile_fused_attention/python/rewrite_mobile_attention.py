#!/usr/bin/env python
"""Fuse the attention blocks of Google's Gemma-4 mobile graph into one custom op.

On CPU, LiteRT runs each `odml.runtime_bmm` composite through its decomposition: it
dequantizes a slice of the int8 KV cache, builds a full-width score tensor (FILL +
DYNAMIC_UPDATE_SLICE), then the main graph broadcasts the mask, selects, softmaxes over the
whole cache length and runs the second composite. XNNPACK allocates, per partition,
workspaces that grow with the square of the cache length (8 * ctx^2 bytes for E4B: 20 x 128 MB
at a 4k context). This tool replaces each block

    runtime_bmm(q, K) -> SELECT_V2(mask) -> SOFTMAX -> runtime_bmm(P, V)

with one custom op

    voxsum.i8_attention(q, K, V, mask, param, meta_f, meta_i) -> ctx

that reads the int8 cache directly, on the live columns only (`mask` true and below
param[2]), with double accumulation. The KV cache, its updates (`odml.cache_update`), the
mask computation and everything else are left as they are.

Inputs of the op: q (1,KV,R,d) f32, K (1,KV,C,d) i8, V (1,KV,d,C) i8 (transposed), mask
(1,1,T,C) bool, param (1,1,1,7) i32, meta_f [beta, k_scale, k_zero, v_scale, v_zero] f32,
meta_i [KV, R, d, T] i32. Row r of q belongs to token r % T. C is not stored anywhere: the
runtime replaces the magic cache length when it loads the model, and the kernel infers C from
the mask size.

The .tflite uses the extended flatbuffer format (weights after the flatbuffer, absolute
offsets): the data section is carried over verbatim and offsets fixed up in two passes
(the approach of vieenrose/LiteRT turboquant-tq3 rewrite_tq3.py).

Usage: rewrite_mobile_attention.py in.tflite out.tflite [signature ...]   (default: prefill_128 decode)
"""
import sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s

B = s.BuiltinOperator
CODE = "voxsum.i8_attention"


def name_of(x):
    return x.decode() if isinstance(x, bytes) else (x or "")


def main(inp, outp, *sigs):
    sigs = set(sigs or ("prefill_128", "decode"))
    raw = open(inp, "rb").read()
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(bytearray(raw), 0))
    offs = [(b.offset, b.size) for b in model.buffers if b.offset and b.offset > 1]
    extended = bool(offs)              # weights after the flatbuffer (files over 2 GB)
    data_start = min(o for o, _ in offs) if extended else len(raw)
    data = raw[data_start:]

    oc = s.OperatorCodeT()
    oc.builtinCode = B.CUSTOM
    oc.deprecatedBuiltinCode = B.CUSTOM
    oc.customCode = CODE
    oc.version = 1
    custom_idx = len(model.operatorCodes)
    model.operatorCodes.append(oc)

    def bcode(op):
        c = model.operatorCodes[op.opcodeIndex]
        return max(c.builtinCode or 0, c.deprecatedBuiltinCode or 0)

    def composite(op):
        if op.builtinOptions2Type == s.BuiltinOptions2.StableHLOCompositeOptions:
            return name_of(op.builtinOptions2.name)
        return None

    ext = []                       # (BufferT, bytes): small constants placed after the flatbuffer

    def const(g, arr, name):
        # The custom-op dispatcher wraps every input as a host buffer and requires 64-byte
        # alignment, which inline flatbuffer data does not give: store the bytes in the data
        # section after the flatbuffer, at aligned offsets (offsets fixed up after packing).
        buf = s.BufferT()
        buf.offset, buf.size = 2, arr.nbytes        # placeholders > 1 keep the u64 fields
        ext.append((buf, arr.tobytes()))
        model.buffers.append(buf)
        t = s.TensorT()
        t.shape = list(arr.shape)
        t.type = s.TensorType.FLOAT32 if arr.dtype == np.float32 else s.TensorType.INT32
        t.buffer = len(model.buffers) - 1
        t.name = name
        g.tensors.append(t)
        return len(g.tensors) - 1

    sig_sub = {name_of(sd.signatureKey): sd.subgraphIndex for sd in model.signatureDefs}
    for key in sorted(sigs):
        g = model.subgraphs[sig_sub[key]]
        cons, prod = {}, {}
        for oi, op in enumerate(g.operators):
            for ti in op.inputs:
                if ti >= 0:
                    cons.setdefault(ti, []).append(oi)
            for ti in op.outputs:
                prod[ti] = oi
        kill, maybe_dead, new_ops = set(), set(), {}
        n = 0
        for oa, a in enumerate(g.operators):
            if composite(a) != "odml.runtime_bmm":
                continue
            K = a.inputs[1]
            if g.tensors[K].type != s.TensorType.INT8:
                continue
            q, param = a.inputs[0], a.inputs[2]
            kshape = list(g.tensors[K].shape)          # (1, KV, C, d): the score bmm
            qshape = list(g.tensors[q].shape)          # (1, KV, R, d)
            if kshape[-1] != qshape[-1] or qshape[-1] > 1024:
                continue                                # the P.V bmm (last dim = cache length): reached from its QK partner
            (osel,) = cons[a.outputs[0]]
            sel = g.operators[osel]
            assert bcode(sel) == B.SELECT_V2, key
            cond = sel.inputs[0]
            (osm,) = cons[sel.outputs[0]]
            sm = g.operators[osm]
            assert bcode(sm) == B.SOFTMAX
            beta = float(sm.builtinOptions.beta) if sm.builtinOptions is not None else 1.0
            (ob,) = cons[sm.outputs[0]]
            bm = g.operators[ob]
            assert composite(bm) == "odml.runtime_bmm" and bm.inputs[0] == sm.outputs[0]
            V = bm.inputs[1]
            ctx_out = bm.outputs[0]
            # mask: RESHAPE <- NOT_EQUAL <- CONCATENATION(xG of one CAST) <- CAST <- bool (1,1,T,C)
            r = g.operators[prod[cond]]
            assert bcode(r) == B.RESHAPE
            ne = g.operators[prod[r.inputs[0]]]
            assert bcode(ne) == B.NOT_EQUAL
            cc = g.operators[prod[ne.inputs[0]]]
            assert bcode(cc) == B.CONCATENATION and len(set(cc.inputs)) == 1
            cast = g.operators[prod[cc.inputs[0]]]
            assert bcode(cast) == B.CAST
            mask = cast.inputs[0]
            mshape = list(g.tensors[mask].shape)
            assert g.tensors[mask].type == s.TensorType.BOOL and len(mshape) == 4
            T = mshape[2]
            kv, R, d = qshape[1], qshape[2], qshape[3]
            assert R % T == 0, (R, T)
            kq, vq = g.tensors[K].quantization, g.tensors[V].quantization
            ks, vs = float(kq.scale[0]), float(vq.scale[0])
            kz = float(kq.zeroPoint[0]) if kq.zeroPoint is not None and len(kq.zeroPoint) else 0.0
            vz = float(vq.zeroPoint[0]) if vq.zeroPoint is not None and len(vq.zeroPoint) else 0.0
            mf = const(g, np.array([beta, ks, kz, vs, vz], np.float32), f"{key}_i8attn_meta_f_{n}")
            mi = const(g, np.array([kv, R, d, T], np.int32), f"{key}_i8attn_meta_i_{n}")
            op = s.OperatorT()
            op.opcodeIndex = custom_idx
            op.inputs = [q, K, V, mask, param, mf, mi]
            op.outputs = [ctx_out]
            op.customOptions = []
            op.customOptionsFormat = 0
            new_ops.setdefault(oa, []).append(op)
            kill |= {oa, osel, osm, ob}
            maybe_dead |= {prod[cond], prod[r.inputs[0]], prod[ne.inputs[0]], prod[cc.inputs[0]]}
            n += 1
        # mask broadcasts (and the CAST, unless something else reads it) whose consumers all died
        changed = True
        while changed:
            changed = False
            for oi in list(maybe_dead - kill):
                outs = g.operators[oi].outputs
                if all(all(c in kill for c in cons.get(t, [])) for t in outs):
                    kill.add(oi)
                    changed = True
        ops2 = []
        for oi, op in enumerate(g.operators):
            ops2.extend(new_ops.get(oi, []))
            if oi not in kill:
                ops2.append(op)
        g.operators = ops2
        print(f"{key}: fused {n} attention blocks, removed {len(kill)} ops, ops now {len(g.operators)}")

    def pack():
        b = flatbuffers.Builder(64 * 1024 * 1024)
        b.Finish(model.Pack(b), file_identifier=b"TFL3")
        return b.Output()

    fb1 = bytes(pack())
    align = 64
    new_start = (len(fb1) + align - 1) // align * align
    delta = new_start - data_start if extended else 0
    for buf in model.buffers:
        if buf.offset and buf.offset > 1 and all(buf is not b for b, _ in ext):
            buf.offset += delta
    tail = bytearray()
    base = new_start + len(data)
    for b, payload in ext:                      # aligned, after the original data section
        pad = (-(base + len(tail))) % align
        tail += b"\0" * pad
        b.offset = base + len(tail)
        tail += payload
    fb2 = bytes(pack())
    assert len(fb2) == len(fb1), (len(fb1), len(fb2))
    with open(outp, "wb") as fo:
        fo.write(fb2)
        fo.write(b"\0" * (new_start - len(fb2)))
        fo.write(data)
        fo.write(tail)
    print(f"wrote {outp}: flatbuffer {len(fb2)} B + data {len(data)} B + {len(ext)} aligned constants "
          f"(delta {delta:+d})")


if __name__ == "__main__":
    main(*sys.argv[1:])
