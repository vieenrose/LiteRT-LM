#include "tq3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_all(const char *path, void *dst, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(dst, 1, bytes, f);
    fclose(f);
    return n == bytes ? 0 : -1;
}

/* IEEE-754 binary16 <-> binary32, no hardware dependency (the Boox is ARMv8.0
 * without fp16 arithmetic).  Round-to-nearest-even on the way down. */
static inline uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff)                    /* inf / nan */
        return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u); /* overflow -> inf */
    if (exp <= 0) {                                     /* subnormal / zero */
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)exp << 10) | (man >> 13);
    uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t)(sign | h);
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            int sh = 0;
            while (!(man & 0x400u)) { man <<= 1; ++sh; }
            man &= 0x3ffu;
            bits = sign | ((uint32_t)(127 - 15 - sh + 1) << 23) | (man << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

int tq3_init(tq3_ctx *ctx, int d, int bits, const char *rot_path,
             const char *cb_path) {
    memset(ctx, 0, sizeof(*ctx));
    if (bits == 16) {                 /* exact fp16: no rotation, no codebook */
        ctx->d = d;
        ctx->bits = 16;
        ctx->nlev = 0;
        ctx->rot = NULL;
        ctx->block_bytes = (size_t)2 * d;
        return 0;
    }
    if (bits < 1 || bits > 4) return -1;
    ctx->d = d;
    ctx->bits = bits;
    ctx->nlev = 1 << bits;
    ctx->block_bytes = 4 + (size_t)(bits * d + 7) / 8;
    ctx->rot = (float *)malloc((size_t)d * d * sizeof(float));
    if (!ctx->rot) return -1;
    if (read_all(rot_path, ctx->rot, (size_t)d * d * sizeof(float))) return -1;
    float cb[2 * TQ3_MAX_LEVELS];
    const size_t nb = (size_t)(2 * ctx->nlev - 1) * sizeof(float);
    if (read_all(cb_path, cb, nb)) return -1;
    memcpy(ctx->centroids, cb, (size_t)ctx->nlev * sizeof(float));
    memcpy(ctx->bounds, cb + ctx->nlev, (size_t)(ctx->nlev - 1) * sizeof(float));
    return 0;
}

void tq3_free(tq3_ctx *ctx) { free(ctx->rot); ctx->rot = NULL; }

/* sequential B-bit little-endian bit packing (gist tq_pack_indices layout).
 * B <= 8, so an index straddles at most two bytes. */
static inline void packN(const uint8_t *idx, uint8_t *out, int n, int bits) {
    memset(out, 0, (size_t)(bits * n + 7) / 8);
    for (int i = 0; i < n; ++i) {
        int bit = i * bits;
        int byte = bit >> 3, sh = bit & 7;
        out[byte] |= (uint8_t)(idx[i] << sh);
        if (sh + bits > 8) out[byte + 1] |= (uint8_t)(idx[i] >> (8 - sh));
    }
}
static inline void unpackN(const uint8_t *in, uint8_t *idx, int n, int bits) {
    const unsigned mask = (1u << bits) - 1u;
    for (int i = 0; i < n; ++i) {
        int bit = i * bits;
        int byte = bit >> 3, sh = bit & 7;
        unsigned v = in[byte] >> sh;
        if (sh + bits > 8) v |= (unsigned)in[byte + 1] << (8 - sh);
        idx[i] = (uint8_t)(v & mask);
    }
}

void tq3_quantize(const tq3_ctx *ctx, const float *src, uint8_t *dst,
                  float *scratch) {
    const int d = ctx->d, nb = ctx->nlev - 1;
    if (ctx->bits == 16) {
        uint16_t *o = (uint16_t *)dst;
        for (int j = 0; j < d; ++j) o[j] = f32_to_f16(src[j]);
        return;
    }
    double ss = 0.0;
    for (int j = 0; j < d; ++j) ss += (double)src[j] * src[j];
    float norm = (float)sqrt(ss);
    memcpy(dst, &norm, 4);
    const float inv = 1.0f / (norm + 1e-10f);
    for (int j = 0; j < d; ++j) scratch[j] = src[j] * inv;
    uint8_t idx[512];
    for (int i = 0; i < d; ++i) {
        const float *row = ctx->rot + (size_t)i * d;
        float y = 0.f;
        for (int j = 0; j < d; ++j) y += row[j] * scratch[j];
        /* torch.searchsorted(left): count of boundaries strictly < y */
        int k = 0;
        while (k < nb && y > ctx->bounds[k]) ++k;
        idx[i] = (uint8_t)k;
    }
    packN(idx, dst + 4, d, ctx->bits);
}

void tq3_dequantize(const tq3_ctx *ctx, const uint8_t *src, float *dst) {
    const int d = ctx->d;
    if (ctx->bits == 16) {
        const uint16_t *in = (const uint16_t *)src;
        for (int j = 0; j < d; ++j) dst[j] = f16_to_f32(in[j]);
        return;
    }
    float norm;
    memcpy(&norm, src, 4);
    uint8_t idx[512];
    unpackN(src + 4, idx, d, ctx->bits);
    memset(dst, 0, (size_t)d * sizeof(float));
    for (int i = 0; i < d; ++i) {
        const float c = ctx->centroids[idx[i]];
        const float *row = ctx->rot + (size_t)i * d;
        for (int j = 0; j < d; ++j) dst[j] += c * row[j];
    }
    for (int j = 0; j < d; ++j) dst[j] *= norm;
}
