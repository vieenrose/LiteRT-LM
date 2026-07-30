/* TurboQuant packed KV quantizer, numerically matched to turboquant's
 * TurboQuantMSE (per-vector L2 norm + fixed seed-42 QR rotation + Lloyd-Max
 * codebook).  Rotation and codebook are LOADED from files exported by
 * prep_assets.py -- the gist kernel's own xoshiro/QR path is NOT bit-compatible
 * with torch.
 *
 * Packed block for dimension d at B bits: 4-byte fp32 norm + ceil(B*d/8) bytes
 * of sequentially bit-packed indices (little-endian bit order, same as the
 * validated gist tq_pack_indices).
 *   B=3: d=256 -> 100 B, d=512 -> 196 B
 *   B=4: d=256 -> 132 B, d=512 -> 260 B
 */
#ifndef TQ3_H
#define TQ3_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define TQ3_MAX_LEVELS 16

typedef struct {
    int d;
    int bits;                          /* 3, 4, or 16 (exact fp16) */
    int nlev;                          /* 1 << bits */
    float centroids[TQ3_MAX_LEVELS];
    float bounds[TQ3_MAX_LEVELS - 1];  /* interior decision boundaries */
    float *rot;                        /* d*d row-major Pi */
    size_t block_bytes;                /* 4 + (bits*d+7)/8 */
} tq3_ctx;

/* bits == 16 is a special EXACT mode: raw little-endian fp16 storage, no
 * rotation and no codebook (rot_path/cb_path are ignored, pass NULL).
 * block_bytes = 2*d.  This is the right choice for models with few query heads
 * per KV head, where 3/4-bit KV costs real accuracy and the sliding-window
 * structure already bounds the cache size.
 *
 * rot_path: d*d fp32; cb_path: (1<<bits) centroids + (1<<bits)-1 boundaries
 * fp32.  Returns 0 on success. */
int tq3_init(tq3_ctx *ctx, int d, int bits, const char *rot_path,
             const char *cb_path);
void tq3_free(tq3_ctx *ctx);

void tq3_quantize(const tq3_ctx *ctx, const float *src, uint8_t *dst,
                  float *scratch /* d floats */);
void tq3_dequantize(const tq3_ctx *ctx, const uint8_t *src, float *dst);

#ifdef __cplusplus
}
#endif
#endif
