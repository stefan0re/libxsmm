/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "equation_common.h"
#include <libxsmm_intrinsics_x86.h>

#define FWD_LNORM 1
#define BWD_LNORM 2
#define FWD_BWD_LNORM 3

/*#define USE_VECTORIZED_PATH*/
/*#define USE_SUM*/

#if 0
LIBXSMM_INLINE __m512 _mm512_loadu_ps_auto(libxsmm_bfloat16 const* mem_addr) { return LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)mem_addr)); }
LIBXSMM_INLINE __m512 _mm512_maskz_loadu_ps_auto(__mmask16 k, libxsmm_bfloat16 const* mem_addr) { return LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr)); }
LIBXSMM_INLINE void _mm512_storeu_ps_auto(libxsmm_bfloat16* mem_addr, __m512 a) { _mm256_storeu_si256((__m256i*)mem_addr, LIBXSMM_INTRINSICS_MM512_CVT_FP32_BF16(a)); }
LIBXSMM_INLINE void _mm512_mask_storeu_ps_auto(libxsmm_bfloat16* mem_addr, __mmask16 k, __m512 a) { _mm256_mask_storeu_epi16((__m256i*)mem_addr, k, LIBXSMM_INTRINSICS_MM512_CVT_FP32_BF16(a)); }
#endif

unsigned int is_reference_kernel = 0;
libxsmm_kernel_info info;

LIBXSMM_INLINE
void vectorized_layernorm_fwd_bf16(long S1, long S2, long S3, libxsmm_bfloat16 *pinp, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pbeta, float *mean, float *var, libxsmm_bfloat16 *pout, float eps) {
  int s1, s2, s3;
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, out, pout, S2, S3);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, beta, pbeta, S3);
#if 0
  for (s2 = 0; s2 < S2; s2++) {
    __m512 vm = _mm512_setzero_ps();
    __m512 vv = _mm512_setzero_ps();
    for (s1 = 0; s1 < S1; s1++) {
      for ( s3 = 0; s3 < S3-15; s3+=16) {
        __m512 vin = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        vm = _mm512_add_ps(vm, vin);
        vv = _mm512_add_ps(vv, _mm512_mul_ps(vin, vin));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vin = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        vm = _mm512_add_ps(vm, vin);
        vv = _mm512_add_ps(vv, _mm512_mul_ps(vin, vin));
      }
    }
    float c = 1.0 / (S1*S3);
    float m = _mm512_reduce_add_ps(vm) * c;
    float v = _mm512_reduce_add_ps(vv) * c;
    v = LIBXSMM_MAX(v - m * m, 0.0f);
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    float s = v;
    float b = -1.0 * v * m;
    __m512 vs = _mm512_set1_ps(s);
    __m512 vb = _mm512_set1_ps(b);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3-15; s3+=16) {
        __m512 vin = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vg = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3));
        __m512 vbt = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3));
        __m512 vout = _mm512_add_ps(_mm512_mul_ps(vin, vs), vb);
        vout = _mm512_add_ps(_mm512_mul_ps(vout, vg), vbt);
        _mm512_storeu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), vout);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vin = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vg = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3));
        __m512 vbt = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3));
        __m512 vout = _mm512_add_ps(_mm512_mul_ps(vin, vs), vb);
        vout = _mm512_add_ps(_mm512_mul_ps(vout, vg), vbt);
        _mm512_mask_storeu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), mask, vout);
      }
    }
  }
#else
  for (s2 = 0; s2 < S2; s2++) {
    float m = 0, v = 0, s, b;
    float c = (float)(1.0 / (S1*S3));
    for (s1 = 0; s1 < S1; s1++) {
      for ( s3 = 0; s3 < S3; s3++) {
        m +=  upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        v +=  upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)) * upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
      }
    }
    m = m * c;
    v = v * c;
    v = LIBXSMM_MAX(v - m * m, 0.0f);
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    s = v;
    b = -1.f * v * m;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        float res;
        res = (upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)) * s + b) *  upconvert_bf16(LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3)) + upconvert_bf16(LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3));
        libxsmm_rne_convert_fp32_bf16( &res, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), 1 );
      }
    }
  }
#endif
}

LIBXSMM_INLINE
void vectorized_layernorm_bwd_bf16(long S1, long S2, long S3, libxsmm_bfloat16 *pdout, libxsmm_bfloat16 *pinp, float *mean, float *var, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pdin, float *pdgamma, float *pdbeta) {
  int s1, s2, s3;
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, din, pdin, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, dout, pdout, S2, S3);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);
#if 0
  for (s2 = 0; s2 < S2; s2++) {
    float a = var[s2];
    float b = -a*mean[s2];
    __m512 va = _mm512_set1_ps(a);
    __m512 vb = _mm512_set1_ps(b);
    __m512 vds = _mm512_setzero_ps();
    __m512 vdb = _mm512_setzero_ps();
    float ds = 0.0f;
    float db = 0.0f;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3-15; s3+=16) {
        __m512 vdout = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        __m512 vin = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vdg = _mm512_mul_ps(vdout, _mm512_add_ps(_mm512_mul_ps(va, vin), vb));
        _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3), _mm512_add_ps(vdg, _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3))));
        _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3), _mm512_add_ps(vdout, _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3))));
        __m512 vtmp = _mm512_mul_ps(vdout, _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3)));
        vds = _mm512_add_ps(vds, _mm512_mul_ps(vtmp, vin));
        vdb = _mm512_add_ps(vdb, vtmp);
      }

      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vdout = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        __m512 vin = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vdg = _mm512_mul_ps(vdout, _mm512_add_ps(_mm512_mul_ps(va, vin), vb));
        _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3), mask, _mm512_add_ps(vdg, _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3))));
        _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3), mask, _mm512_add_ps(vdout, _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3))));
        __m512 vtmp = _mm512_mul_ps(vdout, _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3)));
        vds = _mm512_add_ps(vds, _mm512_mul_ps(vtmp, vin));
        vdb = _mm512_add_ps(vdb, vtmp);
      }
    }
    ds += _mm512_reduce_add_ps(vds);
    db += _mm512_reduce_add_ps(vdb);
    float scale = 1.0f / (S1 * S3);
    b = (db * mean[s2] - ds) * a * a * a * scale;
    float c = -b * mean[s2] - db * a * scale;

    vb = _mm512_set1_ps(b);
    __m512 vc = _mm512_set1_ps(c);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3-15; s3+=16) {
        __m512 vdout = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        __m512 vin = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vg = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3));
        __m512 vtmp1 = _mm512_mul_ps(_mm512_mul_ps(va, vdout), vg);
        __m512 vtmp2 = _mm512_add_ps(vtmp1, _mm512_mul_ps(vb, vin));
        __m512 vdin = _mm512_add_ps(vtmp2, vc);
        _mm512_storeu_ps_auto(&LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3), vdin);
      }

      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vdout = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        __m512 vin = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vg = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3));
        __m512 vtmp1 = _mm512_mul_ps(_mm512_mul_ps(va, vdout), vg);
        __m512 vtmp2 = _mm512_add_ps(vtmp1, _mm512_mul_ps(vb, vin));
        __m512 vdin = _mm512_add_ps(vtmp2, vc);
        _mm512_mask_storeu_ps_auto(&LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3), mask, vdin);
      }
    }
  }
#else
  for (s2 = 0; s2 < S2; s2++) {
    float a = var[s2], c;
    float b = -a*mean[s2];
    float ds = 0.0f;
    float db = 0.0f;
    float scale = 1.0f / (S1 * S3);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3) += (a * upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)) + b) * upconvert_bf16(LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3) += upconvert_bf16(LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        ds += upconvert_bf16(LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3)) * upconvert_bf16(LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3)) * upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        db += upconvert_bf16(LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3)) * upconvert_bf16(LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3));
      }
    }
    b = (db * mean[s2] - ds) * a * a * a * scale;
    c = -b * mean[s2] - db * a * scale;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        float res;
        res = upconvert_bf16(LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3))  * a * upconvert_bf16(LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3)) + b * upconvert_bf16(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)) + c;
        libxsmm_rne_convert_fp32_bf16( &res, &LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3), 1 );
      }
    }
  }
#endif
}

LIBXSMM_INLINE
void vectorized_layernorm_fwd_fp32(long S1, long S2, long S3, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps) {
  int s1, s2, s3;
  LIBXSMM_VLA_DECL(3, float, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, out, pout, S2, S3);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, S3);
#if 0
  for (s2 = 0; s2 < S2; s2++) {
    __m512 vm = _mm512_setzero_ps();
    __m512 vv = _mm512_setzero_ps();
    for (s1 = 0; s1 < S1; s1++) {
      for ( s3 = 0; s3 < S3-15; s3+=16) {
        __m512 vin = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        vm = _mm512_add_ps(vm, vin);
        vv = _mm512_add_ps(vv, _mm512_mul_ps(vin, vin));
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vin = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        vm = _mm512_add_ps(vm, vin);
        vv = _mm512_add_ps(vv, _mm512_mul_ps(vin, vin));
      }
    }
    float c = 1.0 / (S1*S3);
    float m = _mm512_reduce_add_ps(vm) * c;
    float v = _mm512_reduce_add_ps(vv) * c;
    v = LIBXSMM_MAX(v - m * m, 0.0f);
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    float s = v;
    float b = -1.0 * v * m;
    __m512 vs = _mm512_set1_ps(s);
    __m512 vb = _mm512_set1_ps(b);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3-15; s3+=16) {
        __m512 vin = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vg = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3));
        __m512 vbt = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3));
        __m512 vout = _mm512_add_ps(_mm512_mul_ps(vin, vs), vb);
        vout = _mm512_add_ps(_mm512_mul_ps(vout, vg), vbt);
        _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), vout);
      }
      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vin = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vg = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3));
        __m512 vbt = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3));
        __m512 vout = _mm512_add_ps(_mm512_mul_ps(vin, vs), vb);
        vout = _mm512_add_ps(_mm512_mul_ps(vout, vg), vbt);
        _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), mask, vout);
      }
    }
  }
#else
  for (s2 = 0; s2 < S2; s2++) {
    float m = 0, v = 0, s, b;
    float c = (float)(1.0 / (S1*S3));
    for (s1 = 0; s1 < S1; s1++) {
      for ( s3 = 0; s3 < S3; s3++) {
        m += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
        v += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
      }
    }
    m = m * c;
    v = v * c;
    v = LIBXSMM_MAX(v - m * m, 0.0f);
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    s = v;
    b = -1.f * v * m;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = (LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * s + b) * LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) + LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3);
      }
    }
  }
#endif
}

LIBXSMM_INLINE
void vectorized_layernorm_bwd_fp32(long S1, long S2, long S3, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta) {
  int s1, s2, s3;
  LIBXSMM_VLA_DECL(3, float, din, pdin, S2, S3);
  LIBXSMM_VLA_DECL(3, float, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, dout, pdout, S2, S3);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);
#if 0
  for (s2 = 0; s2 < S2; s2++) {
    float a = var[s2];
    float b = -a*mean[s2];
    __m512 va = _mm512_set1_ps(a);
    __m512 vb = _mm512_set1_ps(b);
    __m512 vds = _mm512_setzero_ps();
    __m512 vdb = _mm512_setzero_ps();
    float ds = 0.0f;
    float db = 0.0f;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3-15; s3+=16) {
        __m512 vdout = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        __m512 vin = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vdg = _mm512_mul_ps(vdout, _mm512_add_ps(_mm512_mul_ps(va, vin), vb));
        _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3), _mm512_add_ps(vdg, _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3))));
        _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3), _mm512_add_ps(vdout, _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3))));
        __m512 vtmp = _mm512_mul_ps(vdout, _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3)));
        vds = _mm512_add_ps(vds, _mm512_mul_ps(vtmp, vin));
        vdb = _mm512_add_ps(vdb, vtmp);
      }

      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vdout = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        __m512 vin = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vdg = _mm512_mul_ps(vdout, _mm512_add_ps(_mm512_mul_ps(va, vin), vb));
        _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3), mask, _mm512_add_ps(vdg, _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3))));
        _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3), mask, _mm512_add_ps(vdout, _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3))));
        __m512 vtmp = _mm512_mul_ps(vdout, _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3)));
        vds = _mm512_add_ps(vds, _mm512_mul_ps(vtmp, vin));
        vdb = _mm512_add_ps(vdb, vtmp);
      }
    }
    ds += _mm512_reduce_add_ps(vds);
    db += _mm512_reduce_add_ps(vdb);
    float scale = 1.0f / (S1 * S3);
    b = (db * mean[s2] - ds) * a * a * a * scale;
    float c = -b * mean[s2] - db * a * scale;

    vb = _mm512_set1_ps(b);
    __m512 vc = _mm512_set1_ps(c);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3-15; s3+=16) {
        __m512 vdout = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        __m512 vin = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vg = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3));
        __m512 vtmp1 = _mm512_mul_ps(_mm512_mul_ps(va, vdout), vg);
        __m512 vtmp2 = _mm512_add_ps(vtmp1, _mm512_mul_ps(vb, vin));
        __m512 vdin = _mm512_add_ps(vtmp2, vc);
        _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3), vdin);
      }

      if (s3 < S3) {
        int rem = S3 - s3;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vdout = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3));
        __m512 vin = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
        __m512 vg = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3));
        __m512 vtmp1 = _mm512_mul_ps(_mm512_mul_ps(va, vdout), vg);
        __m512 vtmp2 = _mm512_add_ps(vtmp1, _mm512_mul_ps(vb, vin));
        __m512 vdin = _mm512_add_ps(vtmp2, vc);
        _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3), mask, vdin);
      }
    }
  }
#else
  for (s2 = 0; s2 < S2; s2++) {
    float a = var[s2], c;
    float b = -a*mean[s2];
    float ds = 0.0f;
    float db = 0.0f;
    float scale = 1.0f / (S1 * S3);
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3) += (a * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + b) * LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
        LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3) += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
        ds += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
        db += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3);
      }
    }
    b = (db * mean[s2] - ds) * a * a * a * scale;
    c = -b * mean[s2] - db * a * scale;
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3) = LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3)  * a * LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) + b * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + c;
      }
    }
  }
#endif
}

LIBXSMM_INLINE
void tpp_layernorm_fwd_fp32(long S1, long S2, long S3, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps, libxsmm_meqn_function func0, libxsmm_meltwfunction_unary reduce_rows_kernel, libxsmm_meltwfunction_unary reduce_cols_kernel) {
  int s2;
  libxsmm_meqn_param eqn_param;
  libxsmm_meltw_unary_param m_reduce_rows_params, v_reduce_rows_params, reduce_cols_params;
  LIBXSMM_ALIGNED(float tmp[2048], 64);
  const float c = (float)(1.0/(S1*S3));
  float m, v, s, b;
  libxsmm_matrix_arg  arg_array[5];
  LIBXSMM_VLA_DECL(3, float, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, out, pout, S2, S3);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, S3);
  assert((sizeof(*tmp) * S3 * 2) <= sizeof(tmp));

  eqn_param.inputs = arg_array;
  reduce_cols_params.out.primary   = tmp;
  arg_array[1].primary = &s;
  arg_array[2].primary = &b;
  arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, 0, 0, S3);
  arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, 0, 0, S3);
  m_reduce_rows_params.in.primary    = tmp;
  m_reduce_rows_params.out.primary   = &m;
  v_reduce_rows_params.in.primary    = &tmp[S3];
  v_reduce_rows_params.out.primary   = &v;

  for (s2 = 0; s2 < S2; s2++) {
    reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    reduce_cols_kernel(&reduce_cols_params);
    reduce_rows_kernel(&m_reduce_rows_params);
    reduce_rows_kernel(&v_reduce_rows_params);
    m = m * c;
    v = v * c;
    v = LIBXSMM_MAX(v - m * m, 0.0f);
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    s = v;
    b = -1.f * v * m;
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    func0(&eqn_param);
  }
}

LIBXSMM_INLINE
void tpp_layernorm_fwd_bf16(long S1, long S2, long S3, libxsmm_bfloat16 *pinp, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pbeta, float *mean, float *var, libxsmm_bfloat16 *pout, float eps, libxsmm_meqn_function func0, libxsmm_meltwfunction_unary reduce_rows_kernel, libxsmm_meltwfunction_unary reduce_cols_kernel) {
  int s2;
  libxsmm_meqn_param eqn_param;
  libxsmm_meltw_unary_param m_reduce_rows_params, v_reduce_rows_params, reduce_cols_params;
  LIBXSMM_ALIGNED(float tmp[2048], 64);
  const float c = (float)(1.0/(S1*S3));
  float m, v, s, b;
  libxsmm_matrix_arg  arg_array[5];
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, out, pout, S2, S3);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, beta, pbeta, S3);
  assert((sizeof(*tmp) * S3 * 2) <= sizeof(tmp));

  eqn_param.inputs = arg_array;
  reduce_cols_params.out.primary   = tmp;
  arg_array[1].primary = &s;
  arg_array[2].primary = &b;
  arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, 0, 0, S3);
  arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, 0, 0, S3);
  m_reduce_rows_params.in.primary    = tmp;
  m_reduce_rows_params.out.primary   = &m;
  v_reduce_rows_params.in.primary    = &tmp[S3];
  v_reduce_rows_params.out.primary   = &v;

  for (s2 = 0; s2 < S2; s2++) {
    reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    reduce_cols_kernel(&reduce_cols_params);
    reduce_rows_kernel(&m_reduce_rows_params);
    reduce_rows_kernel(&v_reduce_rows_params);
    m = m * c;
    v = v * c;
    v = LIBXSMM_MAX(v - m * m, 0.0f);
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    s = v;
    b = -1.f * v * m;
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    func0(&eqn_param);
  }
}

LIBXSMM_INLINE
void tpp_layernorm_bwd_fp32(long S1, long S2, long S3, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
    libxsmm_meqn_function dgamma_func, libxsmm_meqn_function dbeta_func, libxsmm_meqn_function db_func, libxsmm_meqn_function ds_func, libxsmm_meqn_function din_func) {
  int s2;
  float a, b, c, db, ds;
  const float scale = 1.0f / ((float)S1*S3);
  LIBXSMM_VLA_DECL(3, float, din, pdin, S2, S3);
  LIBXSMM_VLA_DECL(3, float, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, dout, pdout, S2, S3);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);

  libxsmm_meqn_param eqn_param;
  libxsmm_matrix_arg arg_array[8];
  eqn_param.inputs = arg_array;

  arg_array[1].primary = &a;
  arg_array[2].primary = &b;
  arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, 0, 0, S3);
  arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, 0, 0, S3);
  arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, 0, 0, S3);
  arg_array[7].primary = &c;

  for (s2 = 0; s2 < S2; s2++) {
    a = var[s2];
    b = -a*mean[s2];
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    arg_array[3].primary = &LIBXSMM_VLA_ACCESS(3, dout, 0, s2, 0, S2, S3);

    eqn_param.output.primary = &ds;
    ds_func(&eqn_param);

    eqn_param.output.primary = &db;
    db_func(&eqn_param);

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, 0, 0, S3);
    dgamma_func(&eqn_param);

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, 0, 0, S3);
    dbeta_func(&eqn_param);

    b = (db * mean[s2] - ds) * a * a * a * scale;
    c = -b * mean[s2] - db * a * scale;

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, din, 0, s2, 0, S2, S3);
    din_func(&eqn_param);
  }
}

LIBXSMM_INLINE
void tpp_layernorm_bwd_bf16(long S1, long S2, long S3, libxsmm_bfloat16 *pdout, libxsmm_bfloat16 *pinp, float *mean, float *var, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pdin, float *pdgamma, float *pdbeta,
    libxsmm_meqn_function dgamma_func, libxsmm_meqn_function dbeta_func, libxsmm_meqn_function db_func, libxsmm_meqn_function ds_func, libxsmm_meqn_function din_func) {
  int s2;
  float a, b, c, db, ds;
  const float scale = 1.0f / ((float)S1*S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, din, pdin, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, dout, pdout, S2, S3);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);

  libxsmm_meqn_param eqn_param;
  libxsmm_matrix_arg arg_array[8];
  eqn_param.inputs = arg_array;

  arg_array[1].primary = &a;
  arg_array[2].primary = &b;
  arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, 0, 0, S3);
  arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, 0, 0, S3);
  arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, 0, 0, S3);
  arg_array[7].primary = &c;

  for (s2 = 0; s2 < S2; s2++) {
    a = var[s2];
    b = -a*mean[s2];
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    arg_array[3].primary = &LIBXSMM_VLA_ACCESS(3, dout, 0, s2, 0, S2, S3);

    eqn_param.output.primary = &ds;
    ds_func(&eqn_param);

    eqn_param.output.primary = &db;
    db_func(&eqn_param);

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, 0, 0, S3);
    dgamma_func(&eqn_param);

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, 0, 0, S3);
    dbeta_func(&eqn_param);

    b = (db * mean[s2] - ds) * a * a * a * scale;
    c = -b * mean[s2] - db * a * scale;

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, din, 0, s2, 0, S2, S3);
    din_func(&eqn_param);
  }
}

int main( int argc, char* argv[] ) {
  int ret = EXIT_SUCCESS;
  double error_bound = 0.0002;
  libxsmm_blasint my_eqn0, my_eqn1, my_eqn2, my_eqn3, my_eqn4, my_eqn5;
  libxsmm_meqn_function func0, func1, func2, func3, func4, func5;
  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltw_unary_shape reduce_rows_shape, reduce_cols_shape;
  libxsmm_meltwfunction_unary reduce_rows_kernel, reduce_cols_kernel;

  const float eps = FLT_EPSILON;
  libxsmm_blasint i, it, ld, tmp_ld, tmp_ld2;
  libxsmm_timer_tickint l_start, l_end;
  double l_total = 0, l_total2 = 0;
  double t_vec = 0, t_tpp = 0;
  libxsmm_matdiff_info norms_out;
  float *inp, *out, *dinp, *dout, *eqn_dinp, *eqn_dout, *dbeta, *eqn_dbeta, *dgamma, *eqn_dgamma, *eqn_out, *gamma, *beta, *cache_fl, *mean, *var;
  libxsmm_bfloat16 *bf16_inp, *bf16_out, *bf16_dinp, *bf16_dout, *bf16_eqn_dinp, *bf16_eqn_dout, *bf16_gamma, *bf16_beta, *bf16_eqn_out;
  const char *matdiff_env;
#if defined(USE_SUM)
  float sum = 0.0;
#endif
  int S1 = 64;
  int S2 = 64;
  int S3 = 64;
  int iters = 100;
  int datatype_mode = 0;
  int pass = FWD_BWD_LNORM;
  libxsmm_datatype in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype out_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_meqn_arg_metadata arg_metadata;
  libxsmm_meqn_op_metadata  op_metadata;
  libxsmm_meqn_arg_shape          arg_shape_in, arg_shape_out;
  libxsmm_matrix_arg_attributes   arg_singular_attr = libxsmm_create_matrix_arg_attributes( LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);

  if ( argc > 1 ) S1 = atoi(argv[1]);
  if ( argc > 2 ) S2 = atoi(argv[2]);
  if ( argc > 3 ) S3 = atoi(argv[3]);
  if ( argc > 4 ) datatype_mode = atoi(argv[4]);
  if ( argc > 5 ) pass = atoi(argv[5]);
  if ( argc > 6 ) iters = atoi(argv[6]);

  if (datatype_mode == 0) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_F32;
    error_bound = LIBXSMM_MAX(0.007, error_bound);
  } else if (datatype_mode == 1) {
    in_dt = LIBXSMM_DATATYPE_BF16;
    out_dt = LIBXSMM_DATATYPE_BF16;
    error_bound = LIBXSMM_MAX(0.007, error_bound);
  } else {
    printf("ERROR: Supporting only FP32 and BF16 precisions...\n");
  }

  /* eventually amend LIBXSMM_MATDIFF output with error bound */
  matdiff_env = getenv("LIBXSMM_MATDIFF");
  if (NULL != matdiff_env) {
    static char matdiff_ext[1024];
    const int nchars = LIBXSMM_SNPRINTF(matdiff_ext, sizeof(matdiff_ext),
      "LIBXSMM_MATDIFF=%s %.17g", matdiff_env, error_bound);
    if (0 < nchars && nchars < (int)sizeof(matdiff_ext)) {
      LIBXSMM_EXPECT(EXIT_SUCCESS == LIBXSMM_PUTENV(matdiff_ext));
    }
  }

  inp = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  out = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  dout = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S3,   2097152);
  dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S3,   2097152);
  eqn_dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  eqn_dout = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  eqn_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S3,   2097152);
  eqn_dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S3,   2097152);
  gamma = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S3,   2097152);
  beta = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S3,   2097152);
  mean = (float*) libxsmm_aligned_malloc( sizeof(float)*S2,   2097152);
  var = (float*) libxsmm_aligned_malloc( sizeof(float)*S2,   2097152);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*S2*S3,   2097152);
  cache_fl  = (float*) libxsmm_aligned_malloc( sizeof(float)*1024*1024,   2097152);

  bf16_inp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);
  bf16_out = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);
  bf16_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);
  bf16_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);
  bf16_eqn_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);
  bf16_eqn_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);
  bf16_gamma = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S3,   2097152);
  bf16_beta = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S3,   2097152);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*S1*S2*S3,   2097152);

  /* Initializing arrays */
  for ( i = 0; i < S1*S2*S3; ++i ) {
    inp[i] = (float)libxsmm_rng_f64();
    out[i] = (float)libxsmm_rng_f64();
    eqn_out[i] = out[i];
    dinp[i] = (float)libxsmm_rng_f64();
    dout[i] = (float)libxsmm_rng_f64();
    eqn_dinp[i] = dinp[i];
    eqn_dout[i] = dout[i];
    libxsmm_rne_convert_fp32_bf16( &inp[i], &bf16_inp[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &out[i], &bf16_out[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &eqn_out[i], &bf16_eqn_out[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &dout[i], &bf16_dout[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &eqn_dout[i], &bf16_eqn_dout[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &dinp[i], &bf16_dinp[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &eqn_dinp[i], &bf16_eqn_dinp[i], 1 );
  }

  for ( i = 0; i < S1*S3; ++i ) {
    gamma[i] = (float)libxsmm_rng_f64();
    beta[i] = (float)libxsmm_rng_f64();
    dbeta[i] = (float)libxsmm_rng_f64();
    dgamma[i] = (float)libxsmm_rng_f64();
    eqn_dbeta[i] = dbeta[i];
    eqn_dgamma[i] = dgamma[i];
    libxsmm_rne_convert_fp32_bf16( &gamma[i], &bf16_gamma[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &beta[i], &bf16_beta[i], 1 );
  }

  for (i = 0; i < 1024 * 1024; i++ ) {
    cache_fl[i] = (float)libxsmm_rng_f64();
  }

  if ((pass & FWD_LNORM) > 0) {
    /* TPPs for reducing X and X2 */
    ld = S2*S3;
    tmp_ld = S3;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
    jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
    reduce_cols_shape = libxsmm_create_meltw_unary_shape( S3, S1, ld, tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    reduce_cols_kernel = libxsmm_dispatch_meltw_unary( unary_type, reduce_cols_shape, jit_reduce_flags );
    libxsmm_get_kernel_info((const void*) reduce_cols_kernel, &info);
    is_reference_kernel = info.is_reference_kernel;
    if ( reduce_cols_kernel == NULL ) {
      fprintf( stderr, "JIT for reduce_cols_kernel failed. Bailing...!\n");
      exit(-1);
    }
    ld = S3;
    tmp_ld = 1;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
    jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
    reduce_rows_shape = libxsmm_create_meltw_unary_shape( S3, 1, ld, tmp_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    reduce_rows_kernel = libxsmm_dispatch_meltw_unary( unary_type, reduce_rows_shape, jit_reduce_flags );
    libxsmm_get_kernel_info((const void*) reduce_rows_kernel, &info);
    is_reference_kernel = info.is_reference_kernel;
    if ( reduce_rows_kernel == NULL ) {
      fprintf( stderr, "JIT for reduce_rows_kernel failed. Bailing...!\n");
      exit(-1);
    }
    /* TPP for scaling */
    ld = S2*S3;
    tmp_ld = 1;
    tmp_ld2 = S3;
    my_eqn0 = libxsmm_meqn_create();
    op_metadata   = libxsmm_create_meqn_op_metadata(my_eqn0, -1);
    libxsmm_meqn_push_back_ternary_op( op_metadata, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT );
    libxsmm_meqn_push_back_ternary_op( op_metadata, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32,
      (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT) );
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 0);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, 1, tmp_ld, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 1);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, 1, tmp_ld, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 2);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld2, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 3);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld2, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 4);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, ld, out_dt );
    func0 = libxsmm_dispatch_meqn( my_eqn0, arg_shape_out );
    libxsmm_get_kernel_info((const void*) func0, &info);
    is_reference_kernel = info.is_reference_kernel;
    if ( func0 == NULL ) {
      fprintf( stderr, "JIT for func0 failed. Bailing...!\n");
      exit(-1);
    }

    /* Check correctness */
    if (datatype_mode == 0) {
      vectorized_layernorm_fwd_fp32(S1, S2, S3, inp, gamma, beta, mean, var, out, eps);
      tpp_layernorm_fwd_fp32(S1, S2, S3, inp, gamma, beta, mean, var, eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel);
    } else if (datatype_mode == 1) {
      vectorized_layernorm_fwd_bf16(S1, S2, S3, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_out, eps);
      tpp_layernorm_fwd_bf16(S1, S2, S3, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel);
      for ( i = 0; i < S1*S2*S3; ++i ) {
        out[i] = upconvert_bf16(bf16_out[i]);
        eqn_out[i] = upconvert_bf16(bf16_eqn_out[i]);
      }
    }

    /* compare */
    printf("############################################\n");
    if (datatype_mode == 0) {
      printf("# Correctness FP32 FWD Layernorm - Output  #\n");
    } else {
      printf("# Correctness BF16 FWD Layernorm - Output  #\n");
    }
    printf("############################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, S1*S2*S3, 1, out, eqn_out, 0, 0);
    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    if ( norms_out.normf_rel > error_bound ) {
      ret = EXIT_FAILURE;
    }

    if (iters > 0) {
      if (datatype_mode == 0) {
#if defined(USE_SUM)
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i];
        }
#endif
        vectorized_layernorm_fwd_fp32(S1, S2, S3, inp, gamma, beta, mean, var, out, eps);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          vectorized_layernorm_fwd_fp32(S1, S2, S3, inp, gamma, beta, mean, var, out, eps);
        }
        l_end = libxsmm_timer_tick();
        l_total = libxsmm_timer_duration(l_start, l_end);
        printf("Intrinsics layernorm time FWD = %.5g\n", l_total);
#if defined(USE_SUM)
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i] + (float)l_total;
        }
#endif
        tpp_layernorm_fwd_fp32(S1, S2, S3, inp, gamma, beta, mean, var, eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          tpp_layernorm_fwd_fp32(S1, S2, S3, inp, gamma, beta, mean, var, eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel);
        }
        l_end = libxsmm_timer_tick();
        l_total2 = libxsmm_timer_duration(l_start, l_end);
        printf("TPP layernorm time FWD = %.5g\n", l_total2);
        if (0 < l_total2) printf("Speedup FWD is = %.5g\n", l_total/l_total2);
      } else if (datatype_mode == 1) {
#if defined(USE_SUM)
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i];
        }
#endif
        vectorized_layernorm_fwd_bf16(S1, S2, S3, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_out, eps);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          vectorized_layernorm_fwd_bf16(S1, S2, S3, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_out, eps);
        }
        l_end = libxsmm_timer_tick();
        l_total = libxsmm_timer_duration(l_start, l_end);
        printf("Intrinsics layernorm time FWD = %.5g\n", l_total);
#if defined(USE_SUM)
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i] + (float)l_total;
        }
#endif
        tpp_layernorm_fwd_bf16(S1, S2, S3, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          tpp_layernorm_fwd_bf16(S1, S2, S3, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel);
        }
        l_end = libxsmm_timer_tick();
        l_total2 = libxsmm_timer_duration(l_start, l_end);
        printf("TPP layernorm time FWD = %.5g\n", l_total2);
        if (0 < l_total2) printf("Speedup FWD is = %.5g\n", l_total/l_total2);
      }
    }
  }

  t_tpp = l_total2;
  t_vec = l_total;

  if ((pass & BWD_LNORM) > 0) {
    /* Create MatEq for bwd layernorm */
    tmp_ld = S3;
    ld = S2*S3;
    tmp_ld2 = 1;

    for ( i = 0; i < S2; ++i ) {
      mean[i] = (float)libxsmm_rng_f64();
      var[i] = (float)libxsmm_rng_f64();
    }

    /* dgamma function */
    my_eqn1 = libxsmm_meqn_create();
    op_metadata   = libxsmm_create_meqn_op_metadata(my_eqn1, -1);
    libxsmm_meqn_push_back_ternary_op( op_metadata, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT );
    libxsmm_meqn_push_back_ternary_op( op_metadata, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32,
      (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT) );
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn1, 0);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, 1, 1, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn1, 1);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, 1, 1, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn1, 2);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn1, 3);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn1, 4);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld, LIBXSMM_DATATYPE_F32 );
    func1 = libxsmm_dispatch_meqn( my_eqn1, arg_shape_out );
    libxsmm_get_kernel_info((const void*) func1, &info);
    is_reference_kernel = info.is_reference_kernel;

    if ( func1 == NULL ) {
      fprintf( stderr, "JIT for func1 failed. Bailing...!\n");
      exit(-1);
    }

    /* dbeta function */
    my_eqn2 = libxsmm_meqn_create();
    op_metadata   = libxsmm_create_meqn_op_metadata(my_eqn2, -1);
    libxsmm_meqn_push_back_binary_op( op_metadata, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE );
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn2, 3);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn2, 5);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld, LIBXSMM_DATATYPE_F32 );
    func2 = libxsmm_dispatch_meqn( my_eqn2, arg_shape_out );
    libxsmm_get_kernel_info((const void*) func2, &info);
    is_reference_kernel = info.is_reference_kernel;

    if ( func2 == NULL ) {
      fprintf( stderr, "JIT for func2 failed. Bailing...!\n");
      exit(-1);
    }

    /* db equation */
    my_eqn3 = libxsmm_meqn_create();
    op_metadata   = libxsmm_create_meqn_op_metadata(my_eqn3, -1);
    libxsmm_meqn_push_back_binary_op( op_metadata, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE );
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn3, 3);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn3, 6);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_out = libxsmm_create_meqn_arg_shape( 1, 1, tmp_ld2, LIBXSMM_DATATYPE_F32 );
    func3 = libxsmm_dispatch_meqn( my_eqn3, arg_shape_out );
    libxsmm_get_kernel_info((const void*) func3, &info);
    is_reference_kernel = info.is_reference_kernel;
    if ( func3 == NULL ) {
      fprintf( stderr, "JIT for func3 failed. Bailing...!\n");
      exit(-1);
    }

    /* ds equation */
    my_eqn4 = libxsmm_meqn_create();
    op_metadata   = libxsmm_create_meqn_op_metadata(my_eqn4, -1);
    libxsmm_meqn_push_back_binary_op( op_metadata, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE );
    libxsmm_meqn_push_back_binary_op( op_metadata, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE );
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn4, 3);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn4, 6);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn4, 0);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_out = libxsmm_create_meqn_arg_shape( 1, 1, tmp_ld2, LIBXSMM_DATATYPE_F32 );
    func4 = libxsmm_dispatch_meqn( my_eqn4, arg_shape_out );
    libxsmm_get_kernel_info((const void*) func4, &info);
    is_reference_kernel = info.is_reference_kernel;
    if ( func4 == NULL ) {
      fprintf( stderr, "JIT for func4 failed. Bailing...!\n");
      exit(-1);
    }

    /* din equation */
    my_eqn5 = libxsmm_meqn_create();
    op_metadata   = libxsmm_create_meqn_op_metadata(my_eqn5, -1);
    libxsmm_meqn_push_back_ternary_op( op_metadata, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT );
    libxsmm_meqn_push_back_binary_op( op_metadata, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1 );
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn5, 6);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, 1, 1, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn5, 1);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn5, 3);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    libxsmm_meqn_push_back_ternary_op( op_metadata, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32,
      (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT) );
    arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn5, 0);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, 1, 1, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn5, 2);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, 1, 1, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn5, 7);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
    func5 = libxsmm_dispatch_meqn( my_eqn5, arg_shape_out );
    libxsmm_get_kernel_info((const void*) func5, &info);
    is_reference_kernel = info.is_reference_kernel;
    if ( func5 == NULL ) {
      fprintf( stderr, "JIT for func5 failed. Bailing...!\n");
      exit(-1);
    }
    if (datatype_mode == 0) {
      vectorized_layernorm_bwd_fp32(S1, S2, S3, dout, inp, mean, var, gamma, dinp, dgamma, dbeta);
      tpp_layernorm_bwd_fp32(S1, S2, S3, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func1, func2, func3, func4, func5);
    } else if (datatype_mode == 1) {
      vectorized_layernorm_bwd_bf16(S1, S2, S3, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_dinp, dgamma, dbeta);
      tpp_layernorm_bwd_bf16(S1, S2, S3, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func1, func2, func3, func4, func5);
      for ( i = 0; i < S1*S2*S3; ++i ) {
        dinp[i] = upconvert_bf16(bf16_dinp[i]);
        eqn_dinp[i] = upconvert_bf16(bf16_eqn_dinp[i]);
      }
    }

    /* compare */
    printf("############################################\n");
    if (datatype_mode == 0) {
      printf("# Correctness FP32 BWD Layernorm - Dinput  #\n");
    } else {
      printf("# Correctness BF16 BWD Layernorm - Dinput  #\n");
    }
    printf("############################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, S1*S2*S3, 1, dinp, eqn_dinp, 0, 0);
    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    if ( norms_out.normf_rel > error_bound ) {
      ret = EXIT_FAILURE;
    }

    printf("###########################################\n");
    if (datatype_mode == 0) {
      printf("# Correctness FP32 BWD Layernorm - Dbeta  #\n");
    } else {
      printf("# Correctness BF16 BWD Layernorm - Dbeta  #\n");
    }
    printf("###########################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, S1*S3, 1, dbeta, eqn_dbeta, 0, 0);
    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    if ( norms_out.normf_rel > error_bound ) {
      ret = EXIT_FAILURE;
    }

    printf("############################################\n");
    if (datatype_mode == 0) {
      printf("# Correctness FP32 BWD Layernorm - Dgamma  #\n");
    } else {
      printf("# Correctness BF16 BWD Layernorm - Dgamma #\n");
    }
    printf("############################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, S1*S3, 1, dgamma, eqn_dgamma, 0, 0);
    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    if ( norms_out.normf_rel > error_bound ) {
      ret = EXIT_FAILURE;
    }

    if (iters > 0 ) {
      if (datatype_mode == 0) {
#if defined(USE_SUM)
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i];
        }
#endif
        vectorized_layernorm_bwd_fp32(S1, S2, S3, dout, inp, mean, var, gamma, dinp, dgamma, dbeta);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          vectorized_layernorm_bwd_fp32(S1, S2, S3, dout, inp, mean, var, gamma, dinp, dgamma, dbeta);
        }
        l_end = libxsmm_timer_tick();
        l_total = libxsmm_timer_duration(l_start, l_end);
        printf("Intrinsics layernorm time BWD = %.5g\n", l_total);
#if defined(USE_SUM)
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i] + (float)l_total;
        }
#endif
        tpp_layernorm_bwd_fp32(S1, S2, S3, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func1, func2, func3, func4, func5);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          tpp_layernorm_bwd_fp32(S1, S2, S3, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func1, func2, func3, func4, func5);
        }
        l_end = libxsmm_timer_tick();
        l_total2 = libxsmm_timer_duration(l_start, l_end);
        printf("TPP layernorm time BWD = %.5g\n", l_total2);
        if (0 < l_total2) printf("Speedup BWD is = %.5g\n", l_total/l_total2);
      } else if (datatype_mode == 1) {
#if defined(USE_SUM)
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i];
        }
#endif
        vectorized_layernorm_bwd_bf16(S1, S2, S3, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_dinp, dgamma, dbeta);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          vectorized_layernorm_bwd_bf16(S1, S2, S3, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_dinp, dgamma, dbeta);
        }
        l_end = libxsmm_timer_tick();
        l_total = libxsmm_timer_duration(l_start, l_end);
        printf("Intrinsics layernorm time BWD = %.5g\n", l_total);
#if defined(USE_SUM)
        for (i = 0; i < 1024 * 1024; i++ ) {
          sum += cache_fl[i] + (float)l_total;
        }
#endif
        tpp_layernorm_bwd_bf16(S1, S2, S3, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func1, func2, func3, func4, func5);
        l_start = libxsmm_timer_tick();
        for (it = 0; it < iters; it++) {
          tpp_layernorm_bwd_bf16(S1, S2, S3, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func1, func2, func3, func4, func5);
        }
        l_end = libxsmm_timer_tick();
        l_total2 = libxsmm_timer_duration(l_start, l_end);
        printf("TPP layernorm time BWD = %.5g\n", l_total2);
        if (0 < l_total2) printf("Speedup BWD is = %.5g\n", l_total/l_total2);
      }
    }
  }
#if defined(USE_SUM)
  printf("Running sum is %.5f\n", sum);
#endif
  t_tpp += l_total2;
  t_vec += l_total;

  if (iters > 0) {
    printf("\n\n=================================\n");
    printf("Total Speedup via TPP Matrix equation is %.5g\n", t_vec/t_tpp);
    printf("=================================\n");
  }

  libxsmm_free(inp);
  libxsmm_free(out);
  libxsmm_free(dinp);
  libxsmm_free(dout);
  libxsmm_free(eqn_dinp);
  libxsmm_free(eqn_dout);
  libxsmm_free(bf16_dinp);
  libxsmm_free(bf16_dout);
  libxsmm_free(bf16_eqn_dinp);
  libxsmm_free(bf16_eqn_dout);
  libxsmm_free(dgamma);
  libxsmm_free(dbeta);
  libxsmm_free(eqn_dgamma);
  libxsmm_free(eqn_dbeta);
  libxsmm_free(mean);
  libxsmm_free(var);
  libxsmm_free(gamma);
  libxsmm_free(beta);
  libxsmm_free(eqn_out);
  libxsmm_free(bf16_inp);
  libxsmm_free(bf16_out);
  libxsmm_free(bf16_gamma);
  libxsmm_free(bf16_beta);
  libxsmm_free(bf16_eqn_out);
  libxsmm_free(cache_fl);

  ret = (ret == EXIT_SUCCESS) ? libxsmm_return_success_code(is_reference_kernel) : ret;
  return ret;
}
