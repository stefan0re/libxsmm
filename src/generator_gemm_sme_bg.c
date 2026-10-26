/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Stefan Remke (Univ. Jena)
******************************************************************************/
/*
 * SGEMM on SME (Apple M4) with the C traffic and the transposition of op(B) hidden in the FMOPA stream.
 * C is computed in blocks of up to 4 ZA tiles, ZA_t[j][i] += op(B)[k,j] * op(A)[i,k]; operands are
 * loaded two k-steps ahead.
 *  - dump mode (larger k): a 32x32 block accumulates A*B, ZA is read out into a dump buffer and zeroed;
 *    C of the block (C = dump + beta*C) is updated in the background of the next block's k-loop
 *  - background transposition (dump mode, op(B) = B): the next 32-column panel of op(B) is transposed
 *    (ZIPs) in the background, placed by a list scheduler; TT is computed as C^T = B A
 *  - ZA mode (smaller k): 16x64 .. 64x16 blocks in row strips, C added into ZA inside the k-loop
 *  - a transposed op(A) is packed once up front through ZA (in row groups if needed)
 */
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_sme_bg.h"
#include "generator_common.h"

#define LIBXSMM_SME_BG_VL          16   /* fp32 lanes per vector */
#define LIBXSMM_SME_BG_DIST         2   /* operand load-to-use distance in k-steps */
#define LIBXSMM_SME_BG_KUNROLL     12   /* k-steps per iteration of the rolled k-loop */
#define LIBXSMM_SME_BG_MINK       128   /* dump mode from this k on */
#define LIBXSMM_SME_BG_START        1   /* k-loop iterations before the background C update starts */
#define LIBXSMM_SME_BG_SDIST        1   /* k-steps from the last fadd of a column to its store */
#define LIBXSMM_SME_BG_TR_START     8   /* first k-step of the background transposition in a block */
#define LIBXSMM_SME_BG_CU_START    16   /* first k-step of the scheduled background C update */
#define LIBXSMM_SME_BG_D_ZIP        2   /* scheduler distances in k-steps */
#define LIBXSMM_SME_BG_D_FADD       2
#define LIBXSMM_SME_BG_MAX_G        8   /* k-groups (16 k) transposed per block at most */
#define LIBXSMM_SME_BG_MAX_WS  393216   /* stack workspace limit in bytes */
#define LIBXSMM_SME_BG_WS_GAP    4096   /* distance of the workspace from the register save area: SME loads of
                                         * lines near the ones the core writes every call (register saves) are
                                         * slow, e.g. 32^3 NN: 86 ns with 256 bytes, 75 ns from 4 KB on */
#define LIBXSMM_SME_BG_MAX_CODE 122880  /* estimated code size limit in bytes (JIT buffer: 128 KB) */
#define LIBXSMM_SME_BG_SAVE    0xfef    /* callee-saved registers: d8-d15, x19-x30 */

#define LIBXSMM_SME_BG_X(N)    (LIBXSMM_AARCH64_GP_REG_X0 + (N))
#define LIBXSMM_SME_BG_XZR     LIBXSMM_AARCH64_GP_REG_XZR

/* kinds of predecessor pointers for the background C update */
#define LIBXSMM_SME_BG_PREV_WITHIN 0   /* previous block of the same panel: x23 - one block, column stride ldc */
#define LIBXSMM_SME_BG_PREV_PANEL  1   /* last block of the previous panel (or the dummy area): x7, x30 */

typedef struct libxsmm_sme_bg_block {
  unsigned int mb, nb, tm, tn;        /* valid rows/columns, tiles in m/n */
} libxsmm_sme_bg_block;

typedef struct libxsmm_sme_bg_prev {
  unsigned int mb, nb, kind;          /* rows/columns of the predecessor C block (C layout) */
} libxsmm_sme_bg_prev;

typedef struct libxsmm_sme_bg_op {    /* background instruction(s) for the list scheduler */
  unsigned int w_off, w_cnt;          /* instruction words */
  unsigned int d_off, d_cnt;          /* dependencies: pairs (op, minimum distance in k-steps) */
  int vec;                            /* uses the vector (shuffle/FP) pipe */
  int k;                              /* scheduled k-step, -1: not placed */
} libxsmm_sme_bg_op;

typedef struct libxsmm_sme_bg_ctx {
  libxsmm_generated_code* code;
  libxsmm_loop_label_tracker lt;
  /* the computed problem (TT: C^T = B A; m: rows of the current row group) */
  unsigned int m, n, k, lda, ldb, ldc, ta, tb, beta0, swap;
  unsigned int mpad, npad, kpad, pa_bytes, pb_bytes, a_packed, b_packed;
  /* blocking: full 32-row blocks + edge row block, full 32-column panels + edge panel */
  unsigned int fcnt, medge, nfull, nedge, last_i0, last_mb;
  /* background transposition: k-groups per block, predicated k-groups */
  unsigned int bgt, G, pred;
  /* workspace offsets (x27) */
  unsigned int pb0, pb1, pbe, dump_off, dummy_off, scr_off, ws_bytes;
  /* operand strides per k-step: register, or immediate if the register is UNDEF */
  unsigned int a_stride_reg, b_stride_reg, a_stride_imm, b_stride_imm;
  /* scheduler */
  unsigned int d_load, d_mem, max_extra, op_words;
  libxsmm_sme_bg_op* ops;
  unsigned int nops, ops_cap;
  unsigned int* words;
  unsigned int nwords, words_cap;
  unsigned int* deps;
  unsigned int ndeps, deps_cap;
  unsigned char* used_n;
  unsigned char* used_v;
  libxsmm_generated_code opcode;
  unsigned int scr_last[2][16], scr_last_n[2];
  unsigned int plast[2][32], plast_n[2];
} libxsmm_sme_bg_ctx;

LIBXSMM_API_INLINE unsigned int libxsmm_sme_bg_cdiv( unsigned int a, unsigned int b ) { return (a + b - 1) / b; }
LIBXSMM_API_INLINE unsigned int libxsmm_sme_bg_rup( unsigned int a, unsigned int b ) { return libxsmm_sme_bg_cdiv(a, b) * b; }

/* ------------------------------------------------------------------------------------ emitters */
LIBXSMM_API_INLINE void libxsmm_sme_bg_mov( libxsmm_generated_code* c, unsigned int dst, unsigned int src ) {
  if ( dst != src ) {
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( c, LIBXSMM_AARCH64_INSTR_GP_ORR_SR, LIBXSMM_SME_BG_XZR, src, dst,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  }
}

/* dst = src + imm; src/dst may be SP if |imm| < 2^24 */
LIBXSMM_API_INLINE void libxsmm_sme_bg_add_imm( libxsmm_generated_code* c, unsigned int dst, unsigned int src, long long imm ) {
  const unsigned long long l_v = (unsigned long long)(imm < 0 ? -imm : imm);
  if ( 0 == imm ) {
    libxsmm_sme_bg_mov( c, dst, src );
  } else if ( l_v < (1ULL << 24) ) {
    const unsigned int l_instr = (imm < 0) ? LIBXSMM_AARCH64_INSTR_GP_SUB_I : LIBXSMM_AARCH64_INSTR_GP_ADD_I;
    unsigned int l_src = src;
    if ( 0 != (l_v >> 12) ) {
      libxsmm_aarch64_instruction_alu_compute_imm12( c, l_instr, l_src, dst, (unsigned int)(l_v >> 12), 1 );
      l_src = dst;
    }
    if ( 0 != (l_v & 0xfff) ) {
      libxsmm_aarch64_instruction_alu_compute_imm12( c, l_instr, l_src, dst, (unsigned int)(l_v & 0xfff), 0 );
    }
  } else {
    const unsigned int l_tmp = (dst != LIBXSMM_SME_BG_X(17) && src != LIBXSMM_SME_BG_X(17)) ? LIBXSMM_SME_BG_X(17) : LIBXSMM_SME_BG_X(16);
    if ( l_tmp == src ) {
      LIBXSMM_HANDLE_ERROR( c, LIBXSMM_ERR_ILLEGAL_REGNUM );
      return;
    }
    libxsmm_aarch64_instruction_alu_set_imm64( c, l_tmp, l_v );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( c, (imm < 0) ? LIBXSMM_AARCH64_INSTR_GP_SUB_SR : LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                         src, l_tmp, dst, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  }
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_alu_reg( libxsmm_generated_code* c, unsigned int instr, unsigned int dst, unsigned int src0,
                                                unsigned int src1, unsigned int lsl ) {
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( c, instr, src0, src1, dst, lsl, LIBXSMM_AARCH64_SHIFTMODE_LSL );
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_wset( libxsmm_generated_code* c ) {
  unsigned int l_i;   /* w12..w15 = 0, 4, 8, 12: 4-slice groups of the 16x16 tiles */
  for ( l_i = 0; l_i < 4; ++l_i ) {
    libxsmm_aarch64_instruction_alu_move_imm16( c, LIBXSMM_AARCH64_INSTR_GP_MOVZ, LIBXSMM_AARCH64_GP_REG_W12 + l_i, 0, 4 * l_i );
  }
}

/* ld1w/st1w of 1, 2 or 4 vectors (consecutive or strided registers), offset in vectors */
LIBXSMM_API_INLINE void libxsmm_sme_bg_ldst( libxsmm_generated_code* c, unsigned int store, unsigned int z, unsigned int nregs,
                                             unsigned int strided, unsigned int xn, int imm_vl, unsigned int pred ) {
  if ( 1 == nregs ) {
    libxsmm_aarch64_instruction_sve_move( c, store ? LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF : LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                          xn, LIBXSMM_AARCH64_GP_REG_UNDEF, imm_vl, z, pred );
  } else {
    unsigned int l_instr;
    if ( 0 != strided ) {
      l_instr = (2 == nregs) ? (store ? LIBXSMM_AARCH64_INSTR_SME_ST1W_2 : LIBXSMM_AARCH64_INSTR_SME_LD1W_2)
                             : (store ? LIBXSMM_AARCH64_INSTR_SME_ST1W_4 : LIBXSMM_AARCH64_INSTR_SME_LD1W_4);
    } else {
      l_instr = (2 == nregs) ? (store ? LIBXSMM_AARCH64_INSTR_SVE2_ST1W_2 : LIBXSMM_AARCH64_INSTR_SVE2_LD1W_2)
                             : (store ? LIBXSMM_AARCH64_INSTR_SVE2_ST1W_4 : LIBXSMM_AARCH64_INSTR_SVE2_LD1W_4);
    }
    libxsmm_aarch64_instruction_sme_mov( c, l_instr, z, xn, (unsigned int)(imm_vl / (int)nregs), pred );
  }
}

/* mova of 4 vectors: tile slices -> z..z+3 (horizontal or vertical) or z..z+3 -> horizontal slices */
LIBXSMM_API_INLINE void libxsmm_sme_bg_mova( libxsmm_generated_code* c, unsigned int to_tile, unsigned int vertical, unsigned int tile,
                                             unsigned int widx, unsigned int z ) {
  const unsigned int l_instr = (0 != to_tile) ? LIBXSMM_AARCH64_INSTR_SME_MOVA_H_VECTOR_TO_TILE
                             : ((0 != vertical) ? LIBXSMM_AARCH64_INSTR_SME_MOVA_V_TILE_TO_VECTOR : LIBXSMM_AARCH64_INSTR_SME_MOVA_H_TILE_TO_VECTOR);
  libxsmm_aarch64_instruction_sme_mova( c, l_instr, tile, LIBXSMM_AARCH64_GP_REG_W12 + widx, z / 4 );
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_fmopa( libxsmm_generated_code* c, unsigned int tile, unsigned int zb, unsigned int za ) {
  libxsmm_aarch64_instruction_sme_compute( c, LIBXSMM_AARCH64_INSTR_SME_FMOPA_SP, tile, zb, za, LIBXSMM_AARCH64_SVE_REG_P0, LIBXSMM_AARCH64_SVE_REG_P0 );
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_fadd( libxsmm_generated_code* c, unsigned int zd, unsigned int zn, unsigned int zm ) {
  libxsmm_aarch64_instruction_sve_compute( c, LIBXSMM_AARCH64_INSTR_SVE_FADD_V, zn, zm, 0, zd, LIBXSMM_AARCH64_SVE_REG_P0, LIBXSMM_AARCH64_SVE_TYPE_S );
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_zeroz( libxsmm_generated_code* c, unsigned int z ) {
  libxsmm_aarch64_instruction_sve_compute( c, LIBXSMM_AARCH64_INSTR_SVE_EOR_V, z, z, 0, z, LIBXSMM_AARCH64_SVE_REG_P0, LIBXSMM_AARCH64_SVE_TYPE_D );
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_movz( libxsmm_generated_code* c, unsigned int zd, unsigned int zn ) {
  libxsmm_aarch64_instruction_sve_compute( c, LIBXSMM_AARCH64_INSTR_SVE_ORR_V, zn, zn, 0, zd, LIBXSMM_AARCH64_SVE_REG_P0, LIBXSMM_AARCH64_SVE_TYPE_D );
}

/* whilelt pn.s, xn, xm, vlx2/vlx4 (counter) or whilelt p.s, xn, xm */
LIBXSMM_API_INLINE void libxsmm_sme_bg_whilelt( libxsmm_generated_code* c, unsigned int pred, unsigned int nvec, unsigned int xn, unsigned int xm ) {
  if ( 1 == nvec ) {
    libxsmm_aarch64_instruction_sve_pcompute( c, LIBXSMM_AARCH64_INSTR_SVE_WHILELT, pred, xn, LIBXSMM_AARCH64_GP_WIDTH_X, xm,
                                              LIBXSMM_AARCH64_SVE_PATTERN_ALL, LIBXSMM_AARCH64_SVE_TYPE_S );
  } else {
    libxsmm_aarch64_instruction_set_predication_as_counter( c, LIBXSMM_AARCH64_INSTR_SVE2_WHILELT, pred, xn, (4 == nvec) ? 1 : 0, xm, 2 );
  }
}

/* predicate for nlanes active lanes of an nvec-vector access; all lanes: p0 / pn8 */
LIBXSMM_API_INLINE unsigned int libxsmm_sme_bg_pred( libxsmm_generated_code* c, unsigned int nlanes, unsigned int nvec, unsigned int pred ) {
  if ( nlanes >= LIBXSMM_SME_BG_VL * nvec ) {
    return (1 == nvec) ? LIBXSMM_AARCH64_SVE_REG_P0 : LIBXSMM_AARCH64_SVE_REG_P8;
  }
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(16), nlanes );
  libxsmm_sme_bg_whilelt( c, pred, nvec, LIBXSMM_SME_BG_XZR, LIBXSMM_SME_BG_X(16) );
  return pred;
}

/* ------------------------------------------------------------------------------------ op pool */
LIBXSMM_API_INLINE unsigned int libxsmm_sme_bg_op_begin( libxsmm_sme_bg_ctx* x ) {
  libxsmm_sme_bg_op* l_op;
  if ( x->nops >= x->ops_cap || x->words_cap - x->nwords < 16 ) {
    LIBXSMM_HANDLE_ERROR( x->code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
    return 0;
  }
  l_op = &x->ops[x->nops];
  l_op->w_off = x->nwords;
  l_op->w_cnt = 0;
  l_op->d_off = x->ndeps;
  l_op->d_cnt = 0;
  l_op->vec = 0;
  l_op->k = -1;
  x->opcode.generated_code = x->words + x->nwords;
  x->opcode.buffer_size = 4 * (x->words_cap - x->nwords);
  x->opcode.code_size = 0;
  return x->nops++;
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_op_dep( libxsmm_sme_bg_ctx* x, unsigned int op, unsigned int dep, unsigned int dist ) {
  if ( x->ndeps + 2 > x->deps_cap ) {
    LIBXSMM_HANDLE_ERROR( x->code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
    return;
  }
  x->deps[x->ndeps++] = dep;
  x->deps[x->ndeps++] = dist;
  x->ops[op].d_cnt++;
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_op_end( libxsmm_sme_bg_ctx* x, unsigned int op, int vec ) {
  if ( 0 != x->opcode.last_error ) {
    LIBXSMM_HANDLE_ERROR( x->code, x->opcode.last_error );
    return;
  }
  x->ops[op].w_cnt = x->opcode.code_size / 4;
  x->ops[op].vec = vec;
  x->nwords += x->ops[op].w_cnt;
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_op_emit( libxsmm_sme_bg_ctx* x, unsigned int op ) {
  libxsmm_generated_code* c = x->code;
  const libxsmm_sme_bg_op* l_op = &x->ops[op];
  unsigned int l_i;
  if ( c->buffer_size - c->code_size < 4 * l_op->w_cnt ) {
    LIBXSMM_HANDLE_ERROR( c, LIBXSMM_ERR_BUFFER_TOO_SMALL );
    return;
  }
  for ( l_i = 0; l_i < l_op->w_cnt; ++l_i ) {
    ((unsigned int*)c->generated_code)[c->code_size / 4] = x->words[l_op->w_off + l_i];
    c->code_size += 4;
  }
}

/* ------------------------------------------------------------------------------------ prologue / epilogue */
LIBXSMM_API_INLINE void libxsmm_sme_bg_prologue( libxsmm_sme_bg_ctx* x ) {
  libxsmm_generated_code* c = x->code;
  const int l_off_a = (int)sizeof(libxsmm_matrix_op_arg);
  const int l_off_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));
  const int l_off_c = (int)(sizeof(libxsmm_matrix_op_arg) + 2 * sizeof(libxsmm_matrix_arg));

  libxsmm_aarch64_instruction_open_stream( c, LIBXSMM_SME_BG_SAVE );
  /* x0 = param struct; TT: A' = B, B' = A */
  libxsmm_aarch64_instruction_alu_move( c, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_SME_BG_X(0), LIBXSMM_AARCH64_GP_REG_UNDEF,
                                        (0 != x->swap) ? l_off_a : l_off_b, LIBXSMM_SME_BG_X(1) );
  libxsmm_aarch64_instruction_alu_move( c, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_SME_BG_X(0), LIBXSMM_AARCH64_GP_REG_UNDEF,
                                        l_off_c, LIBXSMM_SME_BG_X(2) );
  libxsmm_aarch64_instruction_alu_move( c, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_SME_BG_X(0), LIBXSMM_AARCH64_GP_REG_UNDEF,
                                        (0 != x->swap) ? l_off_b : l_off_a, LIBXSMM_SME_BG_X(0) );
  /* workspace on the stack, 128-byte aligned, LIBXSMM_SME_BG_WS_GAP below the register save area */
  libxsmm_aarch64_instruction_alu_compute_imm12( c, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_SME_BG_X(29), 0, 0 );
  if ( 0 != x->ws_bytes ) {
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_AARCH64_GP_REG_XSP, -(long long)(x->ws_bytes + 128 + LIBXSMM_SME_BG_WS_GAP) );
    libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(17), 0xffffffffffffff80ULL );
    libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_AND_SR, LIBXSMM_SME_BG_X(27), LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(17), 0 );
    libxsmm_aarch64_instruction_alu_compute_imm12( c, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_SME_BG_X(27), LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  }
  /* leading dimensions in bytes (other strides are immediates), k-stride of op(A) */
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(4), 4ULL * x->ldb );
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(5), 4ULL * x->ldc );
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(6), (0 != x->a_packed) ? 4ULL * x->mpad : 4ULL * x->lda );

  libxsmm_aarch64_instruction_sm( c, LIBXSMM_AARCH64_INSTR_SME_SMSTART );
  libxsmm_aarch64_instruction_sve_pcompute( c, LIBXSMM_AARCH64_INSTR_SVE_PTRUE, LIBXSMM_AARCH64_SVE_REG_P0, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            LIBXSMM_AARCH64_GP_WIDTH_W, LIBXSMM_AARCH64_GP_REG_UNDEF, LIBXSMM_AARCH64_SVE_PATTERN_ALL,
                                            LIBXSMM_AARCH64_SVE_TYPE_S );
  libxsmm_aarch64_instruction_set_ptrue_as_counter_sve2( c, LIBXSMM_AARCH64_INSTR_SVE2_PTRUE_AS_COUNTER, LIBXSMM_AARCH64_SVE_REG_P8 );
  libxsmm_sme_bg_wset( c );
  if ( 0 != x->beta0 ) {
    /* C is only stored: one dummy load of C keeps the store stream to the same lines (call after
     * call) out of a slow state of the SME unit; its value is never used */
    const unsigned int l_p = libxsmm_sme_bg_pred( c, LIBXSMM_MIN(LIBXSMM_SME_BG_VL, (0 != x->swap) ? x->n : x->m), 1, LIBXSMM_AARCH64_SVE_REG_P5 );
    libxsmm_sme_bg_ldst( c, 0, 31, 1, 0, LIBXSMM_SME_BG_X(2), 0, l_p );
  }
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_epilogue( libxsmm_sme_bg_ctx* x ) {
  libxsmm_aarch64_instruction_sm( x->code, LIBXSMM_AARCH64_INSTR_SME_SMSTOP );
  libxsmm_aarch64_instruction_alu_compute_imm12( x->code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_SME_BG_X(29), LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  libxsmm_aarch64_instruction_close_stream( x->code, LIBXSMM_SME_BG_SAVE );
}

/* ------------------------------------------------------------------------------------ packing through ZA */
/* one round: kr <= 32 k x jr <= 32 columns at src (columns contiguous over k, stride ld elements)
 * -> rows dst[k][0..32) (row stride row_bytes). Columns go into horizontal slices of the 4 tiles
 * (tile jh: k 0..15, tile 2+jh: k 16..31 of columns 16jh..), rows come out of vertical slices. */
LIBXSMM_API_INLINE void libxsmm_sme_bg_pack_round( libxsmm_sme_bg_ctx* x, unsigned int src, unsigned int ld, unsigned int kr, unsigned int jr,
                                                   unsigned int dst, unsigned int row_bytes ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int l_pk = libxsmm_sme_bg_pred( c, kr, 2, LIBXSMM_AARCH64_SVE_REG_P12 );
  const unsigned int l_halves = libxsmm_sme_bg_cdiv( jr, 16 );
  unsigned int l_c, l_q, l_i, l_cc, l_nn;
  for ( l_c = 0; l_c < libxsmm_sme_bg_cdiv( jr, 4 ); ++l_c ) {
    const unsigned int l_r = (0 == l_c % 2) ? 0 : 4;
    const unsigned int l_jh = l_c / 4;
    for ( l_q = 0; l_q < 4; ++l_q ) {
      const unsigned int l_j = 4 * l_c + l_q;
      if ( l_j >= jr ) {
        libxsmm_sme_bg_zeroz( c, l_r + l_q );
        libxsmm_sme_bg_zeroz( c, l_r + 8 + l_q );
        continue;
      }
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), src, 4LL * l_j * ld );
      libxsmm_sme_bg_ldst( c, 0, l_r + l_q, 2, 1, LIBXSMM_SME_BG_X(16), 0, l_pk );
    }
    libxsmm_sme_bg_mova( c, 1, 0, l_jh, l_c % 4, l_r );
    if ( kr > 16 ) {
      libxsmm_sme_bg_mova( c, 1, 0, 2 + l_jh, l_c % 4, l_r + 8 );
    }
  }
  for ( l_i = 0; l_i < libxsmm_sme_bg_cdiv( kr, 16 ); ++l_i ) {
    for ( l_cc = 0; l_cc < 4; ++l_cc ) {
      const unsigned int l_rr = (0 == l_cc % 2) ? 16 : 20;
      if ( 16 * l_i + 4 * l_cc >= kr ) break;
      libxsmm_sme_bg_mova( c, 0, 1, 2 * l_i, l_cc, l_rr );
      if ( l_halves > 1 ) {
        libxsmm_sme_bg_mova( c, 0, 1, 2 * l_i + 1, l_cc, l_rr + 8 );
      }
      for ( l_nn = 0; l_nn < 4; ++l_nn ) {
        libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), dst, (long long)(16 * l_i + 4 * l_cc + l_nn) * row_bytes );
        if ( l_halves > 1 ) {
          libxsmm_sme_bg_ldst( c, 1, l_rr + l_nn, 2, 1, LIBXSMM_SME_BG_X(16), 0, LIBXSMM_AARCH64_SVE_REG_P8 );
        } else {
          libxsmm_sme_bg_ldst( c, 1, l_rr + l_nn, 1, 0, LIBXSMM_SME_BG_X(16), 0, LIBXSMM_AARCH64_SVE_REG_P0 );
        }
      }
    }
  }
  libxsmm_aarch64_instruction_sme_zero( c, 0xff );
}

/* transpose a K x ncols column-major piece (columns contiguous over k, stride ld elements, at src)
 * into rows dst[k][0 .. ncols) (row stride row_bytes, at dst), loops over identical rounds.
 * Rows up to the next multiple of 32 and lanes up to the next multiple of 32 are written too.
 * Uses x10, x21-x25. */
LIBXSMM_API_INLINE void libxsmm_sme_bg_pack( libxsmm_sme_bg_ctx* x, unsigned int src, unsigned int ld, unsigned int K, unsigned int ncols,
                                             unsigned int dst, unsigned int row_bytes ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_kr[2], l_kc[2], l_jr[2], l_jc[2], l_nk = 0, l_nj = 0, l_a, l_b;
  if ( K >= 32 ) { l_kr[l_nk] = 32; l_kc[l_nk++] = K / 32; }
  if ( 0 != K % 32 ) { l_kr[l_nk] = K % 32; l_kc[l_nk++] = 1; }
  if ( ncols >= 32 ) { l_jr[l_nj] = 32; l_jc[l_nj++] = ncols / 32; }
  if ( 0 != ncols % 32 ) { l_jr[l_nj] = ncols % 32; l_jc[l_nj++] = 1; }
  libxsmm_sme_bg_wset( c );   /* the background C update uses x12-x15 as pointers */
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(21), src );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(22), dst );
  for ( l_a = 0; l_a < l_nk; ++l_a ) {
    if ( l_kc[l_a] > 1 ) {
      libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(24), l_kc[l_a] );
    }
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(21) );
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(10), LIBXSMM_SME_BG_X(22) );
    for ( l_b = 0; l_b < l_nj; ++l_b ) {
      if ( l_jc[l_b] > 1 ) {
        libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), l_jc[l_b] );
      }
      libxsmm_sme_bg_pack_round( x, LIBXSMM_SME_BG_X(23), ld, l_kr[l_a], l_jr[l_b], LIBXSMM_SME_BG_X(10), row_bytes );
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(23), 128LL * ld );
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(10), LIBXSMM_SME_BG_X(10), 128 );
      if ( l_jc[l_b] > 1 ) {
        libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), 1 );
      }
    }
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(21), LIBXSMM_SME_BG_X(21), 128 );
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(22), LIBXSMM_SME_BG_X(22), 32LL * row_bytes );
    if ( l_kc[l_a] > 1 ) {
      libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(24), 1 );
    }
  }
}

/* ------------------------------------------------------------------------------------ blocks: tiles, operands */
LIBXSMM_API_INLINE unsigned int libxsmm_sme_bg_tile( const libxsmm_sme_bg_block* b, unsigned int ib, unsigned int jb ) {
  return b->tm * jb + ib;
}

/* operand registers of k-step kk (3 sets): blocks of <= 2x2 tiles: A in z4s, z4s+1, B in z4s+2, z4s+3;
 * 1 x 3..4 tiles: A in z12+s, B in z4s..z4s+3; 3..4 x 1 tiles: A in z4s..z4s+3, B in z12+s */
LIBXSMM_API_INLINE void libxsmm_sme_bg_opregs( const libxsmm_sme_bg_block* b, unsigned int kk, unsigned int* za, unsigned int* nza,
                                               unsigned int* zb, unsigned int* nzb ) {
  const unsigned int l_s = kk % 3;
  if ( b->tm <= 2 && b->tn <= 2 ) {
    *za = 4 * l_s; *nza = b->tm; *zb = 4 * l_s + 2; *nzb = b->tn;
  } else if ( 1 == b->tm ) {
    *za = 12 + l_s; *nza = 1; *zb = 4 * l_s; *nzb = 4;
  } else {
    *za = 4 * l_s; *nza = 4; *zb = 12 + l_s; *nzb = 1;
  }
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_load( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, unsigned int kk, unsigned int pa, unsigned int pb ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_za, l_nza, l_zb, l_nzb;
  libxsmm_sme_bg_opregs( b, kk, &l_za, &l_nza, &l_zb, &l_nzb );
  libxsmm_sme_bg_ldst( c, 0, l_za, l_nza, 0, LIBXSMM_SME_BG_X(10), 0, pa );
  if ( LIBXSMM_AARCH64_GP_REG_UNDEF != x->a_stride_reg ) {
    libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, LIBXSMM_SME_BG_X(10), LIBXSMM_SME_BG_X(10), x->a_stride_reg, 0 );
  } else {
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(10), LIBXSMM_SME_BG_X(10), x->a_stride_imm );
  }
  libxsmm_sme_bg_ldst( c, 0, l_zb, l_nzb, 0, LIBXSMM_SME_BG_X(11), 0, pb );
  if ( LIBXSMM_AARCH64_GP_REG_UNDEF != x->b_stride_reg ) {
    libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, LIBXSMM_SME_BG_X(11), LIBXSMM_SME_BG_X(11), x->b_stride_reg, 0 );
  } else {
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(11), LIBXSMM_SME_BG_X(11), x->b_stride_imm );
  }
}

/* FMOPA i (tile rows fastest) of k-step kk */
LIBXSMM_API_INLINE void libxsmm_sme_bg_fmopa_one( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, unsigned int kk, unsigned int i ) {
  const unsigned int l_ib = i % b->tm, l_jb = i / b->tm;
  unsigned int l_za, l_nza, l_zb, l_nzb;
  libxsmm_sme_bg_opregs( b, kk, &l_za, &l_nza, &l_zb, &l_nzb );
  libxsmm_sme_bg_fmopa( x->code, libxsmm_sme_bg_tile( b, l_ib, l_jb ), l_zb + l_jb, l_za + l_ib );
}

/* predicates of the operand loads of a block */
LIBXSMM_API_INLINE void libxsmm_sme_bg_oppreds( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, unsigned int* pa, unsigned int* pb ) {
  unsigned int l_za, l_nza, l_zb, l_nzb;
  libxsmm_sme_bg_opregs( b, 0, &l_za, &l_nza, &l_zb, &l_nzb );
  *pa = libxsmm_sme_bg_pred( x->code, b->mb, l_nza, (1 == l_nza) ? LIBXSMM_AARCH64_SVE_REG_P1 : LIBXSMM_AARCH64_SVE_REG_P9 );
  *pb = libxsmm_sme_bg_pred( x->code, b->nb, l_nzb, (1 == l_nzb) ? LIBXSMM_AARCH64_SVE_REG_P2 : LIBXSMM_AARCH64_SVE_REG_P10 );
}

/* ------------------------------------------------------------------------------------ dump mode */
LIBXSMM_API_INLINE void libxsmm_sme_bg_prev_ptr( libxsmm_sme_bg_ctx* x, unsigned int kind ) {
  libxsmm_generated_code* c = x->code;
  if ( LIBXSMM_SME_BG_PREV_WITHIN == kind ) {
    if ( 0 != x->swap ) {   /* blocks step along C columns */
      libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, LIBXSMM_SME_BG_X(14), LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(5), 5 );
    } else {
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(14), LIBXSMM_SME_BG_X(23), -128 );
    }
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(15), LIBXSMM_SME_BG_X(5) );
  } else {
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(14), LIBXSMM_SME_BG_X(7) );
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(15), LIBXSMM_AARCH64_GP_REG_X30 );
  }
}

/* pointers and predicate of the background update of the predecessor: x12 = x14 = its C,
 * x15 = column stride, x13 = dump, pn14 = its rows */
LIBXSMM_API_INLINE void libxsmm_sme_bg_bg_setup( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_prev* p ) {
  libxsmm_generated_code* c = x->code;
  libxsmm_sme_bg_prev_ptr( x, p->kind );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(12), LIBXSMM_SME_BG_X(14) );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(13), LIBXSMM_SME_BG_X(27), x->dump_off );
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(16), p->mb );
  libxsmm_sme_bg_whilelt( c, LIBXSMM_AARCH64_SVE_REG_P14, 2, LIBXSMM_SME_BG_XZR, LIBXSMM_SME_BG_X(16) );
}

/* ZA -> dump: column c (32 rows) at ws + dump_off + 128*c; swap: ZA holds C^T, vertical slices */
LIBXSMM_API_INLINE void libxsmm_sme_bg_dump_readout( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int l_nouter = (0 != x->swap) ? b->tm : b->tn;   /* tiles along the C columns */
  const unsigned int l_ninner = (0 != x->swap) ? b->tn : b->tm;   /* tiles along the C rows */
  const unsigned int l_ncols = (0 != x->swap) ? b->mb : b->nb;
  unsigned int l_nbk = 0, l_o, l_s0, l_i, l_q;
  libxsmm_sme_bg_wset( c );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(27), x->dump_off );
  for ( l_o = 0; l_o < l_nouter; ++l_o ) {
    for ( l_s0 = 0; l_s0 < 16; l_s0 += 4 ) {
      const unsigned int l_sh = (0 == l_nbk % 2) ? 0 : 4;   /* alternate register banks */
      if ( 16 * l_o + l_s0 >= l_ncols ) continue;
      ++l_nbk;
      for ( l_i = 0; l_i < l_ninner; ++l_i ) {
        const unsigned int l_tile = (0 != x->swap) ? libxsmm_sme_bg_tile( b, l_o, l_i ) : libxsmm_sme_bg_tile( b, l_i, l_o );
        libxsmm_sme_bg_mova( c, 0, x->swap, l_tile, l_s0 / 4, 16 + 8 * l_i + l_sh );
      }
      for ( l_q = 0; l_q < 4; ++l_q ) {
        const unsigned int l_col = 16 * l_o + l_s0 + l_q;
        if ( l_col >= l_ncols ) continue;
        if ( 0 == l_col % 8 && 0 != l_col ) {
          libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(16), 1024 );
        }
        if ( 1 == l_ninner ) {
          libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(16), (l_col % 8) * 128 );
          libxsmm_sme_bg_ldst( c, 1, 16 + l_q + l_sh, 1, 0, LIBXSMM_SME_BG_X(17), 0, LIBXSMM_AARCH64_SVE_REG_P0 );
        } else {
          libxsmm_sme_bg_ldst( c, 1, 16 + l_q + l_sh, 2, 1, LIBXSMM_SME_BG_X(16), (int)(l_col % 8) * 2, LIBXSMM_AARCH64_SVE_REG_P8 );
        }
      }
    }
  }
}

/* C (x14, mb x nb, column stride x15) = dump + beta*C */
LIBXSMM_API_INLINE void libxsmm_sme_bg_sync_update( libxsmm_sme_bg_ctx* x, unsigned int mb, unsigned int nb ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_c;
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(12), LIBXSMM_SME_BG_X(14) );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(13), LIBXSMM_SME_BG_X(27), x->dump_off );
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(16), mb );
  libxsmm_sme_bg_whilelt( c, LIBXSMM_AARCH64_SVE_REG_P14, 2, LIBXSMM_SME_BG_XZR, LIBXSMM_SME_BG_X(16) );
  for ( l_c = 0; l_c < nb + 2; ++l_c ) {
    if ( l_c < nb ) {
      const unsigned int l_r = 16 + 4 * (l_c % 4);
      libxsmm_sme_bg_ldst( c, 0, l_r, 2, 0, LIBXSMM_SME_BG_X(13), 0, LIBXSMM_AARCH64_SVE_REG_P8 );
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(13), LIBXSMM_SME_BG_X(13), 128 );
      if ( 0 == x->beta0 ) {
        libxsmm_sme_bg_ldst( c, 0, l_r + 2, 2, 0, LIBXSMM_SME_BG_X(12), 0, LIBXSMM_AARCH64_SVE_REG_P14 );
        libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, LIBXSMM_SME_BG_X(12), LIBXSMM_SME_BG_X(12), LIBXSMM_SME_BG_X(15), 0 );
      }
    }
    if ( l_c >= 2 ) {
      const unsigned int l_r = 16 + 4 * ((l_c - 2) % 4);
      if ( 0 == x->beta0 ) {
        libxsmm_sme_bg_fadd( c, l_r, l_r, l_r + 2 );
        libxsmm_sme_bg_fadd( c, l_r + 1, l_r + 1, l_r + 3 );
      }
      libxsmm_sme_bg_ldst( c, 1, l_r, 2, 0, LIBXSMM_SME_BG_X(14), 0, LIBXSMM_AARCH64_SVE_REG_P14 );
      libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, LIBXSMM_SME_BG_X(14), LIBXSMM_SME_BG_X(14), LIBXSMM_SME_BG_X(15), 0 );
    }
  }
}

/* background C update in the rolled k-loop: chunks of cpi columns per iteration of
 * LIBXSMM_SME_BG_KUNROLL k-steps; which: 0 = none, 1 = load chunk, 2 = consume + load, 3 = consume.
 * Column c of a chunk lives in z(16+4c) .. z(19+4c): dump (2 vectors), C (2 vectors). */
LIBXSMM_API_INLINE void libxsmm_sme_bg_iter_ops( libxsmm_sme_bg_ctx* x, unsigned int which, unsigned int cpi, unsigned int kk ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_c;
  for ( l_c = 0; l_c < cpi; ++l_c ) {
    const unsigned int l_r = 16 + 4 * l_c;
    if ( 2 == which || 3 == which ) {
      if ( 0 == x->beta0 && kk == 2 * l_c ) {
        libxsmm_sme_bg_fadd( c, l_r, l_r, l_r + 2 );
      }
      if ( 0 == x->beta0 && kk == 2 * l_c + 1 ) {
        libxsmm_sme_bg_fadd( c, l_r + 1, l_r + 1, l_r + 3 );
      }
      if ( kk == 2 * l_c + 1 + LIBXSMM_SME_BG_SDIST ) {
        libxsmm_sme_bg_ldst( c, 1, l_r, 2, 0, LIBXSMM_SME_BG_X(14), 0, LIBXSMM_AARCH64_SVE_REG_P14 );
        libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, LIBXSMM_SME_BG_X(14), LIBXSMM_SME_BG_X(14), LIBXSMM_SME_BG_X(15), 0 );
      }
    }
    if ( (1 == which || 2 == which) && kk == LIBXSMM_MIN(2 * l_c + 2 + LIBXSMM_SME_BG_SDIST, LIBXSMM_SME_BG_KUNROLL - 1) ) {
      libxsmm_sme_bg_ldst( c, 0, l_r, 2, 0, LIBXSMM_SME_BG_X(13), 0, LIBXSMM_AARCH64_SVE_REG_P8 );
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(13), LIBXSMM_SME_BG_X(13), 128 );
      if ( 0 == x->beta0 ) {
        libxsmm_sme_bg_ldst( c, 0, l_r + 2, 2, 0, LIBXSMM_SME_BG_X(12), 0, LIBXSMM_AARCH64_SVE_REG_P14 );
        libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, LIBXSMM_SME_BG_X(12), LIBXSMM_SME_BG_X(12), LIBXSMM_SME_BG_X(15), 0 );
      }
    }
  }
}

/* cnt iterations of LIBXSMM_SME_BG_KUNROLL k-steps starting at k-step k0 (a multiple of 3) */
LIBXSMM_API_INLINE void libxsmm_sme_bg_iters( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, unsigned int k0, unsigned int cnt,
                                              unsigned int which, unsigned int cpi, unsigned int pa, unsigned int pb ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int l_nf = b->tm * b->tn;
  unsigned int l_kk, l_i;
  if ( 0 == cnt ) return;
  if ( cnt > 1 ) {
    libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(26), cnt );
  }
  for ( l_kk = 0; l_kk < LIBXSMM_SME_BG_KUNROLL; ++l_kk ) {
    libxsmm_sme_bg_load( x, b, k0 + l_kk + LIBXSMM_SME_BG_DIST, pa, pb );
    for ( l_i = 0; l_i < l_nf; ++l_i ) {
      libxsmm_sme_bg_fmopa_one( x, b, k0 + l_kk, l_i );
      if ( 0 == l_i && 0 != which ) {
        libxsmm_sme_bg_iter_ops( x, which, cpi, l_kk );
      }
    }
  }
  if ( cnt > 1 ) {
    libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(26), 1 );
  }
}

/* one block in dump mode with a rolled k-loop: op(A) from x21, op(B) from x22; the C update of
 * the predecessor p runs in the background of the k-loop if it fits, else after it */
LIBXSMM_API_INLINE void libxsmm_sme_bg_block_dump( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, const libxsmm_sme_bg_prev* p ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int K = x->k, D = LIBXSMM_SME_BG_DIST, U = LIBXSMM_SME_BG_KUNROLL;
  const unsigned int l_niter = (K - D) / U;
  const unsigned int l_nf = b->tm * b->tn;
  const unsigned int l_s0 = LIBXSMM_SME_BG_START;
  unsigned int l_pa, l_pb, l_cpi = 0, l_nch = 0, l_kk, l_i;
  libxsmm_sme_bg_oppreds( x, b, &l_pa, &l_pb );
  libxsmm_aarch64_instruction_sme_zero( c, 0xff );
  if ( NULL != p ) {   /* columns per iteration: as few as fit */
    const unsigned int l_try[3] = { 1, 2, 4 };
    for ( l_i = 0; l_i < 3; ++l_i ) {
      if ( 0 == p->nb % l_try[l_i] && l_s0 + p->nb / l_try[l_i] + 1 <= l_niter ) {
        l_cpi = l_try[l_i];
        l_nch = p->nb / l_cpi;
        break;
      }
    }
  }
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(10), LIBXSMM_SME_BG_X(21) );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(11), LIBXSMM_SME_BG_X(22) );
  for ( l_kk = 0; l_kk < LIBXSMM_MIN(D, K); ++l_kk ) {
    libxsmm_sme_bg_load( x, b, l_kk, l_pa, l_pb );
  }
  if ( 0 != l_cpi ) {
    libxsmm_sme_bg_bg_setup( x, p );
    libxsmm_sme_bg_iters( x, b, 0, l_s0, 0, l_cpi, l_pa, l_pb );
    libxsmm_sme_bg_iters( x, b, 0, 1, 1, l_cpi, l_pa, l_pb );
    libxsmm_sme_bg_iters( x, b, 0, l_nch - 1, 2, l_cpi, l_pa, l_pb );
    libxsmm_sme_bg_iters( x, b, 0, 1, 3, l_cpi, l_pa, l_pb );
    libxsmm_sme_bg_iters( x, b, 0, l_niter - l_s0 - l_nch - 1, 0, l_cpi, l_pa, l_pb );
  } else {
    libxsmm_sme_bg_iters( x, b, 0, l_niter, 0, 0, l_pa, l_pb );
  }
  for ( l_kk = l_niter * U; l_kk < K; ++l_kk ) {
    if ( l_kk + D < K ) {
      libxsmm_sme_bg_load( x, b, l_kk + D, l_pa, l_pb );
    }
    for ( l_i = 0; l_i < l_nf; ++l_i ) {
      libxsmm_sme_bg_fmopa_one( x, b, l_kk, l_i );
    }
  }
  if ( NULL != p && 0 == l_cpi ) {   /* no room in the k-loop: update the predecessor now */
    libxsmm_sme_bg_prev_ptr( x, p->kind );
    libxsmm_sme_bg_sync_update( x, p->mb, p->nb );
  }
  libxsmm_sme_bg_dump_readout( x, b );
}

/* dump mode for all blocks, op(B) native or packed up front (x22 base): panels of 32 columns,
 * 32-row blocks + an optional edge row block; every block updates its predecessor */
LIBXSMM_API_INLINE void libxsmm_sme_bg_bg_all( libxsmm_sme_bg_ctx* x, unsigned int abase ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_run, l_prev_nb = 0;
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(7), LIBXSMM_SME_BG_X(27), x->dummy_off );
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_AARCH64_GP_REG_X30, 128 );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(20), LIBXSMM_SME_BG_X(2) );
  for ( l_run = 0; l_run < 2; ++l_run ) {
    const unsigned int l_nb = (0 == l_run) ? 32 : x->nedge;
    const unsigned int l_cnt = (0 == l_run) ? x->nfull : ((0 != x->nedge) ? 1 : 0);
    libxsmm_sme_bg_block l_full, l_edge;
    libxsmm_sme_bg_prev l_p;
    if ( 0 == l_cnt ) continue;
    l_full.mb = 32; l_full.nb = l_nb; l_full.tm = 2; l_full.tn = libxsmm_sme_bg_cdiv( l_nb, 16 );
    l_edge.mb = x->medge; l_edge.nb = l_nb; l_edge.tm = libxsmm_sme_bg_cdiv( x->medge, 16 ); l_edge.tn = l_full.tn;
    if ( l_cnt > 1 ) {
      libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(24), l_cnt );
    }
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(21), abase );
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(20) );
    l_p.mb = x->last_mb; l_p.nb = (0 == l_prev_nb) ? l_nb : l_prev_nb; l_p.kind = LIBXSMM_SME_BG_PREV_PANEL;
    libxsmm_sme_bg_block_dump( x, &l_full, &l_p );
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(21), LIBXSMM_SME_BG_X(21), 128 );
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(23), 128 );
    l_p.mb = 32; l_p.nb = l_nb; l_p.kind = LIBXSMM_SME_BG_PREV_WITHIN;
    if ( x->fcnt > 1 ) {
      if ( x->fcnt > 2 ) {
        libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), x->fcnt - 1 );
      }
      libxsmm_sme_bg_block_dump( x, &l_full, &l_p );
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(21), LIBXSMM_SME_BG_X(21), 128 );
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(23), 128 );
      if ( x->fcnt > 2 ) {
        libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), 1 );
      }
    }
    if ( 0 != x->medge ) {
      libxsmm_sme_bg_block_dump( x, &l_edge, &l_p );
    }
    /* predecessor of the next panel's first block: the last block of this panel */
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(7), LIBXSMM_SME_BG_X(20), 4LL * x->last_i0 );
    libxsmm_sme_bg_mov( c, LIBXSMM_AARCH64_GP_REG_X30, LIBXSMM_SME_BG_X(5) );
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(22), LIBXSMM_SME_BG_X(22), 4LL * l_nb );
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(20), LIBXSMM_SME_BG_X(20), 4LL * l_nb * x->ldc );
    if ( l_cnt > 1 ) {
      libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(24), 1 );
    }
    l_prev_nb = l_nb;
  }
  /* the last block of the call: synchronous update */
  libxsmm_sme_bg_prev_ptr( x, LIBXSMM_SME_BG_PREV_PANEL );
  libxsmm_sme_bg_sync_update( x, x->last_mb, l_prev_nb );
}

/* ------------------------------------------------------------------------------------ background transposition */
/* ops transposing G k-groups (16 k each) of the next panel: x8 = source (column 0 of the next panel,
 * first k-group of this block), x9 = destination row of that k-group, x26 = 16 * that k-group.
 * Per 16x16 piece: stage 1 loads 4 groups of 4 columns, zips each group and stores it to a scratch
 * buffer; stage 2 reloads them crosswise, zips again and stores 16 rows of the panel. */
LIBXSMM_API_INLINE void libxsmm_sme_bg_tr_ops( libxsmm_sme_bg_ctx* x, unsigned int G, unsigned int pred ) {
  unsigned int l_qlast[2][4], l_qlast_n[2] = { 0, 0 };
  unsigned int l_g, l_h, l_gq, l_q, l_mm, l_nn, l_i;
  for ( l_g = 0; l_g < G; ++l_g ) {
    unsigned int l_pr = LIBXSMM_AARCH64_SVE_REG_P0, l_wl = 0, l_ploads[32], l_nploads = 0;
    if ( 0 != pred ) {
      const unsigned int l_pi = l_g % 2;
      l_pr = (0 == l_pi) ? LIBXSMM_AARCH64_SVE_REG_P6 : LIBXSMM_AARCH64_SVE_REG_P7;
      l_wl = libxsmm_sme_bg_op_begin( x );
      libxsmm_sme_bg_add_imm( &x->opcode, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(26), 16 * l_g );
      libxsmm_aarch64_instruction_alu_set_imm64( &x->opcode, LIBXSMM_SME_BG_X(16), x->k );
      libxsmm_sme_bg_whilelt( &x->opcode, l_pr, 1, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(16) );
      for ( l_i = 0; l_i < x->plast_n[l_pi]; ++l_i ) libxsmm_sme_bg_op_dep( x, l_wl, x->plast[l_pi][l_i], 0 );
      libxsmm_sme_bg_op_end( x, l_wl, 0 );
    }
    for ( l_h = 0; l_h < 2; ++l_h ) {
      const unsigned int l_piece = 2 * l_g + l_h;
      const unsigned int l_scr = x->scr_off + 1024 * (l_piece % 2);
      unsigned int l_st1[4], l_ld2[16], l_nld2 = 0;
      for ( l_gq = 0; l_gq < 4; ++l_gq ) {
        const unsigned int l_qi = l_gq % 2, l_qb = 24 + 4 * l_qi;
        unsigned int l_lds[4], l_z, l_st;
        for ( l_q = 0; l_q < 4; ++l_q ) {
          const unsigned int l_col = 16 * l_h + 4 * l_q + l_gq;
          l_lds[l_q] = libxsmm_sme_bg_op_begin( x );
          libxsmm_sme_bg_add_imm( &x->opcode, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(8), 4LL * l_col * x->ldb );
          libxsmm_sme_bg_ldst( &x->opcode, 0, l_qb + l_q, 1, 0, LIBXSMM_SME_BG_X(16), (int)l_g, l_pr );
          for ( l_i = 0; l_i < l_qlast_n[l_qi]; ++l_i ) libxsmm_sme_bg_op_dep( x, l_lds[l_q], l_qlast[l_qi][l_i], 0 );
          if ( 0 != pred ) libxsmm_sme_bg_op_dep( x, l_lds[l_q], l_wl, 0 );
          libxsmm_sme_bg_op_end( x, l_lds[l_q], 0 );
          if ( l_nploads < 32 ) l_ploads[l_nploads++] = l_lds[l_q];
        }
        l_z = libxsmm_sme_bg_op_begin( x );
        libxsmm_aarch64_instruction_sme2_zip( &x->opcode, LIBXSMM_AARCH64_INSTR_SME2_ZIP_4_S, l_qb, l_qb );
        for ( l_q = 0; l_q < 4; ++l_q ) libxsmm_sme_bg_op_dep( x, l_z, l_lds[l_q], x->d_load );
        libxsmm_sme_bg_op_end( x, l_z, 1 );
        l_st = libxsmm_sme_bg_op_begin( x );
        libxsmm_sme_bg_add_imm( &x->opcode, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(27), l_scr );
        libxsmm_sme_bg_ldst( &x->opcode, 1, l_qb, 4, 0, LIBXSMM_SME_BG_X(17), 4 * (int)l_gq, LIBXSMM_AARCH64_SVE_REG_P8 );
        libxsmm_sme_bg_op_dep( x, l_st, l_z, LIBXSMM_SME_BG_D_ZIP );
        for ( l_i = 0; l_i < x->scr_last_n[l_piece % 2]; ++l_i ) libxsmm_sme_bg_op_dep( x, l_st, x->scr_last[l_piece % 2][l_i], 0 );
        libxsmm_sme_bg_op_end( x, l_st, 0 );
        l_qlast[l_qi][0] = l_st;
        l_qlast_n[l_qi] = 1;
        l_st1[l_gq] = l_st;
      }
      for ( l_mm = 0; l_mm < 4; ++l_mm ) {
        const unsigned int l_qi = l_mm % 2, l_qb = 24 + 4 * l_qi;
        unsigned int l_lds[4], l_z;
        for ( l_gq = 0; l_gq < 4; ++l_gq ) {
          const unsigned int l_idx = 4 * l_gq + l_mm;
          l_lds[l_gq] = libxsmm_sme_bg_op_begin( x );
          libxsmm_sme_bg_add_imm( &x->opcode, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(27), l_scr + ((l_idx < 8) ? 0 : 512) );
          libxsmm_sme_bg_ldst( &x->opcode, 0, l_qb + l_gq, 1, 0, LIBXSMM_SME_BG_X(17), (int)(l_idx % 8), LIBXSMM_AARCH64_SVE_REG_P0 );
          libxsmm_sme_bg_op_dep( x, l_lds[l_gq], l_st1[l_gq], x->d_mem );
          for ( l_i = 0; l_i < l_qlast_n[l_qi]; ++l_i ) libxsmm_sme_bg_op_dep( x, l_lds[l_gq], l_qlast[l_qi][l_i], 0 );
          libxsmm_sme_bg_op_end( x, l_lds[l_gq], 0 );
          l_ld2[l_nld2++] = l_lds[l_gq];
        }
        l_z = libxsmm_sme_bg_op_begin( x );
        libxsmm_aarch64_instruction_sme2_zip( &x->opcode, LIBXSMM_AARCH64_INSTR_SME2_ZIP_4_S, l_qb, l_qb );
        for ( l_gq = 0; l_gq < 4; ++l_gq ) libxsmm_sme_bg_op_dep( x, l_z, l_lds[l_gq], x->d_load );
        libxsmm_sme_bg_op_end( x, l_z, 1 );
        for ( l_nn = 0; l_nn < 4; ++l_nn ) {
          const unsigned int l_row = 16 * l_g + 4 * l_mm + l_nn;
          const unsigned int l_st = libxsmm_sme_bg_op_begin( x );
          libxsmm_sme_bg_add_imm( &x->opcode, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(9), (long long)l_row * 128 + 64 * l_h );
          libxsmm_sme_bg_ldst( &x->opcode, 1, l_qb + l_nn, 1, 0, LIBXSMM_SME_BG_X(16), 0, LIBXSMM_AARCH64_SVE_REG_P0 );
          libxsmm_sme_bg_op_dep( x, l_st, l_z, LIBXSMM_SME_BG_D_ZIP );
          libxsmm_sme_bg_op_end( x, l_st, 0 );
          l_qlast[l_qi][l_nn] = l_st;
        }
        l_qlast_n[l_qi] = 4;
      }
      for ( l_i = 0; l_i < l_nld2; ++l_i ) x->scr_last[l_piece % 2][l_i] = l_ld2[l_i];
      x->scr_last_n[l_piece % 2] = l_nld2;
    }
    if ( 0 != pred ) {
      for ( l_i = 0; l_i < l_nploads; ++l_i ) x->plast[l_g % 2][l_i] = l_ploads[l_i];
      x->plast_n[l_g % 2] = l_nploads;
    }
  }
}

/* ops of the background update of the predecessor's C, columns 0 .. pnb-1 (x12 = C, x15 = column
 * stride, x13 = dump, pn14 = rows); two register sets z16-z19, z20-z23 */
LIBXSMM_API_INLINE void libxsmm_sme_bg_cu_ops( libxsmm_sme_bg_ctx* x, unsigned int pnb ) {
  unsigned int l_last[2] = { 0, 0 }, l_has_last[2] = { 0, 0 }, l_c;
  for ( l_c = 0; l_c < pnb; ++l_c ) {
    const unsigned int l_s = 16 + 4 * (l_c % 2);
    unsigned int l_ldd, l_ldc = 0, l_f1 = 0, l_f2 = 0, l_st;
    l_ldd = libxsmm_sme_bg_op_begin( x );
    libxsmm_sme_bg_add_imm( &x->opcode, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(13), 128 * l_c );
    libxsmm_sme_bg_ldst( &x->opcode, 0, l_s, 2, 0, LIBXSMM_SME_BG_X(16), 0, LIBXSMM_AARCH64_SVE_REG_P8 );
    if ( 0 != l_has_last[l_c % 2] ) libxsmm_sme_bg_op_dep( x, l_ldd, l_last[l_c % 2], 0 );
    libxsmm_sme_bg_op_end( x, l_ldd, 0 );
    if ( 0 == x->beta0 ) {
      l_ldc = libxsmm_sme_bg_op_begin( x );
      if ( 0 != l_c ) {
        libxsmm_aarch64_instruction_alu_set_imm64( &x->opcode, LIBXSMM_SME_BG_X(17), l_c );
        libxsmm_aarch64_instruction_alu_compute_madd( &x->opcode, LIBXSMM_AARCH64_INSTR_GP_MADD, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(15),
                                                      LIBXSMM_SME_BG_X(12), LIBXSMM_SME_BG_X(16) );
      } else {
        libxsmm_sme_bg_mov( &x->opcode, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(12) );
      }
      libxsmm_sme_bg_ldst( &x->opcode, 0, l_s + 2, 2, 0, LIBXSMM_SME_BG_X(16), 0, LIBXSMM_AARCH64_SVE_REG_P14 );
      if ( 0 != l_has_last[l_c % 2] ) libxsmm_sme_bg_op_dep( x, l_ldc, l_last[l_c % 2], 0 );
      libxsmm_sme_bg_op_end( x, l_ldc, 0 );
      l_f1 = libxsmm_sme_bg_op_begin( x );
      libxsmm_sme_bg_fadd( &x->opcode, l_s, l_s, l_s + 2 );
      libxsmm_sme_bg_op_dep( x, l_f1, l_ldd, x->d_load );
      libxsmm_sme_bg_op_dep( x, l_f1, l_ldc, x->d_load );
      libxsmm_sme_bg_op_end( x, l_f1, 1 );
      l_f2 = libxsmm_sme_bg_op_begin( x );
      libxsmm_sme_bg_fadd( &x->opcode, l_s + 1, l_s + 1, l_s + 3 );
      libxsmm_sme_bg_op_dep( x, l_f2, l_ldd, x->d_load );
      libxsmm_sme_bg_op_dep( x, l_f2, l_ldc, x->d_load );
      libxsmm_sme_bg_op_end( x, l_f2, 1 );
    }
    l_st = libxsmm_sme_bg_op_begin( x );
    if ( 0 != l_c ) {
      libxsmm_aarch64_instruction_alu_set_imm64( &x->opcode, LIBXSMM_SME_BG_X(17), l_c );
      libxsmm_aarch64_instruction_alu_compute_madd( &x->opcode, LIBXSMM_AARCH64_INSTR_GP_MADD, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(15),
                                                    LIBXSMM_SME_BG_X(12), LIBXSMM_SME_BG_X(16) );
    } else {
      libxsmm_sme_bg_mov( &x->opcode, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(12) );
    }
    libxsmm_sme_bg_ldst( &x->opcode, 1, l_s, 2, 0, LIBXSMM_SME_BG_X(16), 0, LIBXSMM_AARCH64_SVE_REG_P14 );
    if ( 0 == x->beta0 ) {
      libxsmm_sme_bg_op_dep( x, l_st, l_f1, LIBXSMM_SME_BG_D_FADD );
      libxsmm_sme_bg_op_dep( x, l_st, l_f2, LIBXSMM_SME_BG_D_FADD );
    } else {
      libxsmm_sme_bg_op_dep( x, l_st, l_ldd, x->d_load );
    }
    libxsmm_sme_bg_op_end( x, l_st, 0 );
    l_last[l_c % 2] = l_st;
    l_has_last[l_c % 2] = 1;
  }
}

/* greedy list scheduling of ops [o0, o1) (priority = creation order) into k-steps [k0, k1):
 * <= max_extra ops and <= 1 vector-pipe op per k-step (including what used_n/used_v hold) */
LIBXSMM_API_INLINE int libxsmm_sme_bg_schedule( libxsmm_sme_bg_ctx* x, unsigned int o0, unsigned int o1, unsigned int k0, unsigned int k1 ) {
  unsigned int l_left = o1 - o0, l_kk, l_o, l_d;
  for ( l_o = o0; l_o < o1; ++l_o ) x->ops[l_o].k = -1;
  for ( l_kk = k0; l_kk < k1 && 0 != l_left; ++l_kk ) {
    unsigned int l_nvec = x->used_v[l_kk], l_nex = x->used_n[l_kk];
    for ( l_o = o0; l_o < o1; ++l_o ) {
      libxsmm_sme_bg_op* l_op = &x->ops[l_o];
      int l_ok = 1;
      if ( l_nex >= x->max_extra ) break;
      if ( l_op->k >= 0 ) continue;
      for ( l_d = 0; l_d < l_op->d_cnt; ++l_d ) {
        const libxsmm_sme_bg_op* l_dep = &x->ops[x->deps[l_op->d_off + 2 * l_d]];
        if ( l_dep->k < 0 || (unsigned int)l_dep->k + x->deps[l_op->d_off + 2 * l_d + 1] > l_kk ) {
          l_ok = 0;
          break;
        }
      }
      if ( 0 == l_ok || (0 != l_op->vec && l_nvec >= 1) ) continue;
      l_op->k = (int)l_kk;
      l_nvec += (unsigned int)l_op->vec;
      ++l_nex;
      --l_left;
    }
    x->used_v[l_kk] = (unsigned char)l_nvec;
    x->used_n[l_kk] = (unsigned char)l_nex;
  }
  return (0 == l_left) ? 0 : -1;
}

/* dump-mode block with a fully unrolled k-loop and scheduled background work (C update of the
 * predecessor p, transposition of G k-groups of the next panel); emit = 0: check only */
LIBXSMM_API_INLINE int libxsmm_sme_bg_block_sched( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, const libxsmm_sme_bg_prev* p,
                                                   unsigned int emit ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int K = x->k, D = LIBXSMM_SME_BG_DIST;
  const unsigned int l_nf = b->tm * b->tn;
  unsigned int l_ntr, l_pa, l_pb, l_kk, l_i, l_o;
  x->nops = 0; x->nwords = 0; x->ndeps = 0;
  x->scr_last_n[0] = x->scr_last_n[1] = 0;
  x->plast_n[0] = x->plast_n[1] = 0;
  memset( x->used_n, 0, K );
  memset( x->used_v, 0, K );
  if ( 0 != x->G ) libxsmm_sme_bg_tr_ops( x, x->G, x->pred );
  l_ntr = x->nops;
  if ( NULL != p ) libxsmm_sme_bg_cu_ops( x, p->nb );
  if ( 0 != c->last_error ) return -1;
  if ( 0 != libxsmm_sme_bg_schedule( x, 0, l_ntr, LIBXSMM_SME_BG_TR_START, K ) ) return -1;
  if ( 0 != libxsmm_sme_bg_schedule( x, l_ntr, x->nops, LIBXSMM_SME_BG_CU_START, K ) ) return -1;
  if ( 0 == emit ) return 0;

  libxsmm_sme_bg_oppreds( x, b, &l_pa, &l_pb );
  libxsmm_aarch64_instruction_sme_zero( c, 0xff );
  if ( NULL != p ) libxsmm_sme_bg_bg_setup( x, p );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(10), LIBXSMM_SME_BG_X(21) );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(11), LIBXSMM_SME_BG_X(22) );
  for ( l_kk = 0; l_kk < LIBXSMM_MIN(D, K); ++l_kk ) {
    libxsmm_sme_bg_load( x, b, l_kk, l_pa, l_pb );
  }
  for ( l_kk = 0; l_kk < K; ++l_kk ) {
    unsigned int l_ex[16], l_nex = 0, l_slot[16], l_ordered = 0, l_nrest = 0;
    if ( l_kk + D < K ) {
      libxsmm_sme_bg_load( x, b, l_kk + D, l_pa, l_pb );
    }
    for ( l_o = 0; l_o < x->nops; ++l_o ) {
      if ( x->ops[l_o].k == (int)l_kk && l_nex < 16 ) l_ex[l_nex++] = l_o;
    }
    /* vector-pipe ops after the first FMOPA, the others spread over the following ones; keep
     * creation order if a dependency within this k-step would be reversed */
    for ( l_i = 0; l_i < l_nex; ++l_i ) {
      if ( 0 != x->ops[l_ex[l_i]].vec ) {
        l_slot[l_i] = 0;
      } else {
        l_slot[l_i] = (l_nf > 1) ? (1 + l_nrest) % l_nf : 0;
        ++l_nrest;
      }
    }
    for ( l_i = 0; l_i < l_nex && 0 == l_ordered; ++l_i ) {
      const libxsmm_sme_bg_op* l_op = &x->ops[l_ex[l_i]];
      unsigned int l_d, l_j;
      for ( l_d = 0; l_d < l_op->d_cnt; ++l_d ) {
        const unsigned int l_dep = x->deps[l_op->d_off + 2 * l_d];
        for ( l_j = 0; l_j < l_nex; ++l_j ) {
          if ( l_ex[l_j] == l_dep && (l_slot[l_j] > l_slot[l_i] || (l_slot[l_j] == l_slot[l_i] && l_j > l_i)) ) l_ordered = 1;
        }
      }
    }
    for ( l_i = 0; l_i < l_nf; ++l_i ) {
      unsigned int l_j;
      libxsmm_sme_bg_fmopa_one( x, b, l_kk, l_i );
      for ( l_j = 0; l_j < l_nex; ++l_j ) {
        if ( (0 != l_ordered) ? (0 == l_i) : (l_slot[l_j] == l_i) ) libxsmm_sme_bg_op_emit( x, l_ex[l_j] );
      }
    }
  }
  libxsmm_sme_bg_dump_readout( x, b );
  return 0;
}

/* block structure of a panel and the transposition share G of each block */
LIBXSMM_API_INLINE void libxsmm_sme_bg_bgt_layout( libxsmm_sme_bg_ctx* x ) {
  const unsigned int l_nbl = x->fcnt + ((0 != x->medge) ? 1 : 0);
  const unsigned int l_nkg = libxsmm_sme_bg_cdiv( x->k, 16 );
  x->G = libxsmm_sme_bg_cdiv( l_nkg, l_nbl );
  x->pred = (0 != x->k % 16 || x->G * l_nbl != l_nkg) ? 1 : 0;
}

/* scheduling distances from conservative to aggressive, the first that fits for the full and the
 * edge block is used */
LIBXSMM_API_INLINE int libxsmm_sme_bg_bgt_feasible( libxsmm_sme_bg_ctx* x ) {
  const unsigned int l_dist[3][3] = { { 12, 8, 2 }, { 4, 4, 2 }, { 3, 4, 3 } };
  libxsmm_sme_bg_block l_full, l_edge;
  libxsmm_sme_bg_prev l_p;
  unsigned int l_i;
  if ( 0 != x->tb || x->nfull < 2 ) return 0;
  libxsmm_sme_bg_bgt_layout( x );
  if ( x->G > LIBXSMM_SME_BG_MAX_G ) return 0;
  l_full.mb = 32; l_full.nb = 32; l_full.tm = 2; l_full.tn = 2;
  l_edge.mb = x->medge; l_edge.nb = 32; l_edge.tm = libxsmm_sme_bg_cdiv( x->medge, 16 ); l_edge.tn = 2;
  l_p.mb = 32; l_p.nb = 32; l_p.kind = LIBXSMM_SME_BG_PREV_WITHIN;
  for ( l_i = 0; l_i < 3; ++l_i ) {
    x->d_load = l_dist[l_i][0];
    x->d_mem = l_dist[l_i][1];
    x->max_extra = l_dist[l_i][2];
    if ( 0 != libxsmm_sme_bg_block_sched( x, &l_full, &l_p, 0 ) ) continue;
    x->op_words = x->nwords;   /* background instruction words of a full block */
    if ( 0 != x->medge && 0 != libxsmm_sme_bg_block_sched( x, &l_edge, &l_p, 0 ) ) continue;
    return 1;
  }
  return 0;
}

/* all blocks in dump mode with background transposition of op(B) (tb = N); swap: the problem is
 * C^T = B A (TT), ZA holds C^T */
LIBXSMM_API_INLINE void libxsmm_sme_bg_bgt_all( libxsmm_sme_bg_ctx* x, unsigned int abase ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int G = x->G;
  libxsmm_sme_bg_block l_full, l_edge;
  libxsmm_sme_bg_prev l_p;
  unsigned int l_prev_nb = 32;
  l_full.mb = 32; l_full.nb = 32; l_full.tm = 2; l_full.tn = 2;
  l_edge.mb = x->medge; l_edge.nb = 32; l_edge.tm = libxsmm_sme_bg_cdiv( x->medge, 16 ); l_edge.tn = 2;
  x->b_stride_reg = LIBXSMM_AARCH64_GP_REG_UNDEF;
  x->b_stride_imm = 128;
  /* up front: panel 0 -> PB0, edge panel -> PBE (through ZA) */
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(28), LIBXSMM_SME_BG_X(27), x->pb0 );
  libxsmm_sme_bg_pack( x, LIBXSMM_SME_BG_X(1), x->ldb, x->k, 32, LIBXSMM_SME_BG_X(28), 128 );
  if ( 0 != x->nedge ) {
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(1), 4LL * 32 * x->nfull * x->ldb );
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(9), LIBXSMM_SME_BG_X(27), x->pbe );
    libxsmm_sme_bg_pack( x, LIBXSMM_SME_BG_X(16), x->ldb, x->k, x->nedge, LIBXSMM_SME_BG_X(9), 128 );
  }
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(28), LIBXSMM_SME_BG_X(27), x->pb0 );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(19), LIBXSMM_SME_BG_X(1), 4LL * 32 * x->ldb );   /* source of panel 1 */
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(7), LIBXSMM_SME_BG_X(27), x->dummy_off );
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_AARCH64_GP_REG_X30, 128 );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(20), LIBXSMM_SME_BG_X(2) );
  libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(24), x->nfull );
  /* per panel: op(B) from x28; transposition source x8: the next panel, for the last panel panel 0
   * again (harmless): x8 = x1 + (x19 - x1) * [x24 > 1], [x24 > 1] = (x24 - 1) / (x24 - 1) (udiv by 0 is 0) */
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(21), abase );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(20) );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(22), LIBXSMM_SME_BG_X(28) );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(24), -1 );
  libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_UDIV, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(16), 0 );
  libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_SUB_SR, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(19), LIBXSMM_SME_BG_X(1), 0 );
  libxsmm_aarch64_instruction_alu_compute_madd( c, LIBXSMM_AARCH64_INSTR_GP_MADD, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(17),
                                                LIBXSMM_SME_BG_X(1), LIBXSMM_SME_BG_X(8) );
  /* destination x9: the other panel buffer */
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(27), x->pb0 );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(27), x->pb1 );
  libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_EOR_SR, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(17), 0 );
  libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_EOR_SR, LIBXSMM_SME_BG_X(9), LIBXSMM_SME_BG_X(28), LIBXSMM_SME_BG_X(16), 0 );
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(26), 0 );
  l_p.mb = (0 != x->swap) ? 32 : x->last_mb;
  l_p.nb = (0 != x->swap) ? x->last_mb : 32;
  l_p.kind = LIBXSMM_SME_BG_PREV_PANEL;
  libxsmm_sme_bg_block_sched( x, &l_full, &l_p, 1 );
#define LIBXSMM_SME_BG_ADVANCE() \
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(21), LIBXSMM_SME_BG_X(21), 128 ); \
  if ( 0 != x->swap ) { \
    libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_ADD_SR, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(5), 5 ); \
  } else { \
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(23), 128 ); \
  }
  LIBXSMM_SME_BG_ADVANCE();
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(8), LIBXSMM_SME_BG_X(8), 64 * G );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(9), LIBXSMM_SME_BG_X(9), 2048 * G );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(26), LIBXSMM_SME_BG_X(26), 16 * G );
  l_p.mb = 32; l_p.nb = 32; l_p.kind = LIBXSMM_SME_BG_PREV_WITHIN;
  if ( x->fcnt > 1 ) {
    if ( x->fcnt > 2 ) {
      libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), x->fcnt - 1 );
    }
    libxsmm_sme_bg_block_sched( x, &l_full, &l_p, 1 );
    LIBXSMM_SME_BG_ADVANCE();
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(8), LIBXSMM_SME_BG_X(8), 64 * G );
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(9), LIBXSMM_SME_BG_X(9), 2048 * G );
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(26), LIBXSMM_SME_BG_X(26), 16 * G );
    if ( x->fcnt > 2 ) {
      libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), 1 );
    }
  }
  if ( 0 != x->medge ) {
    libxsmm_sme_bg_block_sched( x, &l_edge, &l_p, 1 );
  }
  /* next panel: toggle the buffers, advance the transposition source */
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(27), x->pb0 );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(17), LIBXSMM_SME_BG_X(27), x->pb1 );
  libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_EOR_SR, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(17), 0 );
  libxsmm_sme_bg_alu_reg( c, LIBXSMM_AARCH64_INSTR_GP_EOR_SR, LIBXSMM_SME_BG_X(28), LIBXSMM_SME_BG_X(28), LIBXSMM_SME_BG_X(16), 0 );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(19), LIBXSMM_SME_BG_X(19), 4LL * 32 * x->ldb );
  /* predecessor of the next panel's first block: the last block of this panel */
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(7), LIBXSMM_SME_BG_X(20), (0 != x->swap) ? 4LL * x->last_i0 * x->ldc : 4LL * x->last_i0 );
  libxsmm_sme_bg_mov( c, LIBXSMM_AARCH64_GP_REG_X30, LIBXSMM_SME_BG_X(5) );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(20), LIBXSMM_SME_BG_X(20), (0 != x->swap) ? 128 : 4LL * 32 * x->ldc );
  libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(24), 1 );

  if ( 0 != x->nedge ) {   /* edge panel: op(B) from PBE, background C update only */
    libxsmm_sme_bg_block l_ef, l_ee;
    l_ef.mb = 32; l_ef.nb = x->nedge; l_ef.tm = 2; l_ef.tn = libxsmm_sme_bg_cdiv( x->nedge, 16 );
    l_ee.mb = x->medge; l_ee.nb = x->nedge; l_ee.tm = libxsmm_sme_bg_cdiv( x->medge, 16 ); l_ee.tn = l_ef.tn;
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(22), LIBXSMM_SME_BG_X(27), x->pbe );
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(21), abase );
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(20) );
    l_p.mb = (0 != x->swap) ? 32 : x->last_mb;
    l_p.nb = (0 != x->swap) ? x->last_mb : 32;
    l_p.kind = LIBXSMM_SME_BG_PREV_PANEL;
    libxsmm_sme_bg_block_dump( x, &l_ef, &l_p );
    LIBXSMM_SME_BG_ADVANCE();
    l_p.mb = (0 != x->swap) ? x->nedge : 32;
    l_p.nb = (0 != x->swap) ? 32 : x->nedge;
    l_p.kind = LIBXSMM_SME_BG_PREV_WITHIN;
    if ( x->fcnt > 1 ) {
      if ( x->fcnt > 2 ) {
        libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), x->fcnt - 1 );
      }
      libxsmm_sme_bg_block_dump( x, &l_ef, &l_p );
      LIBXSMM_SME_BG_ADVANCE();
      if ( x->fcnt > 2 ) {
        libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), 1 );
      }
    }
    if ( 0 != x->medge ) {
      libxsmm_sme_bg_block_dump( x, &l_ee, &l_p );
    }
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(7), LIBXSMM_SME_BG_X(20), (0 != x->swap) ? 4LL * x->last_i0 * x->ldc : 4LL * x->last_i0 );
    libxsmm_sme_bg_mov( c, LIBXSMM_AARCH64_GP_REG_X30, LIBXSMM_SME_BG_X(5) );
    l_prev_nb = x->nedge;
  }
#undef LIBXSMM_SME_BG_ADVANCE
  /* the last block of the call: synchronous update */
  libxsmm_sme_bg_prev_ptr( x, LIBXSMM_SME_BG_PREV_PANEL );
  if ( 0 != x->swap ) {
    libxsmm_sme_bg_sync_update( x, l_prev_nb, x->last_mb );
  } else {
    libxsmm_sme_bg_sync_update( x, x->last_mb, l_prev_nb );
  }
}

/* ------------------------------------------------------------------------------------ ZA mode */
/* For smaller k, or when dump mode would need more blocks, C is computed in blocks of up to 4 tiles
 * (16x64, 32x32, 48x16, 64x16) in row strips that minimize the number of blocks (a block costs >= 4
 * cycles per k-step whatever its tile count). C (beta = 1) is loaded into z16-z31 in rounds inside the
 * k-loop and added to ZA with fadd za.s[.., vgx4] (~1 cycle per 4 vectors, against ~1.5 cycles per
 * vector loading ZA slices directly); a single-block kernel loads C early and adds it after the
 * k-loop (its C loads wait for the C stores of the previous call). ZA is read out with mova into
 * alternating register banks, so the next mova does not wait for the stores. */
typedef struct libxsmm_sme_bg_strip {
  unsigned int i0, mb, tm, width;     /* first row, rows, tile rows, block width */
} libxsmm_sme_bg_strip;

/* row strips of 16/32/48/64 rows (blocks of 16x64 / 32x32 / 48x16 / 64x16); returns the number of
 * strips (0: m too large), nblocks = number of blocks */
LIBXSMM_API_INLINE unsigned int libxsmm_sme_bg_plan_strips( unsigned int m, unsigned int n, libxsmm_sme_bg_strip* strips, unsigned int* nblocks ) {
  const unsigned int l_units = libxsmm_sme_bg_cdiv( m, 16 );
  const unsigned int l_width[5] = { 0, 64, 32, 16, 16 };
  const unsigned int l_hs[4] = { 2, 4, 1, 3 };
  unsigned int l_best[65], l_choice[65], l_u, l_i, l_ns = 0, l_i0 = 0;
  *nblocks = 0;
  if ( 0 == l_units || l_units > 64 ) return 0;
  l_best[l_units] = 0;
  for ( l_u = l_units; l_u-- > 0; ) {   /* cost in 1/100 blocks, 48-row strips slightly penalized */
    l_best[l_u] = 0xffffffff;
    l_choice[l_u] = 1;
    for ( l_i = 0; l_i < 4; ++l_i ) {
      const unsigned int l_h = l_hs[l_i];
      if ( l_u + l_h <= l_units ) {
        const unsigned int l_cost = 100 * libxsmm_sme_bg_cdiv( n, l_width[l_h] ) + ((3 == l_h) ? 1 : 0) + l_best[l_u + l_h];
        if ( l_cost < l_best[l_u] ) {
          l_best[l_u] = l_cost;
          l_choice[l_u] = l_h;
        }
      }
    }
  }
  /* m % 16 != 0, m <= 64: rather one strip of full columns, if that costs at most one block more */
  if ( 0 != m % 16 && m <= 64 && l_choice[0] != l_units && 100 * libxsmm_sme_bg_cdiv( n, l_width[l_units] ) <= l_best[0] + 100 ) {
    l_choice[0] = l_units;
  }
  for ( l_u = 0; l_u < l_units; l_u += l_choice[l_u] ) {
    const unsigned int l_h = l_choice[l_u];
    strips[l_ns].i0 = l_i0;
    strips[l_ns].mb = LIBXSMM_MIN( 16 * l_h, m - l_i0 );
    strips[l_ns].tm = libxsmm_sme_bg_cdiv( strips[l_ns].mb, 16 );
    strips[l_ns].width = l_width[l_h];
    *nblocks += libxsmm_sme_bg_cdiv( n, l_width[l_h] );
    l_i0 += strips[l_ns].mb;
    ++l_ns;
  }
  return l_ns;
}

/* C registers: column q (0..3 of a group of 4) of tile row ib; cnreg vectors per column (48 rows: 4) */
LIBXSMM_API_INLINE unsigned int libxsmm_sme_bg_cnreg( const libxsmm_sme_bg_block* b ) {
  return (3 == b->tm) ? 4 : b->tm;
}

LIBXSMM_API_INLINE unsigned int libxsmm_sme_bg_creg( const libxsmm_sme_bg_block* b, unsigned int q, unsigned int ib ) {
  return 16 + q + ((1 == b->tm) ? 0 : ((2 == b->tm) ? 8 : 4)) * ib;
}

/* C group g: columns 16jb+s+4q (q = 0..3), i.e. the vgx4 group of slices s, s+4, s+8, s+12 of the tiles
 * of tile column jb; a round of the C initialization holds 4 / cnreg groups (16 registers) */
LIBXSMM_API_INLINE int libxsmm_sme_bg_cgroup( const libxsmm_sme_bg_block* b, unsigned int g, unsigned int* jb, unsigned int* s ) {
  unsigned int l_jb, l_s, l_i = 0;
  for ( l_jb = 0; l_jb < b->tn; ++l_jb ) {
    for ( l_s = 0; l_s < 4; ++l_s ) {
      if ( 16 * l_jb + l_s < b->nb ) {
        if ( l_i == g ) {
          *jb = l_jb;
          *s = l_s;
          return 1;
        }
        ++l_i;
      }
    }
  }
  return 0;
}

/* load i of round r: C column 16jb+s+4q of group i/4 of the round (q = i%4) -> registers z16.. */
LIBXSMM_API_INLINE void libxsmm_sme_bg_cload( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, unsigned int pc, unsigned int r, unsigned int i ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int l_cn = libxsmm_sme_bg_cnreg( b ), l_gi = i / 4, l_q = i % 4;
  unsigned int l_jb, l_s, l_col, l_ib;
  if ( 0 == libxsmm_sme_bg_cgroup( b, r * (4 / l_cn) + l_gi, &l_jb, &l_s ) ) return;
  l_col = 16 * l_jb + l_s + 4 * l_q;
  if ( l_col >= b->nb ) {
    for ( l_ib = 0; l_ib < l_cn; ++l_ib ) {
      libxsmm_sme_bg_zeroz( c, libxsmm_sme_bg_creg( b, l_q, l_ib ) + 4 * l_gi );
    }
    return;
  }
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(23), 4LL * l_col * x->ldc );
  libxsmm_sme_bg_ldst( c, 0, libxsmm_sme_bg_creg( b, l_q, 0 ) + 4 * l_gi, l_cn, (l_cn > 1) ? 1 : 0, LIBXSMM_SME_BG_X(16), 0, pc );
}

LIBXSMM_API_INLINE unsigned int libxsmm_sme_bg_cnloads( const libxsmm_sme_bg_block* b, unsigned int r ) {
  const unsigned int l_per = 4 / libxsmm_sme_bg_cnreg( b );
  unsigned int l_gi, l_jb, l_s, l_n = 0;
  for ( l_gi = 0; l_gi < l_per; ++l_gi ) {
    if ( 0 != libxsmm_sme_bg_cgroup( b, r * l_per + l_gi, &l_jb, &l_s ) ) l_n += 4;
  }
  return l_n;
}

/* adds of round r: ZA vgx4 group (tile (ib, jb), slices s + 4q) += C columns 16jb+s+4q */
LIBXSMM_API_INLINE void libxsmm_sme_bg_cadds( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, unsigned int r ) {
  const unsigned int l_per = 4 / libxsmm_sme_bg_cnreg( b );
  unsigned int l_gi, l_jb, l_s, l_ib;
  for ( l_gi = 0; l_gi < l_per; ++l_gi ) {
    if ( 0 == libxsmm_sme_bg_cgroup( b, r * l_per + l_gi, &l_jb, &l_s ) ) break;
    for ( l_ib = 0; l_ib < b->tm; ++l_ib ) {
      const unsigned int l_vec = 4 * l_s + libxsmm_sme_bg_tile( b, l_ib, l_jb );
      libxsmm_aarch64_instruction_sme2_za_vgx4( x->code, LIBXSMM_AARCH64_INSTR_SME2_FADD_ZA_VGX4_S,
                                                (l_vec < 8) ? LIBXSMM_AARCH64_GP_REG_W8 : LIBXSMM_AARCH64_GP_REG_W9, l_vec % 8,
                                                libxsmm_sme_bg_creg( b, 0, l_ib ) + 4 * l_gi );
    }
  }
}

/* C contiguous (ldc == m, m % 16 != 0) and the block covers whole columns: C is moved as a flat image
 * of 64-byte granules (each granule stored once: several stores into one granule serialize) */
LIBXSMM_API_INLINE int libxsmm_sme_bg_flat( const libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b ) {
  return (x->ldc == x->m && 0 != x->m % 16 && b->mb == x->m) ? 1 : 0;
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_whilelt_n( libxsmm_generated_code* c, unsigned int pred, unsigned int n ) {
  libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(17), n );
  libxsmm_sme_bg_whilelt( c, pred, 1, LIBXSMM_SME_BG_XZR, LIBXSMM_SME_BG_X(17) );
}

/* flat store for m < 16: the columns (m lanes each) are compacted into the flat image with splice,
 * z12 = pending lanes (*r valid), z11 = output, x16 = destination */
LIBXSMM_API_INLINE void libxsmm_sme_bg_flat_append( libxsmm_sme_bg_ctx* x, unsigned int z, unsigned int* r ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int L = x->m;
  if ( 0 == *r ) {
    libxsmm_sme_bg_movz( c, 12, z );
    *r = L;
    return;
  }
  libxsmm_sme_bg_whilelt_n( c, LIBXSMM_AARCH64_SVE_REG_P5, *r );
  if ( *r + L < 16 ) {
    libxsmm_aarch64_instruction_sve_ext_splice( c, LIBXSMM_AARCH64_INSTR_SVE_SPLICE_S, z, 5, 12 );
    *r += L;
    return;
  }
  libxsmm_sme_bg_movz( c, 11, 12 );
  libxsmm_aarch64_instruction_sve_ext_splice( c, LIBXSMM_AARCH64_INSTR_SVE_SPLICE_S, z, 5, 11 );
  libxsmm_sme_bg_ldst( c, 1, 11, 1, 0, LIBXSMM_SME_BG_X(16), 0, LIBXSMM_AARCH64_SVE_REG_P0 );
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(16), 64 );
  if ( *r + L > 16 ) {   /* the rest of column z: rotate its remaining lanes down */
    libxsmm_sme_bg_movz( c, 12, z );
    libxsmm_aarch64_instruction_sve_ext_splice( c, LIBXSMM_AARCH64_INSTR_SVE_EXT_V, 12, 4 * (16 - *r), 12 );
  }
  *r = *r + L - 16;
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_cstore_flat1( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_r = 0, l_jb, l_s0, l_q;
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(23) );
  for ( l_jb = 0; l_jb < b->tn; ++l_jb ) {
    for ( l_s0 = 0; l_s0 < 16; l_s0 += 4 ) {
      if ( 16 * l_jb + l_s0 >= b->nb ) continue;
      libxsmm_sme_bg_mova( c, 0, 0, libxsmm_sme_bg_tile( b, 0, l_jb ), l_s0 / 4, 16 );
      for ( l_q = 0; l_q < 4; ++l_q ) {
        if ( 16 * l_jb + l_s0 + l_q < b->nb ) libxsmm_sme_bg_flat_append( x, 16 + l_q, &l_r );
      }
    }
  }
  if ( 0 != l_r ) {
    libxsmm_sme_bg_whilelt_n( c, LIBXSMM_AARCH64_SVE_REG_P5, l_r );
    libxsmm_sme_bg_ldst( c, 1, 12, 1, 0, LIBXSMM_SME_BG_X(16), 0, LIBXSMM_AARCH64_SVE_REG_P5 );
  }
}

/* flat store for 16 < m <= 64: every output granule comes from at most two columns j, j+1: ext (column
 * j from row r0) + splice (rows 0.. of column j+1), no chain between granules; granules in z0-z7, stored
 * 4 at a time; columns of the current mova group: C registers, column j+1 of the next group: z8 */
LIBXSMM_API_INLINE void libxsmm_sme_bg_flat_flush( libxsmm_sme_bg_ctx* x, const unsigned int* pend, unsigned int npend, unsigned int total ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_g0, l_i;
  if ( 0 == npend ) return;
  l_g0 = pend[0] / 8;   /* entries: 8 * granule + register */
  libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(23), 64LL * l_g0 );
  if ( 4 == npend && total >= 16 * (l_g0 + 4) ) {
    libxsmm_sme_bg_ldst( c, 1, pend[0] % 8, 4, 0, LIBXSMM_SME_BG_X(16), 0, LIBXSMM_AARCH64_SVE_REG_P8 );
    return;
  }
  for ( l_i = 0; l_i < npend; ++l_i ) {
    const unsigned int l_rem = total - 16 * (pend[l_i] / 8);
    unsigned int l_p = LIBXSMM_AARCH64_SVE_REG_P0;
    if ( l_rem < 16 ) {
      libxsmm_sme_bg_whilelt_n( c, LIBXSMM_AARCH64_SVE_REG_P6, l_rem );
      l_p = LIBXSMM_AARCH64_SVE_REG_P6;
    }
    libxsmm_sme_bg_ldst( c, 1, pend[l_i] % 8, 1, 0, LIBXSMM_SME_BG_X(16), (int)l_i, l_p );
  }
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_cstore_flat_wide( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int m = x->m, tm = b->tm, l_total = m * b->nb, l_ng = libxsmm_sme_bg_cdiv( l_total, 16 );
  unsigned int l_loaded = 0xffffffff, l_pend[4], l_npend = 0, l_g, l_ib;
  for ( l_g = 0; l_g < l_ng; ++l_g ) {
    const unsigned int l_j = (16 * l_g) / m, l_r0 = (16 * l_g) % m, L = m - l_r0;
    const unsigned int l_grp = (l_j / 16) * 4 + (l_j % 16) / 4;   /* mova group: (tile column, slice group) */
    const unsigned int l_ibr = l_r0 / 16, l_sh = l_r0 % 16;
    const unsigned int l_t = 4 * ((l_g / 4) % 2) + l_g % 4;
    if ( l_grp != l_loaded ) {
      for ( l_ib = 0; l_ib < tm; ++l_ib ) {
        libxsmm_sme_bg_mova( c, 0, 0, libxsmm_sme_bg_tile( b, l_ib, l_j / 16 ), (l_j % 16) / 4, libxsmm_sme_bg_creg( b, 0, l_ib ) );
      }
      l_loaded = l_grp;
    }
    libxsmm_sme_bg_movz( c, l_t, libxsmm_sme_bg_creg( b, l_j % 4, l_ibr ) );
    if ( L >= 16 ) {
      if ( 0 != l_sh ) {
        libxsmm_aarch64_instruction_sve_ext_splice( c, LIBXSMM_AARCH64_INSTR_SVE_EXT_V, libxsmm_sme_bg_creg( b, l_j % 4, l_ibr + 1 ), 4 * l_sh, l_t );
      }
    } else {
      const unsigned int l_nxt = (l_ibr + 1 < tm && l_sh + L > 16) ? libxsmm_sme_bg_creg( b, l_j % 4, l_ibr + 1 ) : libxsmm_sme_bg_creg( b, l_j % 4, l_ibr );
      if ( 0 != l_sh ) {
        libxsmm_aarch64_instruction_sve_ext_splice( c, LIBXSMM_AARCH64_INSTR_SVE_EXT_V, l_nxt, 4 * l_sh, l_t );
      }
      if ( l_j + 1 < b->nb ) {
        const unsigned int l_j1 = l_j + 1, l_grp1 = (l_j1 / 16) * 4 + (l_j1 % 16) / 4;
        unsigned int l_v0 = libxsmm_sme_bg_creg( b, l_j1 % 4, 0 );
        if ( l_grp1 != l_loaded ) {
          libxsmm_aarch64_instruction_sme_mova_slice( c, LIBXSMM_AARCH64_INSTR_SME_MOVA_H_SLICE_TO_VECTOR, libxsmm_sme_bg_tile( b, 0, l_j1 / 16 ),
                                                      LIBXSMM_AARCH64_GP_REG_W12 + (l_j1 % 16) / 4, l_j1 % 4, LIBXSMM_AARCH64_SVE_REG_P0, 8 );
          l_v0 = 8;
        }
        libxsmm_sme_bg_whilelt_n( c, LIBXSMM_AARCH64_SVE_REG_P5, L );
        libxsmm_aarch64_instruction_sve_ext_splice( c, LIBXSMM_AARCH64_INSTR_SVE_SPLICE_S, l_v0, 5, l_t );
      }
    }
    l_pend[l_npend++] = 8 * l_g + l_t;
    if ( 4 == l_npend ) {
      libxsmm_sme_bg_flat_flush( x, l_pend, l_npend, l_total );
      l_npend = 0;
    }
  }
  libxsmm_sme_bg_flat_flush( x, l_pend, l_npend, l_total );
}

/* 1x1-tile block: all of C in z16-z31 (column s+4q in z16+4s+q), loaded before the k-loop, added after it;
 * flat (C contiguous, nb*m <= 256): the flat image in z16.., the columns are cut out at the end (ext) */
LIBXSMM_API_INLINE void libxsmm_sme_bg_clate( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, unsigned int pc, unsigned int add ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_s, l_q;
  if ( 0 != libxsmm_sme_bg_flat( x, b ) && b->nb * x->m <= 256 ) {
    const unsigned int l_nflat = libxsmm_sme_bg_cdiv( b->nb * x->m, 16 );
    unsigned int l_v;
    if ( 0 == add ) {
      for ( l_v = 0; l_v < l_nflat; ++l_v ) {
        const unsigned int l_rem = b->nb * x->m - 16 * l_v;
        unsigned int l_p = LIBXSMM_AARCH64_SVE_REG_P0;
        if ( l_rem < 16 ) {
          libxsmm_sme_bg_whilelt_n( c, LIBXSMM_AARCH64_SVE_REG_P5, l_rem );
          l_p = LIBXSMM_AARCH64_SVE_REG_P5;
        }
        if ( 0 == l_v % 8 ) {
          libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(23), 64LL * l_v );
        }
        libxsmm_sme_bg_ldst( c, 0, 16 + l_v, 1, 0, LIBXSMM_SME_BG_X(16), (int)(l_v % 8), l_p );
      }
    } else {
      for ( l_s = 0; l_s < 4; ++l_s ) {
        for ( l_q = 0; l_q < 4; ++l_q ) {
          const unsigned int l_col = l_s + 4 * l_q, l_z = 4 * l_s + l_q;
          if ( l_col >= b->nb ) {
            libxsmm_sme_bg_zeroz( c, l_z );
          } else if ( 0 == (l_col * x->m) % 16 ) {
            libxsmm_sme_bg_movz( c, l_z, 16 + (l_col * x->m) / 16 );
          } else {
            libxsmm_aarch64_instruction_sve_ext_splice( c, LIBXSMM_AARCH64_INSTR_SVE2_EXT_V, 16 + (l_col * x->m) / 16, 4 * ((l_col * x->m) % 16), l_z );
          }
        }
      }
      for ( l_s = 0; l_s < LIBXSMM_MIN(4, b->nb); ++l_s ) {
        libxsmm_aarch64_instruction_sme2_za_vgx4( c, LIBXSMM_AARCH64_INSTR_SME2_FADD_ZA_VGX4_S,
                                                  (l_s < 2) ? LIBXSMM_AARCH64_GP_REG_W8 : LIBXSMM_AARCH64_GP_REG_W9, (4 * l_s) % 8, 4 * l_s );
      }
    }
    return;
  }
  for ( l_s = 0; l_s < 4; ++l_s ) {
    if ( 0 != add ) {
      if ( l_s < b->nb ) {
        libxsmm_aarch64_instruction_sme2_za_vgx4( c, LIBXSMM_AARCH64_INSTR_SME2_FADD_ZA_VGX4_S,
                                                  (l_s < 2) ? LIBXSMM_AARCH64_GP_REG_W8 : LIBXSMM_AARCH64_GP_REG_W9, (4 * l_s) % 8, 16 + 4 * l_s );
      }
    } else {
      for ( l_q = 0; l_q < 4; ++l_q ) {
        const unsigned int l_col = l_s + 4 * l_q, l_r = 16 + 4 * l_s + l_q;
        if ( l_col >= b->nb ) {
          libxsmm_sme_bg_zeroz( c, l_r );
        } else {
          libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(23), 4LL * l_col * x->ldc );
          libxsmm_sme_bg_ldst( c, 0, l_r, 1, 0, LIBXSMM_SME_BG_X(16), 0, pc );
        }
      }
    }
  }
}

/* ZA -> C: 4 slices per mova into alternating register banks, then one store per C column */
LIBXSMM_API_INLINE void libxsmm_sme_bg_creadout( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, unsigned int pc ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int l_cn = libxsmm_sme_bg_cnreg( b );
  unsigned int l_nbk = 0, l_jb, l_s0, l_ib, l_q;
  if ( 0 != libxsmm_sme_bg_flat( x, b ) ) {
    if ( 1 == b->tm ) {
      libxsmm_sme_bg_cstore_flat1( x, b );
    } else {
      libxsmm_sme_bg_cstore_flat_wide( x, b );
    }
    return;
  }
  for ( l_jb = 0; l_jb < b->tn; ++l_jb ) {
    for ( l_s0 = 0; l_s0 < 16; l_s0 += 4 ) {
      int l_sh;
      if ( 16 * l_jb + l_s0 >= b->nb ) continue;
      if ( 1 == b->tm ) l_sh = 4 * (int)(l_nbk % 4);
      else if ( 2 == b->tm ) l_sh = 4 * (int)(l_nbk % 2);
      else l_sh = (0 == l_nbk % 2) ? 0 : -16;
      ++l_nbk;
      for ( l_ib = 0; l_ib < b->tm; ++l_ib ) {
        libxsmm_sme_bg_mova( c, 0, 0, libxsmm_sme_bg_tile( b, l_ib, l_jb ), l_s0 / 4, (unsigned int)((int)libxsmm_sme_bg_creg( b, 0, l_ib ) + l_sh) );
      }
      for ( l_q = 0; l_q < 4; ++l_q ) {
        const unsigned int l_col = 16 * l_jb + l_s0 + l_q;
        if ( l_col >= b->nb ) continue;
        libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(16), LIBXSMM_SME_BG_X(23), 4LL * l_col * x->ldc );
        libxsmm_sme_bg_ldst( c, 1, (unsigned int)((int)libxsmm_sme_bg_creg( b, l_q, 0 ) + l_sh), l_cn, (l_cn > 1) ? 1 : 0,
                             LIBXSMM_SME_BG_X(16), 0, pc );
      }
    }
  }
}

/* one block in ZA mode: op(A) from x21, op(B) from x22, C at x23; first: ZA is zero already;
 * single: the only block of the call */
LIBXSMM_API_INLINE void libxsmm_sme_bg_block_za( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_block* b, unsigned int first, unsigned int single ) {
  libxsmm_generated_code* c = x->code;
  const unsigned int K = x->k, D = LIBXSMM_SME_BG_DIST, U = LIBXSMM_SME_BG_KUNROLL;
  const unsigned int l_nf = b->tm * b->tn, l_cn = libxsmm_sme_bg_cnreg( b ), l_per = 4 / l_cn;
  const unsigned int l_late = (1 == b->tm && 1 == b->tn) ? 1 : 0;
  unsigned int l_ng = 0, l_nr, l_nin = 0, l_peel, l_niter = 0, l_pa, l_pb, l_pc, l_kk, l_i, l_r, l_l, l_jb;
  for ( l_jb = 0; l_jb < b->tn; ++l_jb ) l_ng += LIBXSMM_MIN( 4, b->nb - 16 * l_jb );
  l_nr = (0 != x->beta0 || 0 != l_late) ? 0 : libxsmm_sme_bg_cdiv( l_ng, l_per );
  if ( 0 == single ) {   /* round r inside the k-loop: C loads from k-step 12r on (2 per k-step), adds at 12r + 8 */
    while ( l_nin < l_nr && 12 * l_nin + 9 <= K ) ++l_nin;
  }
  libxsmm_sme_bg_oppreds( x, b, &l_pa, &l_pb );
  l_pc = libxsmm_sme_bg_pred( c, b->mb, l_cn, (1 == l_cn) ? LIBXSMM_AARCH64_SVE_REG_P3 : LIBXSMM_AARCH64_SVE_REG_P11 );
  if ( 0 == first ) libxsmm_aarch64_instruction_sme_zero( c, 0xff );
  if ( 0 != l_late && 0 == x->beta0 ) libxsmm_sme_bg_clate( x, b, l_pc, 0 );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(10), LIBXSMM_SME_BG_X(21) );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(11), LIBXSMM_SME_BG_X(22) );
  for ( l_kk = 0; l_kk < LIBXSMM_MIN(D, K); ++l_kk ) {
    libxsmm_sme_bg_load( x, b, l_kk, l_pa, l_pb );
  }
  if ( K <= 64 ) l_peel = K;   /* fully unrolled */
  else if ( 0 != single && 0 != l_nr ) l_peel = U;
  else if ( 0 != l_nin ) l_peel = libxsmm_sme_bg_rup( 12 * (l_nin - 1) + 9, U );
  else l_peel = 0;
  for ( l_kk = 0; l_kk < l_peel; ++l_kk ) {
    if ( l_kk + D < K ) {
      libxsmm_sme_bg_load( x, b, l_kk + D, l_pa, l_pb );
    }
    for ( l_i = 0; l_i < l_nf; ++l_i ) {
      libxsmm_sme_bg_fmopa_one( x, b, l_kk, l_i );
      if ( 0 != l_i ) continue;
      if ( 0 != single && 0 != l_nr && 0 == l_kk ) {
        for ( l_l = 0; l_l < libxsmm_sme_bg_cnloads( b, 0 ); ++l_l ) libxsmm_sme_bg_cload( x, b, l_pc, 0, l_l );
      }
      for ( l_r = 0; l_r < l_nin; ++l_r ) {
        for ( l_l = 0; l_l < libxsmm_sme_bg_cnloads( b, l_r ); ++l_l ) {
          if ( 12 * l_r + l_l / 2 == l_kk ) libxsmm_sme_bg_cload( x, b, l_pc, l_r, l_l );
        }
        if ( 12 * l_r + 8 == l_kk ) libxsmm_sme_bg_cadds( x, b, l_r );
      }
    }
  }
  if ( K > l_peel ) {
    if ( K - l_peel > D ) l_niter = (K - l_peel - D) / U;
    libxsmm_sme_bg_iters( x, b, l_peel, l_niter, 0, 0, l_pa, l_pb );
    for ( l_kk = l_peel + l_niter * U; l_kk < K; ++l_kk ) {
      if ( l_kk + D < K ) {
        libxsmm_sme_bg_load( x, b, l_kk + D, l_pa, l_pb );
      }
      for ( l_i = 0; l_i < l_nf; ++l_i ) {
        libxsmm_sme_bg_fmopa_one( x, b, l_kk, l_i );
      }
    }
  }
  /* what did not fit into the k-loop */
  if ( 0 != l_late && 0 == x->beta0 ) libxsmm_sme_bg_clate( x, b, l_pc, 1 );
  for ( l_r = (0 != single) ? 0 : l_nin; l_r < l_nr; ++l_r ) {
    if ( 0 == single || 0 != l_r ) {
      for ( l_l = 0; l_l < libxsmm_sme_bg_cnloads( b, l_r ); ++l_l ) libxsmm_sme_bg_cload( x, b, l_pc, l_r, l_l );
    }
    libxsmm_sme_bg_cadds( x, b, l_r );
  }
  libxsmm_sme_bg_creadout( x, b, l_pc );
}

/* ZA mode for all blocks: strip by strip (x19/x20 = op(A)/C at the strip, x21 = op(A) at the strip,
 * x22 = op(B) at the block, x23 = C block), loops over identical blocks of a strip and over identical
 * strips (x24) */
LIBXSMM_API_INLINE void libxsmm_sme_bg_za_all( libxsmm_sme_bg_ctx* x, unsigned int abase, unsigned int bbase ) {
  libxsmm_generated_code* c = x->code;
  libxsmm_sme_bg_strip l_strips[64];
  unsigned int l_nblocks, l_ns, l_st, l_scnt, l_run, l_first = 1;
  l_ns = libxsmm_sme_bg_plan_strips( x->m, x->n, l_strips, &l_nblocks );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(19), abase );
  libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(20), LIBXSMM_SME_BG_X(2) );
  for ( l_st = 0; l_st < l_ns; l_st += l_scnt ) {
    const libxsmm_sme_bg_strip* l_p = &l_strips[l_st];
    for ( l_scnt = 1; l_st + l_scnt < l_ns && l_strips[l_st + l_scnt].mb == l_p->mb && l_strips[l_st + l_scnt].width == l_p->width; ++l_scnt );
    if ( l_scnt > 1 ) {
      libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(24), l_scnt );
    }
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(21), LIBXSMM_SME_BG_X(19) );
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(20) );
    libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(22), bbase );
    for ( l_run = 0; l_run < 2; ++l_run ) {   /* blocks of full width, then the edge block */
      const unsigned int l_nb = (0 == l_run) ? l_p->width : x->n % l_p->width;
      const unsigned int l_cnt = (0 == l_run) ? x->n / l_p->width : ((0 != l_nb) ? 1 : 0);
      libxsmm_sme_bg_block l_b;
      if ( 0 == l_cnt ) continue;
      l_b.mb = l_p->mb; l_b.nb = l_nb; l_b.tm = l_p->tm; l_b.tn = libxsmm_sme_bg_cdiv( l_nb, 16 );
      if ( l_cnt > 1 ) {
        libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), l_cnt );
      }
      libxsmm_sme_bg_block_za( x, &l_b, (0 != l_first && 1 == l_cnt && 1 == l_scnt) ? 1 : 0, (1 == l_nblocks) ? 1 : 0 );
      l_first = 0;
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(22), LIBXSMM_SME_BG_X(22), 4LL * l_nb );
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(23), LIBXSMM_SME_BG_X(23), 4LL * l_nb * x->ldc );
      if ( l_cnt > 1 ) {
        libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(25), 1 );
      }
    }
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(19), LIBXSMM_SME_BG_X(19), 4LL * l_p->mb );
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(20), LIBXSMM_SME_BG_X(20), 4LL * l_p->mb );
    if ( l_scnt > 1 ) {
      libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(24), 1 );
    }
  }
}

/* ------------------------------------------------------------------------------------ planning */
/* the row-dependent part of the plan for m rows (the whole problem or a row group) */
LIBXSMM_API_INLINE int libxsmm_sme_bg_set_rows( libxsmm_sme_bg_ctx* x, unsigned int m ) {
  if ( m <= 32 || (m <= 64 && 0 != m % 16) ) return 0;
  x->m = m;
  x->fcnt = m / 32;
  x->medge = m % 32;
  x->last_i0 = (0 != x->medge) ? 32 * x->fcnt : 32 * (x->fcnt - 1);
  x->last_mb = (0 != x->medge) ? x->medge : 32;
  return 1;
}

/* fills the plan of the (possibly swapped) problem; returns 1 if dump mode applies */
LIBXSMM_API_INLINE int libxsmm_sme_bg_plan( libxsmm_sme_bg_ctx* x ) {
  if ( x->k < LIBXSMM_SME_BG_MINK || x->n <= 16 || 0 == libxsmm_sme_bg_set_rows( x, x->m ) ) return 0;
  x->nfull = x->n / 32;
  x->nedge = x->n % 32;
  x->mpad = libxsmm_sme_bg_rup( x->m, 64 );
  x->npad = libxsmm_sme_bg_rup( x->n, 64 );
  x->kpad = libxsmm_sme_bg_rup( x->k, 32 );
  return 1;
}

typedef struct libxsmm_sme_bg_variant {   /* plan of a row group */
  unsigned int m, G, pred, d_load, d_mem, max_extra, op_words, panel;
} libxsmm_sme_bg_variant;

/* plan of a row group of m rows, with (bgt) or without background transposition */
LIBXSMM_API_INLINE int libxsmm_sme_bg_variant_plan( libxsmm_sme_bg_ctx* x, unsigned int m, unsigned int bgt, libxsmm_sme_bg_variant* v ) {
  memset( v, 0, sizeof(*v) );
  if ( 0 == libxsmm_sme_bg_set_rows( x, m ) ) return 0;
  v->m = m;
  if ( 0 != bgt ) {
    if ( 0 == libxsmm_sme_bg_bgt_feasible( x ) ) return 0;
    v->G = x->G; v->pred = x->pred; v->op_words = x->op_words;
    v->d_load = x->d_load; v->d_mem = x->d_mem; v->max_extra = x->max_extra;
    v->panel = libxsmm_sme_bg_rup( (x->fcnt + ((0 != x->medge) ? 1 : 0)) * x->G * 16, 32 ) * 128;
  }
  return 1;
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_variant_use( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_variant* v ) {
  libxsmm_sme_bg_set_rows( x, v->m );
  x->G = v->G; x->pred = v->pred; x->op_words = v->op_words;
  x->d_load = v->d_load; x->d_mem = v->d_mem; x->max_extra = v->max_extra;
}

/* code size of the fully unrolled k-loops of a variant with background transposition: up to 3 block
 * variants, per k-step 2 loads, 2 adds, 4 FMOPAs, plus the background ops */
LIBXSMM_API_INLINE unsigned long long libxsmm_sme_bg_variant_code( const libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_variant* v ) {
  return 4ULL * (2 + ((0 != v->m % 32) ? 1 : 0)) * (8ULL * x->k + v->op_words + 256);
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_set_problem( libxsmm_sme_bg_ctx* x, const libxsmm_gemm_descriptor* d, unsigned int swap ) {
  x->swap = swap;
  x->k = (unsigned int)d->k;
  x->ldc = (unsigned int)d->ldc;
  x->beta0 = ((d->flags & LIBXSMM_GEMM_FLAG_BETA_0) != 0) ? 1 : 0;
  if ( 0 == swap ) {
    x->m = (unsigned int)d->m; x->n = (unsigned int)d->n;
    x->lda = (unsigned int)d->lda; x->ldb = (unsigned int)d->ldb;
    x->ta = ((d->flags & LIBXSMM_GEMM_FLAG_TRANS_A) != 0) ? 1 : 0;
    x->tb = ((d->flags & LIBXSMM_GEMM_FLAG_TRANS_B) != 0) ? 1 : 0;
  } else {   /* C^T = B A: A' = B (n x k, not transposed), B' = A (k x m, not transposed) */
    x->m = (unsigned int)d->n; x->n = (unsigned int)d->m;
    x->lda = (unsigned int)d->ldb; x->ldb = (unsigned int)d->lda;
    x->ta = 0;
    x->tb = 0;
  }
}

/* dump mode layout of the planned problem: row groups if all of op(A) packed does not fit into the
 * workspace (op(A) of a group of R rows packed, then all panels; op(B) is transposed again for every
 * group), background transposition if it fits; returns 1 if the workspace fits */
LIBXSMM_API_INLINE int libxsmm_sme_bg_dump_layout( libxsmm_sme_bg_ctx* x, libxsmm_sme_bg_variant* var, unsigned int* nfr, unsigned int* rem ) {
  const unsigned int l_m = x->m;
  unsigned int l_ng;
  x->a_packed = x->ta;
  x->b_packed = (0 == x->tb) ? 1 : 0;
  for ( l_ng = 1; ; ++l_ng ) {
    unsigned int l_rows;
    if ( l_ng > 1 && (0 == x->a_packed || 0 != x->swap) ) return 0;
    l_rows = (1 == l_ng) ? l_m : libxsmm_sme_bg_rup( libxsmm_sme_bg_cdiv( l_m, l_ng ), 32 );
    if ( l_rows < 64 ) return 0;
    *nfr = l_m / l_rows;
    *rem = l_m % l_rows;
    /* all groups with background transposition, if that fits */
    x->bgt = (0 != libxsmm_sme_bg_variant_plan( x, l_rows, 1, &var[0] ) &&
              (0 == *rem || 0 != libxsmm_sme_bg_variant_plan( x, *rem, 1, &var[1] )) &&
              libxsmm_sme_bg_variant_code( x, &var[0] ) + ((0 != *rem) ? libxsmm_sme_bg_variant_code( x, &var[1] ) : 0) + 32768
                <= LIBXSMM_SME_BG_MAX_CODE) ? 1 : 0;
    if ( 0 == x->bgt ) {   /* else op(B) packed up front */
      if ( 0 != x->swap ) return 0;
      if ( 0 == libxsmm_sme_bg_variant_plan( x, l_rows, 0, &var[0] ) ||
           (0 != *rem && 0 == libxsmm_sme_bg_variant_plan( x, *rem, 0, &var[1] )) ) continue;
    }
    x->mpad = libxsmm_sme_bg_rup( l_rows, 64 );
    x->pa_bytes = (0 != x->a_packed) ? x->kpad * x->mpad * 4 : 0;
    if ( 0 != x->bgt ) {
      const unsigned int l_panel = LIBXSMM_MAX( var[0].panel, (0 != *rem) ? var[1].panel : 0 );
      x->pb_bytes = 0;
      x->pb0 = x->pa_bytes;
      x->pb1 = x->pa_bytes + l_panel;
      x->pbe = x->pa_bytes + 2 * l_panel;
      x->dump_off = x->pbe + ((0 != x->nedge) ? x->kpad * 128 : 0);
    } else {
      x->pb_bytes = (0 != x->b_packed) ? x->kpad * x->npad * 4 : 0;
      x->dump_off = x->pa_bytes + x->pb_bytes;
    }
    x->dummy_off = x->dump_off + 4096;
    x->scr_off = x->dummy_off + 4096;
    x->ws_bytes = x->scr_off + 2048;
    if ( x->ws_bytes <= LIBXSMM_SME_BG_MAX_WS ) return 1;
  }
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_dump_generate( libxsmm_sme_bg_ctx* x, const libxsmm_sme_bg_variant* var, unsigned int nfr, unsigned int rem ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_abase, l_g;
  libxsmm_reset_loop_label_tracker( &x->lt );
  libxsmm_sme_bg_variant_use( x, &var[0] );
  libxsmm_sme_bg_prologue( x );
  l_abase = (0 != x->a_packed) ? LIBXSMM_SME_BG_X(27) : LIBXSMM_SME_BG_X(0);
  x->a_stride_reg = LIBXSMM_SME_BG_X(6);
  if ( 0 == x->bgt ) {
    if ( 0 != x->b_packed ) {   /* op(B)[k, j] = B[k + j*ldb]: columns j of B are contiguous over k */
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(22), LIBXSMM_SME_BG_X(27), x->pa_bytes );
      libxsmm_sme_bg_pack( x, LIBXSMM_SME_BG_X(1), x->ldb, x->k, x->n, LIBXSMM_SME_BG_X(22), x->npad * 4 );
      x->b_stride_reg = LIBXSMM_AARCH64_GP_REG_UNDEF;
      x->b_stride_imm = x->npad * 4;
      if ( x->b_stride_imm >= 4096 ) {
        libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(9), x->b_stride_imm );
        x->b_stride_reg = LIBXSMM_SME_BG_X(9);
      }
    } else {
      x->b_stride_reg = LIBXSMM_SME_BG_X(4);
    }
  }
  for ( l_g = 0; l_g < 2; ++l_g ) {   /* full row groups (x3 counts them), then the rest */
    const unsigned int l_cnt = (0 == l_g) ? nfr : ((0 != rem) ? 1 : 0);
    if ( 0 == l_cnt ) continue;
    libxsmm_sme_bg_variant_use( x, &var[l_g] );
    if ( l_cnt > 1 ) {
      libxsmm_generator_loop_header_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(3), l_cnt );
    }
    if ( 0 != x->a_packed ) {   /* op(A)[i, k] = A[k + i*lda]: columns i of A are contiguous over k */
      libxsmm_sme_bg_pack( x, LIBXSMM_SME_BG_X(0), x->lda, x->k, x->m, LIBXSMM_SME_BG_X(27), x->mpad * 4 );
    }
    if ( 0 != x->bgt ) {
      libxsmm_sme_bg_bgt_all( x, l_abase );
    } else {
      if ( 0 != x->b_packed ) {
        libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(22), LIBXSMM_SME_BG_X(27), x->pa_bytes );
      } else {
        libxsmm_sme_bg_mov( c, LIBXSMM_SME_BG_X(22), LIBXSMM_SME_BG_X(1) );
      }
      libxsmm_sme_bg_bg_all( x, l_abase );
    }
    if ( 0 == l_g && (l_cnt > 1 || 0 != rem) ) {   /* next row group */
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(0), LIBXSMM_SME_BG_X(0), 4LL * x->m * x->lda );
      libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(2), LIBXSMM_SME_BG_X(2), 4LL * x->m );
    }
    if ( l_cnt > 1 ) {
      libxsmm_generator_loop_footer_aarch64( c, &x->lt, LIBXSMM_SME_BG_X(3), 1 );
    }
  }
  libxsmm_sme_bg_epilogue( x );
}

/* ZA mode layout (op(A) / op(B) packed up front if transposed); returns 1 if the workspace fits */
LIBXSMM_API_INLINE int libxsmm_sme_bg_za_layout( libxsmm_sme_bg_ctx* x ) {
  libxsmm_sme_bg_strip l_strips[64];
  unsigned int l_nblocks;
  if ( 0 == libxsmm_sme_bg_plan_strips( x->m, x->n, l_strips, &l_nblocks ) ) return 0;
  x->mpad = libxsmm_sme_bg_rup( x->m, 64 );
  x->npad = libxsmm_sme_bg_rup( x->n, 64 );
  x->kpad = libxsmm_sme_bg_rup( x->k, 32 );
  x->a_packed = x->ta;
  x->b_packed = (0 == x->tb) ? 1 : 0;
  x->pa_bytes = (0 != x->a_packed) ? x->kpad * x->mpad * 4 : 0;
  x->pb_bytes = (0 != x->b_packed) ? x->kpad * x->npad * 4 : 0;
  x->bgt = 0;
  x->ws_bytes = x->pa_bytes + x->pb_bytes;
  return (x->ws_bytes <= LIBXSMM_SME_BG_MAX_WS) ? 1 : 0;
}

LIBXSMM_API_INLINE void libxsmm_sme_bg_za_generate( libxsmm_sme_bg_ctx* x ) {
  libxsmm_generated_code* c = x->code;
  unsigned int l_abase = LIBXSMM_SME_BG_X(0), l_bbase = LIBXSMM_SME_BG_X(1);
  libxsmm_reset_loop_label_tracker( &x->lt );
  libxsmm_sme_bg_prologue( x );
  /* w8 = 0, w9 = 8: vector select of the vgx4 groups of ZA */
  libxsmm_aarch64_instruction_alu_move_imm16( c, LIBXSMM_AARCH64_INSTR_GP_MOVZ, LIBXSMM_AARCH64_GP_REG_W8, 0, 0 );
  libxsmm_aarch64_instruction_alu_move_imm16( c, LIBXSMM_AARCH64_INSTR_GP_MOVZ, LIBXSMM_AARCH64_GP_REG_W9, 0, 8 );
  x->a_stride_reg = LIBXSMM_SME_BG_X(6);
  x->b_stride_reg = LIBXSMM_SME_BG_X(4);
  if ( 0 != x->a_packed ) {   /* op(A)[i, k] = A[k + i*lda]: columns i of A are contiguous over k */
    libxsmm_sme_bg_pack( x, LIBXSMM_SME_BG_X(0), x->lda, x->k, x->m, LIBXSMM_SME_BG_X(27), x->mpad * 4 );
    l_abase = LIBXSMM_SME_BG_X(27);
  }
  if ( 0 != x->b_packed ) {   /* op(B)[k, j] = B[k + j*ldb]: columns j of B are contiguous over k */
    libxsmm_sme_bg_add_imm( c, LIBXSMM_SME_BG_X(28), LIBXSMM_SME_BG_X(27), x->pa_bytes );
    libxsmm_sme_bg_pack( x, LIBXSMM_SME_BG_X(1), x->ldb, x->k, x->n, LIBXSMM_SME_BG_X(28), x->npad * 4 );
    l_bbase = LIBXSMM_SME_BG_X(28);
    x->b_stride_reg = LIBXSMM_AARCH64_GP_REG_UNDEF;
    x->b_stride_imm = x->npad * 4;
    if ( x->b_stride_imm >= 4096 ) {
      libxsmm_aarch64_instruction_alu_set_imm64( c, LIBXSMM_SME_BG_X(9), x->b_stride_imm );
      x->b_stride_reg = LIBXSMM_SME_BG_X(9);
    }
  }
  libxsmm_sme_bg_za_all( x, l_abase, l_bbase );
  libxsmm_sme_bg_epilogue( x );
}

LIBXSMM_API_INLINE int libxsmm_sme_bg_alloc( libxsmm_sme_bg_ctx* x ) {
  if ( NULL != x->ops ) return 1;
  x->ops_cap = 2048;      /* background ops of one block: <= ~1200 (8 k-groups + C update) */
  x->words_cap = 16384;
  x->deps_cap = 32768;
  x->ops = (libxsmm_sme_bg_op*)malloc( sizeof(libxsmm_sme_bg_op) * x->ops_cap );
  x->words = (unsigned int*)malloc( sizeof(unsigned int) * x->words_cap );
  x->deps = (unsigned int*)malloc( sizeof(unsigned int) * x->deps_cap );
  x->used_n = (unsigned char*)malloc( (size_t)x->k + 1 );
  x->used_v = (unsigned char*)malloc( (size_t)x->k + 1 );
  x->opcode = *x->code;
  x->opcode.last_error = 0;
  return (NULL != x->ops && NULL != x->words && NULL != x->deps && NULL != x->used_n && NULL != x->used_v) ? 1 : 0;
}

LIBXSMM_API_INTERN
int libxsmm_generator_gemm_aarch64_kernel_sme_bg( libxsmm_generated_code*        io_generated_code,
                                                  const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  libxsmm_sme_bg_ctx* x;
  libxsmm_sme_bg_variant l_var[2];
  unsigned int l_nfr = 1, l_rem = 0, l_dump, l_tried = 0;
  int l_result = 0;
  const unsigned int l_flags = (unsigned int)i_xgemm_desc->flags & ~(unsigned int)LIBXSMM_GEMM_FLAG_BETA_0;
  const unsigned int l_trans = LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B;

  /* plain (non-batch-reduce) SGEMM with the XGEMM ABI, alpha = 1, beta = 0/1 */
  if ( io_generated_code->arch != LIBXSMM_AARCH64_APPL_M4 || io_generated_code->code_type <= 1 ||
       LIBXSMM_DATATYPE_F32 != LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ||
       LIBXSMM_DATATYPE_F32 != LIBXSMM_GEMM_GETENUM_C_PREC( i_xgemm_desc->datatype ) ||
       (l_flags & ~l_trans) != LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI ||
       i_xgemm_desc->m <= 0 || i_xgemm_desc->n <= 0 || i_xgemm_desc->k <= 0 ) {
    return 0;
  }
  x = (libxsmm_sme_bg_ctx*)calloc( 1, sizeof(libxsmm_sme_bg_ctx) );
  if ( NULL == x ) return 0;
  x->code = io_generated_code;

  /* TT: computed as C^T = B A with background transposition of A, if that applies */
  if ( (l_flags & l_trans) == l_trans ) {
    libxsmm_sme_bg_set_problem( x, i_xgemm_desc, 1 );
    if ( 0 != libxsmm_sme_bg_plan( x ) && 0 != libxsmm_sme_bg_alloc( x ) &&
         0 != libxsmm_sme_bg_dump_layout( x, l_var, &l_nfr, &l_rem ) ) {
      libxsmm_sme_bg_dump_generate( x, l_var, l_nfr, l_rem );
      l_result = 1;
      goto cleanup;
    }
  }
  /* dump mode if it does not need more blocks than ZA mode: a block costs >= 4 cycles per k-step
   * whatever its tile count, updating C costs ZA mode ~60 cycles per block */
  libxsmm_sme_bg_set_problem( x, i_xgemm_desc, 0 );
  l_dump = (unsigned int)libxsmm_sme_bg_plan( x );
  if ( 0 != l_dump ) {
    const unsigned long long l_nbg = (unsigned long long)libxsmm_sme_bg_cdiv( x->m, 32 ) * libxsmm_sme_bg_cdiv( x->n, 32 );
    libxsmm_sme_bg_strip l_strips[64];
    unsigned int l_nza;
    libxsmm_sme_bg_plan_strips( x->m, x->n, l_strips, &l_nza );
    if ( 0 == l_nza ) l_nza = 0xffffffff;
    if ( l_nbg * (4ULL * x->k + 20) + 400 < (unsigned long long)l_nza * (4ULL * x->k + 60) ) {
      l_tried = 1;
      if ( 0 != libxsmm_sme_bg_alloc( x ) && 0 != libxsmm_sme_bg_dump_layout( x, l_var, &l_nfr, &l_rem ) ) {
        libxsmm_sme_bg_dump_generate( x, l_var, l_nfr, l_rem );
        l_result = 1;
        goto cleanup;
      }
    }
  }
  /* ZA mode */
  libxsmm_sme_bg_set_problem( x, i_xgemm_desc, 0 );
  if ( 0 != libxsmm_sme_bg_za_layout( x ) ) {
    libxsmm_sme_bg_za_generate( x );
    l_result = 1;
    goto cleanup;
  }
  /* dump mode anyway (e.g. the ZA mode workspace does not fit) */
  if ( 0 != l_dump && 0 == l_tried ) {
    libxsmm_sme_bg_plan( x );
    if ( 0 != libxsmm_sme_bg_alloc( x ) && 0 != libxsmm_sme_bg_dump_layout( x, l_var, &l_nfr, &l_rem ) ) {
      libxsmm_sme_bg_dump_generate( x, l_var, l_nfr, l_rem );
      l_result = 1;
    }
  }

cleanup:
  free( x->ops );
  free( x->words );
  free( x->deps );
  free( x->used_n );
  free( x->used_v );
  free( x );
  return l_result;   /* 1: code was emitted (errors are reported in io_generated_code) */
}
