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
#ifndef GENERATOR_GEMM_SME_BG_H
#define GENERATOR_GEMM_SME_BG_H

#include "generator_common.h"

IBXSMM_API_INTERN
int libxsmm_generator_gemm_aarch64_kernel_sme_bg( libxsmm_generated_code*        io_generated_code,
                                                  const libxsmm_gemm_descriptor* i_xgemm_desc );

#endif /* GENERATOR_GEMM_SME_BG_H */
