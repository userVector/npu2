//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, const int N>
void eltwise_mul(T_in *a, T_in *b, T_out *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
  }
}

template <typename T_in, typename T_out, const int N>
void eltwise_vmul(T_in *a, T_in *b, T_out *c) {

  constexpr int vec_factor = 32;
  event0();
  T_in *__restrict pA1 = a;
  T_in *__restrict pB1 = b;
  T_out *__restrict pC1 = c;
  const int F = N / vec_factor;
  aie::vector<T_in, vec_factor> zero = aie::broadcast<T_in, vec_factor>(0);
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(32, ) {
      aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
      pB1 += vec_factor;
      aie::vector<T_out, vec_factor> cout = aie::mul(A0, B0);
      aie::vector<T_out, vec_factor> relu = aie::max(cout, zero);
      aie::store_v(pC1, relu);
      pC1 += vec_factor;
    }
  event1();
}

template <typename T_in, typename T_out, const int N>
void eltwise_vmul_relu(T_in* __restrict a, T_in* __restrict b, T_out* __restrict c) {
  constexpr int vec_factor = 32;
  event0();
  T_in* __restrict pA = a;
  T_in* __restrict pB = b;
  T_out* __restrict pC = c;
  const int F = N / vec_factor;
  aie::vector<T_in, vec_factor> zero = aie::broadcast<T_in, vec_factor>(0);
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(32, ) {
      aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA);
      pA += vec_factor;
      aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB);
      pB += vec_factor;
      aie::vector<T_in, vec_factor> mult = aie::mul(A0, B0);
      aie::vector<T_in, vec_factor> relu = aie::max(mult, zero);
      aie::store_v(pC, relu);
      pC += vec_factor;
    }
  event1();
}

template <typename T_in, typename T_out, const int N>
void outer_product_sum_relu(T_in* __restrict a, T_in* __restrict b, T_out* __restrict c) {
  constexpr int vec_factor = 32;
  event0();
  T_in *__restrict pA = a;
  T_in *__restrict pB = b;
  T_out *__restrict pC = c;
  const int F = N / vec_factor;
  for (int j = 0; j < F; j++) { //here should be M
    T_out sum = 0;
    T_in b_val = pB[j];
    aie::vector<T_in, vec_factor> vec_b = aie::broadcast<T_in, vec_factor>(b_val);
    for (int i = 0; i < F; i += vec_factor) {
      aie::vector<T_in, vec_factor> vec_a = aie::load_v<vec_factor>(&pA[i]);
      aie::vector<T_in, vec_factor> prod = aie::mul(vec_a, vec_b);
      aie::vector<T_in, vec_factor> relu = aie::max(prod, aie::broadcast<T_in, vec_factor>(0));
      T_in partial_sum = 0;
      for (int k = 0; k < vec_factor; k++) {
        partial_sum += relu[k];
      }
      sum += partial_sum;
    }
    pC[j] = sum;
  }
  event1();
}

extern "C" {

void eltwise_mul_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_mul<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_mul_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_vmul<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void relu_mul_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_vmul_relu<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void dense_relu_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  outer_product_sum_relu<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

} // extern "C"
