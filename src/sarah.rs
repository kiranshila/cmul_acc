#![allow(dead_code, non_snake_case)]

use super::Mat;
use pulp::{x86::*, *};
use std::arch::x86_64::__m256i;

// rowmajor to colmajor: lhs0, lhs1
// vcat: lhs0, lhs1
// mul2: lhs0, lhs1, rhs0, rhs1
// mul2: lhs0, lhs1, rhs0, rhs1
// ...
// vdog: lhs0, lhs1
// colmajor to rowmajor: lhs1
//
// mul1: lhs0, lhs1
// colmajor to rowmajor: lhs0

// lhs * rhs0 * rhs1 | lhs * rhs0...
#[inline(always)]
pub fn rowmajor_to_colmajor(simd: V4, values: Mat) -> [f64x4; 2] {
    let x: [__m256i; 2] = cast!(values);

    cast!([
        simd.avx2
            ._mm256_permute2x128_si256::<0b00100000>(x[0], x[1]),
        simd.avx2
            ._mm256_permute2x128_si256::<0b00110001>(x[0], x[1]),
    ])
}

#[inline(always)]
pub fn colmajor_to_rowmajor(simd: V4, values: [f64x4; 2]) -> Mat {
    cast!(rowmajor_to_colmajor(simd, cast!(values)))
}

// returns output in column major
#[inline(always)]
pub fn mul1(simd: V4, lhs_colmajor: [f64x4; 2], rhs_rowmajor: &Mat) -> [f64x4; 2] {
    // C = C0 C1
    //
    // C0 = A * B0
    // C1 = A * B1
    //
    // C0 = A0 * B00 + A1 * B10
    // C1 = A0 * B01 + A1 * B11

    let simd = *simd;

    let A = lhs_colmajor;
    let B = rhs_rowmajor;

    let C0 = simd.mul_add_c64s(
        A[1],
        simd.splat_c64s(B[1][0]),
        simd.mul_c64s(A[0], simd.splat_c64s(B[0][0])),
    );

    let C1 = simd.mul_add_c64s(
        A[1],
        simd.splat_c64s(B[1][1]),
        simd.mul_c64s(A[0], simd.splat_c64s(B[0][1])),
    );

    cast!([C0, C1])
}

#[inline(always)]
pub fn vcat(simd: V4, A: [f64x4; 2], B: [f64x4; 2]) -> [f64x8; 2] {
    let _ = simd;
    cast!([A[0], B[0], A[1], B[1]])
}

#[inline(always)]
pub fn vdog(simd: V4, AB: [f64x8; 2]) -> ([f64x4; 2], [f64x4; 2]) {
    let _ = simd;
    let AB: [[f64x4; 2]; 2] = cast!(AB);

    (cast!([AB[0][0], AB[0][1]]), cast!([AB[1][0], AB[1][1]]))
}

#[inline(always)]
pub fn mul2(simd: V4, lhs_colmajor: [f64x8; 2], rhs_rowmajor: [&Mat; 2]) -> [f64x8; 2] {
    // C = C0 C1
    //
    // C0 = A * B0
    // C1 = A * B1
    //
    // C0 = A0 * B00 + A1 * B10
    // C1 = A0 * B01 + A1 * B11

    let A = lhs_colmajor;
    let [B0, B1] = rhs_rowmajor;
    let mask = b8(0b00001111);

    let B = {
        #[inline(always)]
        |i: usize, j: usize| {
            simd.select_f64x8(mask, simd.splat_c64s(B0[i][j]), simd.splat_c64s(B1[i][j]))
        }
    };

    let C0 = simd.mul_add_c64s(A[1], B(1, 0), simd.mul_c64s(A[0], B(0, 0)));
    let C1 = simd.mul_add_c64s(A[1], B(1, 1), simd.mul_c64s(A[0], B(0, 1)));

    cast!([C0, C1])
}

pub fn cmul_acc(mats: &[Mat]) -> Mat {
    let simd = V4::try_new().unwrap();
    let (a, b) = mats.split_at(mats.len() / 2);
    let l0 = rowmajor_to_colmajor(simd, a[0]);
    let l1 = rowmajor_to_colmajor(simd, b[0]);
    let mut vcl = vcat(simd, l0, l1);
    for (r0, r1) in a[1..].iter().zip(&b[1..]) {
        vcl = mul2(simd, vcl, [r0, r1]);
    }
    let (x, y) = vdog(simd, vcl);
    let z = mul1(simd, x, &colmajor_to_rowmajor(simd, y));
    colmajor_to_rowmajor(simd, z)
}
