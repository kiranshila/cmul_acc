use crate::Mat;
use pulp::bytemuck;
use std::simd::prelude::*;

#[inline(always)]
pub fn simd_cmul(a: &Mat, b: &Mat) -> Mat {
    let a: f64x8 = Simd::from_slice(bytemuck::cast_ref::<_, [f64; 8]>(a));
    let b: f64x8 = Simd::from_slice(bytemuck::cast_ref::<_, [f64; 8]>(b));
    let col1a = simd_swizzle!(a, [1, 1, 1, 1, 5, 5, 5, 5]);
    let col1b = simd_swizzle!(b, [1, 0, 3, 2, 1, 0, 3, 2]);
    let col2a = simd_swizzle!(a, [0, 0, 0, 0, 4, 4, 4, 4]);
    let col2b = simd_swizzle!(b, [0, 1, 2, 3, 0, 1, 2, 3]);
    let col3a = simd_swizzle!(a, [3, 3, 3, 3, 7, 7, 7, 7]);
    let col3b = simd_swizzle!(b, [5, 4, 7, 6, 5, 4, 7, 6]);
    let col4a = simd_swizzle!(a, [2, 2, 2, 2, 6, 6, 6, 6]);
    let col4b = simd_swizzle!(b, [4, 5, 6, 7, 4, 5, 6, 7]);
    let checker = Simd::from_array([-1., 1., -1., 1., -1., 1., -1., 1.]);
    let col1 = col1a * col1b;
    let col2 = col2a * col2b;
    let col3 = col3a * col3b;
    let col4 = col4a * col4b;
    let res = checker * col1 + col2 + checker * col3 + col4;
    bytemuck::cast(res.to_array())
}

pub fn simd_cmul_acc(mats: &[Mat]) -> Mat {
    let mut acc = mats[0].clone();
    for mat in mats[1..].iter() {
        acc = simd_cmul(&acc, &mat);
    }
    acc
}

pub fn simple_cmul_acc(mats: &[Mat]) -> Mat {
    let mut acc = mats[0].clone();
    for mat in mats[1..].iter() {
        acc = [
            [
                acc[0][0] * mat[0][0] + acc[0][1] * mat[1][0],
                acc[0][0] * mat[0][1] + acc[0][1] * mat[1][1],
            ],
            [
                acc[1][0] * mat[0][0] + acc[1][1] * mat[1][0],
                acc[1][0] * mat[0][1] + acc[1][1] * mat[1][1],
            ],
        ];
    }
    acc
}

// #[autodiff(dsimple_cmul_autodiff, Reverse, Duplicated, Duplicated, Active)]
// pub fn simple_cmul_autodiff(mats: &mut [Mat], angles: &[f64]) -> f64 {
//     // Do some math
//     for (mat, angle) in mats.iter_mut().zip(angles) {
//         mat[0][0] = c64::cis(*angle);
//         mat[0][1] = c64::cis(*angle + PI / 2.0);
//         mat[1][0] = c64::cis(*angle - PI / 2.0);
//         mat[1][1] = c64::cis(*angle + PI);
//     }
//     // Cascade
//     let casc = simple_cmul_acc(mats);
//     // Do some more math
//     let x: c64 = casc[0].iter().sum();
//     let y: c64 = casc[1].iter().sum();
//     let z = x + y;
//     z.im * z.im + z.re * z.re
// }

// #[autodiff(dsimd_cmul_autodiff, Reverse, Duplicated, Duplicated, Active)]
// pub fn simd_cmul_autodiff(mats: &mut [Mat], angles: &[f64]) -> f64 {
//     // Do some math
//     for (mat, angle) in mats.iter_mut().zip(angles) {
//         mat[0][0] = c64::cis(*angle);
//         mat[0][1] = c64::cis(*angle + PI / 2.0);
//         mat[1][0] = c64::cis(*angle - PI / 2.0);
//         mat[1][1] = c64::cis(*angle + PI);
//     }
//     // Cascade
//     let casc = simd_cmul_acc(mats);
//     // Do some more math
//     let x: c64 = casc[0].iter().sum();
//     let y: c64 = casc[1].iter().sum();
//     let z = x + y;
//     z.im * z.im + z.re * z.re
// }
