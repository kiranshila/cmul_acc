#![feature(portable_simd)]
//#![feature(autodiff)]

use pulp::c64;

pub type Mat = [[c64; 2]; 2];

pub mod kiran;
pub mod sarah;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agree() {
        let mut mats: Vec<Mat> = vec![];
        for _ in 1..100 {
            mats.push([
                [rand::random(), rand::random()],
                [rand::random(), rand::random()],
            ]);
        }
        let _ref = kiran::simple_cmul_acc(&mats);
        //assert_eq!(_ref, kiran::simd_cmul_acc(&mats));
        assert_eq!(_ref, sarah::cmul_acc(&mats));
    }
}
