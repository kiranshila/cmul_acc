#![feature(portable_simd)]
//#![feature(autodiff)]

use pulp::c64;

pub type Mat = [[c64; 2]; 2];

pub mod kiran;
pub mod sarah;
