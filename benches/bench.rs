use cmul_acc::{Mat, kiran, sarah};

fn main() {
    divan::main();
}

#[divan::bench(sample_size = 1000)]
fn sarah_simd_cmul_acc(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let mut mats: Vec<Mat> = vec![];
            for _ in 1..100 {
                mats.push([
                    [rand::random(), rand::random()],
                    [rand::random(), rand::random()],
                ]);
            }
            mats
        })
        .bench_values(|mats| divan::black_box(sarah::cmul_acc(&mats)));
}

#[divan::bench(sample_size = 1000)]
fn kiran_simd_cmul_acc(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| {
            let mut mats: Vec<Mat> = vec![];
            for _ in 1..100 {
                mats.push([
                    [rand::random(), rand::random()],
                    [rand::random(), rand::random()],
                ]);
            }
            mats
        })
        .bench_values(|mats| divan::black_box(kiran::simd_cmul_acc(&mats)));
}

// #[divan::bench]
// fn simple_cmul_acc(bencher: divan::Bencher) {
//     bencher
//         .with_inputs(|| {
//             let mut mats: Vec<Mat> = vec![];
//             for _ in 1..100 {
//                 mats.push([
//                     [rand::random(), rand::random()],
//                     [rand::random(), rand::random()],
//                 ]);
//             }
//             mats
//         })
//         .bench_values(|mats| divan::black_box(kiran::simple_cmul_acc(&mats)));
// }

// #[divan::bench]
// fn simple_math(bencher: divan::Bencher) {
//     bencher
//         .with_inputs(|| {
//             let mut angles = vec![];
//             let mut mats = vec![];
//             for _ in 1..100 {
//                 angles.push(rand::random());
//                 mats.push([[c64::ZERO, c64::ZERO], [c64::ZERO, c64::ZERO]]);
//             }
//             (mats, angles)
//         })
//         .bench_values(|(mut mats, angles)| {
//             divan::black_box(kiran::simple_cmul_autodiff(&mut mats, &angles))
//         });
// }

// #[divan::bench]
// fn dsimple_math(bencher: divan::Bencher) {
//     bencher
//         .with_inputs(|| {
//             let mut angles = vec![];
//             let mut mats = vec![];
//             for _ in 1..100 {
//                 angles.push(rand::random());
//                 mats.push([[c64::ZERO, c64::ZERO], [c64::ZERO, c64::ZERO]]);
//             }
//             let dangles = angles.clone();
//             let mats_shadow = mats.clone();
//             (mats, angles, mats_shadow, dangles)
//         })
//         .bench_values(|(mut mats, angles, mut mats_shadow, mut dangles)| {
//             divan::black_box(kiran::dsimple_cmul_autodiff(
//                 &mut mats,
//                 &mut mats_shadow,
//                 &angles,
//                 &mut dangles,
//                 1.0,
//             ))
//         });
// }

// #[divan::bench]
// fn dsimd_math(bencher: divan::Bencher) {
//     bencher
//         .with_inputs(|| {
//             let mut angles = vec![];
//             let mut mats = vec![];
//             for _ in 1..100 {
//                 angles.push(rand::random());
//                 mats.push([[c64::ZERO, c64::ZERO], [c64::ZERO, c64::ZERO]]);
//             }
//             let dangles = angles.clone();
//             let mats_shadow = mats.clone();
//             (mats, angles, mats_shadow, dangles)
//         })
//         .bench_values(|(mut mats, angles, mut mats_shadow, mut dangles)| {
//             divan::black_box(kiran::dsimd_cmul_autodiff(
//                 &mut mats,
//                 &mut mats_shadow,
//                 &angles,
//                 &mut dangles,
//                 1.0,
//             ))
//         });
// }
