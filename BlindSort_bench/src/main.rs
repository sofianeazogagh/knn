#[allow(unused_imports)]
use blindsort::{
    bench, bench_revolut, bench_revolut_upto,
    blindsort::{bitonic_sort, direct_sort, double_blind_permutation, simple_sort},
};
#[allow(unused_imports)]
use revolut::{PublicKey, LUT};
use tfhe::shortint::parameters::*;

fn main() {
    // Generate keys for all parameters
    let params = [
        PARAM_MESSAGE_2_CARRY_0,
        PARAM_MESSAGE_3_CARRY_0,
        PARAM_MESSAGE_4_CARRY_0,
        PARAM_MESSAGE_5_CARRY_0,
        PARAM_MESSAGE_6_CARRY_0,
        PARAM_MESSAGE_7_CARRY_0,
    ];
    for param in params {
        let _ = revolut::key(param);
    }

    println!("bench revolut sorts");
    println!("name\t4 values\t8 values\t16 values\t32 values\t64 values\t128 values");
    bench_revolut(
        "bcs",
        |public_key: &PublicKey, lut: LUT, ctx: &revolut::Context| {
            PublicKey::blind_counting_sort(public_key, &lut, ctx)
        },
    );

    println!("bench tfhe sorts");
    println!("name\t4 values\t8 values\t16 values\t32 values\t64 values\t128 values\t256 values");
    bench("bitonic", bitonic_sort);
    bench("simple", simple_sort);
    bench("direct", direct_sort);
    bench("2bp", double_blind_permutation);
}
