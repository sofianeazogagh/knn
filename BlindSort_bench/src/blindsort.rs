use tfhe::{
    core_crypto::commons::traits::CastFrom,
    prelude::{FheEq, FheMax, FheMin, FheOrd, FheTrivialEncrypt},
};

use crate::{Cipher, Plain};

/// computes the sorting permutation of a given array of ciphertexts
fn sorting_permutation(data: &[Cipher]) -> Vec<Cipher> {
    let mut out = vec![Cipher::encrypt_trivial(Plain::from(0 as Plain)); data.len()];
    for i in 0..data.len() {
        for j in 0..i {
            let z = data[i].gt(&data[j]);
            let nz = !&z;
            out[i] += Cipher::cast_from(z);
            out[j] += Cipher::cast_from(nz);
        }
    }
    out
}

/// re-order given array of ciphertexts based on ciphered indices
fn apply_permutation(data: &[Cipher], permutation: &[Cipher]) -> Vec<Cipher> {
    // permutation values outside the 0..data.len() range will be ignored
    assert_eq!(data.len(), permutation.len());
    (0..data.len())
        .map(|i| {
            (0..data.len())
                .map(|j| {
                    let jtoi = permutation[j].eq(i as Plain);
                    Cipher::cast_from(jtoi) * &data[j]
                })
                .sum::<Cipher>()
        })
        .collect()
}

/// direct sorting algorithm
/// from Cetin, 2015. Depth Optimized Efficient Homomorphic Sorting
pub fn direct_sort(data: &[Cipher]) -> Vec<Cipher> {
    let permutation = sorting_permutation(&data);
    apply_permutation(&data, &permutation)
}

/// 2bp sorting algorithm
/// using TFHE high level API instead of RevoLUT
pub fn double_blind_permutation(data: &[Cipher]) -> Vec<Cipher> {
    let partially = apply_permutation(&data, &data);
    let mut cnt = Cipher::encrypt_trivial(Plain::from(0 as Plain));
    let permutation = Vec::from_iter(partially.iter().map(|x| {
        let z = x.eq(0);
        cnt += Cipher::cast_from(z);
        x - &cnt
    }));
    apply_permutation(&partially, &permutation)
}

/// simplest sorting algorithm
pub fn simple_sort(data: &[Cipher]) -> Vec<Cipher> {
    let mut result = data.to_vec();
    for i in 0..result.len() {
        for j in 0..i {
            let min: Cipher = result[i].min(&result[j]);
            let max: Cipher = result[i].max(&result[j]);
            result[i] = max;
            result[j] = min;
        }
    }
    result
}

/// bitonic sort
/// a data-oblivious sorting network
pub fn bitonic_sort(data: &[Cipher]) -> Vec<Cipher> {
    let n = data.len(); // must be a power of 2
    let mut result = data.to_vec();
    let mut k = 2;
    while k <= n {
        let mut j = k >> 1;
        while j > 0 {
            for i in 0..n {
                let l = i ^ j;
                if l > i {
                    let min: Cipher = result[i].min(&result[l]);
                    let max: Cipher = result[i].max(&result[l]);
                    let (a, b) = if i & k == 0 { (i, l) } else { (l, i) };
                    result[a] = min;
                    result[b] = max;
                }
            }
            j >>= 1;
        }
        k <<= 1;
    }
    result
}
