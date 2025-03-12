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

#[cfg(test)]
mod tests {
    use crate::{cache::read_keys_from_file, decrypt_array, encrypt_array};
    use itertools::Itertools;
    use tfhe::set_server_key;

    use super::*;

    #[test]
    fn test_blind_sort_simple() {
        let (client_key, server_key) = read_keys_from_file();
        set_server_key(server_key);
        let data = Vec::from_iter((0..16).rev());
        let encrypted = encrypt_array(&data, &client_key);

        let sorted = simple_sort(&encrypted);

        let decrypted = decrypt_array(&sorted, &client_key);
        println!("{:?}", decrypted);
        let mut expected = data;
        expected.sort();
        assert_eq!(decrypted, expected);
    }

    #[test]
    fn test_sorting_permutation() {
        let (client_key, server_key) = read_keys_from_file();
        set_server_key(server_key);
        let data = [5, 7, 3, 2];
        let encrypted = encrypt_array(&data, &client_key);

        let permutation = sorting_permutation(&encrypted);

        let decrypted = decrypt_array(&permutation, &client_key);
        assert_eq!(decrypted, [2, 3, 1, 0]);
    }

    #[test]
    fn test_blind_sort_ds() {
        let (client_key, server_key) = read_keys_from_file();
        set_server_key(server_key);
        // let data = [1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let data = Vec::from_iter(0..32);
        let encrypted = encrypt_array(&data, &client_key);

        let sorted = direct_sort(&encrypted);

        let decrypted = decrypt_array(&sorted, &client_key);
        let mut data = data;
        data.sort();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_blind_sort_2bp() {
        let (client_key, server_key) = read_keys_from_file();
        set_server_key(server_key);
        // let data = [1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let data = Vec::from_iter(0..32);
        let encrypted = encrypt_array(&data, &client_key);

        let sorted = double_blind_permutation(&encrypted);

        let decrypted = decrypt_array(&sorted, &client_key);
        let mut data = data;
        data.sort();
        data.rotate_left(1);
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_bitonic_sort() {
        let (client_key, server_key) = read_keys_from_file();
        set_server_key(server_key);
        for data in (0..4).permutations(4) {
            println!("data: {:?}", data);
            let encrypted = encrypt_array(&data, &client_key);

            let sorted = bitonic_sort(&encrypted);

            let decrypted = decrypt_array(&sorted, &client_key);
            println!("decrypted: {:?}", decrypted);
            let mut expected = data;
            expected.sort();
            assert_eq!(decrypted, expected);
        }
    }
}
