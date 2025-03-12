pub mod blindsort;
pub mod cache;

use std::{
    io::{self, Write},
    time::Instant,
};

use cache::read_keys_from_file;
use revolut::{random_lut, Context, PublicKey, LUT};

#[allow(unused_imports)]
use tfhe::{
    prelude::{FheDecrypt, FheEncrypt},
    set_server_key,
    shortint::parameters::*,
    ClientKey, FheUint16, FheUint32, FheUint8,
};

pub type Plain = u32;
pub type Cipher = FheUint32;

pub fn bench_revolut<F: Fn(&PublicKey, LUT, &Context) -> T, T>(name: &str, f: F) {
    bench_revolut_upto(name, f, 7usize)
}

pub fn bench_revolut_upto<F: Fn(&PublicKey, LUT, &Context) -> T, T>(name: &str, f: F, upto: usize) {
    let params = [
        PARAM_MESSAGE_2_CARRY_0,
        PARAM_MESSAGE_3_CARRY_0,
        PARAM_MESSAGE_4_CARRY_0,
        PARAM_MESSAGE_5_CARRY_0,
        PARAM_MESSAGE_6_CARRY_0,
        PARAM_MESSAGE_7_CARRY_0,
    ];

    print!("{}", name);
    for i in 0..params.len() {
        if i < upto {
            let private_key = revolut::key(params[i]);
            let public_key = &private_key.public_key;
            let lut = random_lut(params[i]);
            let begin = Instant::now();
            f(public_key, lut, &Context::from(params[i]));
            let elapsed = Instant::now() - begin;
            print!("\t{:?}", elapsed);
        } else {
            print!("\t-");
        }
        let _ = io::stdout().flush();
    }
    println!();
}

pub fn bench<F: Fn(&[Cipher]) -> T, T>(name: &str, f: F) {
    let (client_key, server_key) = read_keys_from_file();
    set_server_key(server_key);
    let data = encrypt_array(&Vec::from_iter(0..=0xff), &client_key);

    print!("{}", name);
    let _ = io::stdout().flush();
    for n in (2..=8).map(|i| 2u64.pow(i) as usize) {
        let begin = Instant::now();
        f(&data[..n]);
        let elapsed = Instant::now() - begin;
        print!("\t{:?}", elapsed);
        let _ = io::stdout().flush();
    }
    println!();
}

pub fn encrypt_array(data: &[Plain], client_key: &ClientKey) -> Vec<Cipher> {
    data.iter()
        .map(|&c| Cipher::encrypt(c, client_key))
        .collect()
}

pub fn decrypt_array(data: &[Cipher], client_key: &ClientKey) -> Vec<Plain> {
    data.iter().map(|c| c.decrypt(&client_key)).collect()
}
