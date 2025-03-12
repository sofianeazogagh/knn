use std::{
    fs::{create_dir_all, File},
    io::{Read, Write},
};
use tfhe::{generate_keys, ClientKey, ConfigBuilder, ServerKey};

fn get_file_paths() -> (String, String) {
    // Create a tmp dir with the current user name to avoid cluttering the /tmp dir
    let user = std::env::var("USER").unwrap_or_else(|_| "unknown_user".to_string());
    let tmp_dir_for_user = &format!("/tmp/{user}");

    create_dir_all(tmp_dir_for_user).unwrap();

    let server_key_file = format!("{tmp_dir_for_user}/server_key.bin");
    let client_key_file = format!("{tmp_dir_for_user}/client_key.bin");

    (server_key_file, client_key_file)
}

pub fn write_keys_to_file() {
    let (server_key_file, client_key_file) = get_file_paths();

    // We generate a set of client/server keys, using the default parameters:
    let config = ConfigBuilder::default().build();
    let (client_key, server_key) = generate_keys(config);

    // We serialize the keys to bytes:
    let encoded_server_key: Vec<u8> = bincode::serialize(&server_key).unwrap();
    let encoded_client_key: Vec<u8> = bincode::serialize(&client_key).unwrap();

    // We write the keys to files:
    let mut file = File::create(server_key_file).expect("failed to create server key file");
    file.write_all(encoded_server_key.as_slice())
        .expect("failed to write key to file");
    let mut file = File::create(client_key_file).expect("failed to create client key file");
    file.write_all(encoded_client_key.as_slice())
        .expect("failed to write key to file");
}

pub fn read_keys_from_file() -> (ClientKey, ServerKey) {
    let (server_key_file, client_key_file) = get_file_paths();

    // We retrieve the keys:
    let mut file = match File::open(&server_key_file) {
        Ok(f) => f,
        Err(_) => {
            write_keys_to_file();
            File::open(server_key_file).unwrap()
        }
    };
    let mut encoded_server_key: Vec<u8> = Vec::new();
    file.read_to_end(&mut encoded_server_key)
        .expect("failed to read the key");

    let mut file = File::open(client_key_file).expect("failed to open client key file");
    let mut encoded_client_key: Vec<u8> = Vec::new();
    file.read_to_end(&mut encoded_client_key)
        .expect("failed to read the key");

    // We deserialize the keys:
    let loaded_server_key: ServerKey =
        bincode::deserialize(&encoded_server_key[..]).expect("failed to deserialize");
    let loaded_client_key: ClientKey =
        bincode::deserialize(&encoded_client_key[..]).expect("failed to deserialize");
    (loaded_client_key, loaded_server_key)
}
