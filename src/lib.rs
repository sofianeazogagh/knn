use tfhe::core_crypto::prelude::*;
use revolut::*;

// Type aliases
pub type GLWE = GlweCiphertext<Vec<u64>>;
pub type LWE = LweCiphertext<Vec<u64>>;
pub type Poly = Polynomial<Vec<u64>>;

// Constants
pub const THREADS: usize = 4;

// Structs
pub struct Query {
    pub ct: GLWE,
    pub ct_second: LWE,
}

// Enums
#[allow(dead_code)]
pub enum QuantizeType {
    None,
    Binary,
    Ternary,
}

// Modules
pub mod client;
pub mod model;
pub mod server;
