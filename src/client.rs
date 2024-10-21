use revolut::*;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::*;

use crate::Query;
pub struct Client<'a> {
    private_key: &'a PrivateKey,
    public_key: &'a PublicKey,
    ctx: Context,
}

impl<'a> Client<'a> {
    // pub fn new() -> Self {
    //     Client {
    //         private_key,
    //         public_key: private_key.public_key,
    //         ctx,
    //     }
    // }

    pub fn from_parameters(parameters: ClassicPBSParameters) -> Self {
        let private_key = key(parameters);
        let ctx = Context::from(parameters);
        Client {
            private_key: &private_key,
            public_key: &private_key.public_key,
            ctx,
        }
    }

    pub fn create_query(&mut self, feature_vector: Vec<u64>) -> Query {
        let ct = Self::encrypt_first_in_glwe(&feature_vector, &self.private_key, &mut self.ctx);
        let ct_second =
            Self::encrypt_second_in_glwe(&feature_vector, &self.private_key, &mut self.ctx);
        Query { ct, ct_second }
    }

    fn calculate_second(client_feature_vector: &Vec<u64>) -> u64 {
        client_feature_vector.iter().map(|&x| x.pow(2)).sum()
    }

    fn encrypt_first_in_glwe(
        client_feature_vector: &Vec<u64>,
        private_key: &PrivateKey,
        ctx: &mut Context,
    ) -> GlweCiphertext<Vec<u64>> {
        private_key.allocate_and_encrypt_glwe_from_vec(client_feature_vector, ctx)
    }

    fn encrypt_second_in_glwe(
        client_feature_vector: &Vec<u64>,
        private_key: &PrivateKey,
        ctx: &mut Context,
    ) -> GlweCiphertext<Vec<u64>> {
        let dim = client_feature_vector.len();
        let second_value = Self::calculate_second(client_feature_vector);

        // Create a polynomial of size ctx.polynomial_size() filled with zeros
        let mut poly_coeffs: Vec<u64> = vec![0; ctx.polynomial_size().0];

        // Set the value at slot dim - 1
        poly_coeffs[dim - 1] = second_value;

        // Encrypt the polynomial in GLWE
        let result = private_key.allocate_and_encrypt_glwe_from_vec(&poly_coeffs, ctx);
        result
    }
}
