use revolut::*;
use tfhe::shortint::parameters::*;

use crate::{Query, GLWE};
pub struct Client {
    pub private_key: PrivateKey,
    pub public_key: PublicKey,
}

impl Client {
    pub fn new(parameters: &ClassicPBSParameters) -> Client {
        let private_key_ref = key(*parameters);
        let private_key = private_key_ref.clone();
        let public_key = private_key.public_key.clone();

        Client {
            private_key: private_key,
            public_key: public_key,
        }
    }

    pub fn create_query(&self, feature_vector: Vec<u64>, ctx: &mut Context) -> Query {
        let ct = self.encrypt_first_in_glwe(&feature_vector, &self.private_key, ctx);
        let ct_second = self.encrypt_second_in_glwe(&feature_vector, &self.private_key, ctx);
        Query { ct, ct_second }
    }

    fn calculate_second(&self, client_feature_vector: &Vec<u64>) -> u64 {
        client_feature_vector.iter().map(|&x| x.pow(2)).sum()
    }

    fn encrypt_first_in_glwe(
        &self,
        client_feature_vector: &Vec<u64>,
        private_key: &PrivateKey,
        ctx: &mut Context,
    ) -> GLWE {
        private_key.allocate_and_encrypt_glwe_from_vec(client_feature_vector, ctx)
    }

    fn encrypt_second_in_glwe(
        &self,
        client_feature_vector: &Vec<u64>,
        private_key: &PrivateKey,
        ctx: &mut Context,
    ) -> GLWE {
        let dim = client_feature_vector.len();
        let second_value = self.calculate_second(client_feature_vector);

        // Create a polynomial of size ctx.polynomial_size() filled with zeros
        let mut poly_coeffs: Vec<u64> = vec![0; ctx.polynomial_size().0];

        // Set the value at slot dim - 1
        poly_coeffs[dim - 1] = second_value;

        // Encrypt the polynomial in GLWE
        let result = private_key.allocate_and_encrypt_glwe_from_vec(&poly_coeffs, ctx);
        result
    }
}
