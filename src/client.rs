use revolut::*;
use tfhe::shortint::parameters::*;

use crate::{Query, GLWE, LWE};
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

    pub fn create_query(
        &self,
        feature_vector: Vec<u64>,
        ctx: &mut Context,
        delta_dist: u64,
    ) -> Query {
        let ct = self.encrypt_first_in_glwe(&feature_vector, &self.private_key, ctx, delta_dist);
        let ct_second =
            self.encrypt_second_in_lwe(&feature_vector, &self.private_key, ctx, delta_dist);
        Query { ct, ct_second }
    }

    fn calculate_second(&self, client_feature_vector: &Vec<u64>) -> u64 {
        let second = client_feature_vector.iter().map(|&x| x.pow(2)).sum();
        second
    }

    fn encrypt_first_in_glwe(
        &self,
        client_feature_vector: &Vec<u64>,
        private_key: &PrivateKey,
        ctx: &mut Context,
        delta_dist: u64,
    ) -> GLWE {
        private_key.allocate_and_encrypt_glwe_from_vec_customized_delta(
            client_feature_vector,
            delta_dist,
            ctx,
        )
    }

    fn encrypt_second_in_lwe(
        &self,
        client_feature_vector: &Vec<u64>,
        private_key: &PrivateKey,
        ctx: &mut Context,
        delta_dist: u64,
    ) -> LWE {
        let second_value = self.calculate_second(client_feature_vector);

        // Encrypt the polynomial in GLWE
        private_key.allocate_and_encrypt_lwe_customized_delta(second_value, delta_dist, ctx)
    }
}
