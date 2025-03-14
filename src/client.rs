use revolut::*;

use crate::{Query, GLWE, LWE};
pub struct Client {
    pub private_key: PrivateKey,
    pub public_key: PublicKey,
    pub target_vector: Vec<u64>,
}

impl Client {
    pub fn new(ctx: &mut Context, target_vector: Vec<u64>) -> Client {
        let private_key_ref = key(ctx.parameters());
        let private_key = private_key_ref.clone();
        let public_key = private_key.public_key.clone();

        Client {
            private_key,
            public_key,
            target_vector,
        }
    }

    fn encrypt_first_in_glwe(
        &self,
        client_feature_vector: &Vec<u64>,
        private_key: &PrivateKey,
        ctx: &mut Context,
        dist_modulus: u64,
    ) -> GLWE {
        private_key.allocate_and_encrypt_glwe_with_modulus(client_feature_vector, dist_modulus, ctx)
    }

    fn encrypt_second_in_lwe(
        &self,
        client_feature_vector: &Vec<u64>,
        private_key: &PrivateKey,
        ctx: &mut Context,
        dist_modulus: u64,
    ) -> LWE {
        let second_value = self.calculate_second(client_feature_vector);
        // Encrypt the value in LWE
        let ct = private_key.lwe_encrypt_with_modulus(second_value, dist_modulus, ctx);
        ct
    }

    pub fn create_query(&self, ctx: &mut Context, dist_modulus: u64) -> Query {
        let ct =
            self.encrypt_first_in_glwe(&self.target_vector, &self.private_key, ctx, dist_modulus);
        let ct_second =
            self.encrypt_second_in_lwe(&self.target_vector, &self.private_key, ctx, dist_modulus);
        Query { ct, ct_second }
    }

    fn calculate_second(&self, client_feature_vector: &Vec<u64>) -> u64 {
        let second = client_feature_vector.iter().map(|&x| x.pow(2)).sum();
        second
    }
}
