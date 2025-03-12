# Artifact Appendix

Paper title: **A non comparison oblivious sort and its application to private k-NN**

Artifacts HotCRP Id: **16**

Requested Badge: **Available**, **Functional** and **Reproduced**

## Description
This artifact contains a `knn` folder which includes the code and data necessary to evaluate the private k-NN presented in the paper. This implementation uses the Blind Counting Sort algorithm directly implemented in the `RevoLUT` library (tag `v0.1.0`). Additionally, it includes a `BlindSort_bench` folder to run the benchmarks of the Blind Counting Sort algorithm.


### Security/Privacy Issues and Ethical Concerns (All badges)
No security or privacy issues are known.

## Basic Requirements (Only for Functional and Reproduced badges)

### Hardware Requirements

By default, TFHE-rs makes the assumption that hardware AES features are enabled on the target CPU. The required CPU features are:
- x86_64: sse2, aesni
- aarch64: aes, neon

### Software Requirements

Here are the software versions used to develop the artifact:
- rustc 1.85.0-nightly (a4cb3c831 2024-12-17)


### Estimated Time and Storage Consumption

A computer with sufficient RAM (>=16GB) and CPU power to generate all the keys for the BlindSort bench.

- BlindSort bench: Generating the largest key, PrivateKey7, requires approximately 3 minutes on a machine with 32GB of RAM and 12 cores, and the key size is around 10GB. Other keys can be generated on any modern machine with at least 16GB of RAM without issues.
- kNN: The key size for the kNN part is relatively small (235MB) and the time to complete the kNN computation depends on the user-defined parameters.

## Environment 

### Accessibility (All badges)


- [GitHub Repository - sofianeazogagh/knn](https://github.com/sofianeazogagh/knn)
- Tag : `v1.0.0-eval`

### Set up the environment (Only for Functional and Reproduced badges)

```bash
git clone https://github.com/sofianeazogagh/knn.git
cd knn
git checkout v1.0.0-eval
```

Follow the instructions in the [README](README.md) to install the required Rust version depending on your OS.

### Testing the Environment (Only for Functional and Reproduced badges)

Once the environment is set up, you can test the kNN implementation with the following command:
```bash
chmod +x env_test.sh
./env_test.sh
```

This will run the kNN implementation with the following parameters:
- dataset : cancer
- k values : [3,5]
- d values (model sizes) : [10,40,50]
- test size : 1
- number of repetitions : 10

The results are displayed in the console after the computation is finished. Note that a key file named `PrivateKey4` will be generated the first time you run the project. This file contains all the keys needed to run the project. You can allow more verbose output by changing the macro `VERBOSE` in the `knn/src/main.rs` file.

## Artifact Evaluation (Only for Functional and Reproduced badges)

### Main Results and Claims

#### Main Result 1: Private k-NN computation

The first main result is the micro-benchmarking of the private k-NN implementation for the datasets `cancer` and `mnist`. These results are found in the Section 6.2 of the paper. 
Regarding the accuracy :
- for the `cancer` dataset, the accuracy is aligned with the clear kNN implementation.
- for the `mnist` dataset, the accuracy is 1 to 3 percent below the clear kNN implementation.


#### Main Result 2: Micro-benchmarking of the Blind Counting Sort algorithm

The second main result is the micro-benchmarking of the Blind Counting Sort algorithm comparing to the bitonic sort algorithm implemented with `tfhe-rs`API. These results are found in the Section 6.1 of the paper.

### Experiments 

#### Experiment 1: Private k-NN computation for the dataset `cancer`

To run the kNN implementation for the dataset `cancer`, you can use the following command:
```bash
chmod +x bench_cancer.sh
./bench_cancer.sh > cancer_knn.txt
```
The results will be saved in the `cancer_knn.txt` file. It takes around 10 hours to complete all the computation (i.e all the lines of the Table 8 in the paper). You can change the parameters in the `bench_cancer.sh` file to run the benchmark partly and takes less time. For instance, you can change the `k_values` to `3` and the `d_values` to `10` to run the first line of the Table 8 in the paper.

#### Experiment 2: Private k-NN computation for the dataset `mnist`

To run the kNN implementation for the dataset `mnist`, you can use the following command:
```bash
chmod +x bench_mnist.sh
./bench_mnist.sh > mnist_knn.txt
```
The results will be saved in the `mnist_knn.txt` file. It takes around 20 hours to complete all the computation (i.e all the lines of the Table 9 in the paper). You can change the parameters in the `bench_mnist.sh` file to run the benchmark partly and takes less time. For instance, you can change the `k_values` to `3` and the `d_values` to `40` to run one line of the Table 9 in the paper.


#### Experiment 3: Blind Counting Sort

To run the Blind Counting Sort benchmark, you can use the following command:
```bash
chmod +x BlindSort_bench/launcher.sh
./BlindSort_bench/launcher.sh > blind_sort.txt
```
The results will be saved in the `blind_sort.txt` file. Note that the first time you run the `launcher.sh` script, it will display the time taken by revolut to generate the keys for the different message sizes. The resulting file will be saved in the `blind_sort.txt` file.

## Limitations (Only for Functional and Reproduced badges)

The exact number of the Table 10 (accuracy on MNIST)may not be reproducible with the provided artifact because we did not save the model that gave the results. However as mentioned in the paper, the important point is that the accuracy of the private kNN is close to the clear kNN implementation.

## Notes on Reusability (Only for Functional and Reproduced badges)

The Rust implementations of the Blind Counting Sort implemented in the `RevoLUT` library are reusable for other projects. The `BlindSort_bench` folder can be used to benchmark other sorting algorithms.