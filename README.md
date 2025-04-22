# kNN

This repository contains the source code for the private evaluation of a kNN over encrypted data described in the paper **A non comparison oblivious sort and its application to kNN** accepted at Proceedings on Privacy Enhancing Technologies (PoPETs), Issue 3, 2025.

Warning: This code is a proof of concept implementation and is NOT ready for production. As such, use in production at your own risk.


## Dependencies

Rust and Cargo are required to build and run the project.

Other dependencies like `revolut` are present in the `Cargo.toml` file and will be installed automatically by Cargo.

## Install Rust (Nightly `1.85.0`)

Cargo is the Rust package manager and build system. Your system must have the specific nightly version of Rust to build and run this project. Follow the instructions below to install **Rust nightly `1.85.0` (2024-12-17)**.

### macOS

1. Open a terminal.
2. Install Homebrew if you haven't already by following the instructions at [brew.sh](https://brew.sh/).
3. Install Rustup using Homebrew:
   ```bash
   brew install rustup
   rustup-init
   ```
4. Install the required Rust version:
   ```bash
   rustup install nightly-2024-12-17
   rustup default nightly-2024-12-17
   ```
5. Verify the installation:
   ```bash
   rustc --version
   cargo --version
   ```
   You should see `rustc 1.85.0-nightly (a4cb3c831 2024-12-17)`.

### Linux

1. Open a terminal.
2. Install Rust using the Rustup script:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
3. Follow the on-screen instructions.
4. Install the required Rust version:
   ```bash
   rustup install nightly-2024-12-17
   rustup default nightly-2024-12-17
   ```
5. Verify the installation:
   ```bash
   rustc --version
   cargo --version
   ```

### Windows

1. Download and run the Rustup installer from [rustup.rs](https://rustup.rs/).
2. Follow the on-screen instructions.
3. Open **Command Prompt** and install the required Rust version:
   ```bash
   rustup install nightly-2024-12-17
   rustup default nightly-2024-12-17
   ```
4. Verify the installation:
   ```bash
   rustc --version
   cargo --version

## Structure of the project

The kNN project is structured as follows:
```
knn/
├── src/
│   ├── client.rs
│   ├── model.rs
│   ├── server.rs
│   └── main.rs
├── data/
│   ├── cancer.csv
│   └── mnist.csv
├── Cargo.toml
├── Cargo.lock
├── env_test.sh  # using for environment test
├── bench_cancer.sh  # using for benchmark
├── bench_mnist.sh  # using for benchmark
├── BlindSort_bench/
└── README.md
```

### Files

The `client.rs` file contains all the code related to client structure (how to encrypt data, how to create a query, etc..).

The `model.rs` file contains all the code related to model structure (how to encode the kNN model etc..).

The `server.rs` file contains all the code related to server structure (how to perform the prediction, how to find the best model etc..).

The `main.rs` file is the entry point of the project where the client, the server and the model are instantiated and the kNN search is performed.

The `BlindSort_bench` folder is another project that contains the code for the blind sort benchmark, it contains a README file that explains how to run the benchmark.

### Scripts

The `env_test.sh` file is a script that can be used to test the environment. And the `bench_cancer.sh` and `bench_mnist.sh` files are scripts that can be used to benchmark the kNN implementation.



## Running the kNN

To run the project, follow these steps:

1. Build the project in release mode by running:
```bash
cargo build --release
```

2. Run the project with the options wanted. 

The syntax is the following:
```bash
cargo run --release -- <dataset> <k_values> <d_values> <test_size> <number_of_repetitions>
```

For example, 
```bash
    cargo run --release -- cancer 3,5 10,40,50 1 10
```
will run the project with 
- dataset : cancer, 
- k values : [3,5], 
- d values (model sizes) : [10,40,50] 
- test size (number of tested model points) : 1, 
- number of repetitions : 10, 

For the dataset, only two datasets are available:
- `cancer`
- `mnist`

Note : A file named `PrivateKey4` will be generated the first time you run the project. This file contains all the keys needed to run the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
