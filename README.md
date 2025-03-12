# kNN

This repository contains the source code for the private evaluation of a kNN over encrypted data described in the paper **A non comparison oblivious sort and its application to kNN** accepted at Proceedings on Privacy Enhancing Technologies (PoPETs), Issue 3, 2025.

Warning: This code is a proof of concept implementation and is NOT ready for production. As such, use in production at your own risk.

## Dependencies

- [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)

Other dependencies like `revolut` are present in the `Cargo.toml` file and will be installed automatically by Cargo.

### Installing Cargo

Cargo is the Rust package manager and build system. The installation process for Cargo depends on your operating system. Follow the instructions below to install Cargo on your system.

#### macOS
1. Open a terminal.
2. Install Homebrew if you haven't already by following the instructions at [brew.sh](https://brew.sh/).
3. Use Homebrew to install Rust by running:
   ```bash
   brew install rustup
   rustup-init
   ```
4. Follow the on-screen instructions to complete the installation.
5. After installation, run:
   ```bash
   rustup update
   ```

#### Linux
1. Open a terminal.
2. Install Rust using the rustup script by running:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
3. Follow the on-screen instructions to complete the installation.
4. After installation, run:
   ```bash
   rustup update
   ```

#### Windows
1. Download and run the rustup-init.exe installer from [rustup.rs](https://rustup.rs/).
2. Follow the on-screen instructions to complete the installation.
3. After installation, open a new Command Prompt and run:
   ```bash
   rustup update
   ```


After installing Cargo, you can verify the installation by running:

```bash
cargo --version
```

## Using TFHE-rs with nightly toolchain

First, install the needed Rust toolchain:
```bash
rustup toolchain install nightly
```

Then, you can either:
Manually specify the toolchain to use in each of the cargo commands:
For example:
```bash
cargo +nightly build
cargo +nightly run
```
Or override the toolchain to use for the current project:
```bash
rustup override set nightly
# cargo will use the `nightly` toolchain.
cargo build
```

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
- test size : 1, 
- number of repetitions : 10, 

For the dataset, only two datasets are available:
- `cancer`
- `mnist`

Note : A file named `PrivateKey4` will be generated the first time you run the project. This file contains all the keys needed to run the project.