# BlindSort

This repository contains the source code to launch some benchmarks on the `Blind Counting Sort` algortihm presented in the paper **A non comparison oblivious sort and its application to kNN** accepted at Proceedings on Privacy Enhancing Technologies (PoPETs), Issue 3, 2025.

Warning: This code is only for benchmarking purposes.

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
   


## Build

```sh
cargo build --release
```

## Run

```sh
cargo run --release
```

It should display the result of the sort for different number of elements for the `Blind Counting Sort` algorithm and for an implementation of the bitonic sort algorithm using `tfhe-rs`.

Note : Files named `PrivateKey{i}` will be generated the first time you run the program. These files contain all the keys needed to run the benchmarks for various parameter sizes. `PrivateKey7` should take some time and memory to be generated. Once the keys are generated, the display should be nicer.