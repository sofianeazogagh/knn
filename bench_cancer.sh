# This script runs the kNN implementation with different parameters.
# The syntax for running the project is:
# cargo run --release -- <dataset> <k_values> <d_values> <test_size> <number_of_repetitions>
# 
# Options:
# - <dataset>: The dataset to use. Available options are 'cancer' and 'mnist'.
# - <k_values>: Comma-separated list of k values to test.
# - <d_values>: Comma-separated list of model sizes (d values) to test.
# - <test_size>: The number of test samples to use.
# - <number_of_repetitions>: The number of times to repeat the test.


cargo run --release -- cancer 3,5 10,30,50,200 200 10