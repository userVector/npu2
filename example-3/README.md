# example-3

takes vectors A, B, C and calculates a forward pass over a two-layer relu MLP where A is input and B and C are weights, in parallel on two cores, gives one final output although test.cpp buffer interface could indicate otherwise


IMPORTANT:
custom-kernels/mul.cc is a custom kernel that contains methods with a modified functionality for development purposes
to run this example, this mul.cc needs to replace the original in mlir-aie/aie_kernels/aie2/mul.cc

note: numerical errors are okay cuz the reference CPU math is incorrect
