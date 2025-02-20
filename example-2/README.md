# example-2

takes vectors A and B and calculates A*B=C on 1 core and in parallel relu(C^2) on another core, gives one final output although test.cpp buffer interface could indicate otherwise


IMPORTANT:
custom-kernels/mul.cc is a custom kernel that contains methods with a modified functionality for development purposes
to run this example, this mul.cc needs to replace the original in mlir-aie/aie_kernels/aie2/mul.cc

note: numerical errors are okay cuz the reference CPU math is incorrect
