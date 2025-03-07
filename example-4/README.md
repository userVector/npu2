# example-4

adaptation of example-3 to demonstrate the broadcasting data-movement pattern (ObjectFifo with a single producer but multiple consumers)


IMPORTANT:
custom-kernels/mul.cc is a custom kernel that contains methods with a modified functionality for development purposes
to run this example, this mul.cc needs to replace the original in mlir-aie/aie_kernels/aie2/mul.cc

note: numerical errors are okay cuz the reference CPU math is incorrect
