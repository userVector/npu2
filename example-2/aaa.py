from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape


def my_aaa(dev, trace_size):
    N = 65536

    # Tile sizes
    n = 512
    N_div_n = N // n

    n_cores = 1
    tiles = N_div_n // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # Type used in the tile memory
    A_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    #C_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    CA_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # Type used in the memory tile which aggregates across the 2 cores
    A_memTile_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    B_memTile_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    #C_memTile_ty = np.ndarray[n, np.dtype[bfloat16]]
    CA_memTile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # AIE Core Function declarations
    eltwise_mul_bf16_vector = Kernel(
        "eltwise_mul_bf16_vector", "mul.o", [tile_ty, tile_ty, tile_ty]
    )

    relu_mul_bf16_vector = Kernel(
        "relu_mul_bf16_vector", "mul.o", [tile_ty, tile_ty, tile_ty]
    )

    # Input A
    inA = ObjectFifo(A_memTile_ty, name="inA")#, obj_type=A_ty[0]) #placement=Tile(1, 1) obj_types=[A_ty]

    # Input B
    inB = ObjectFifo(B_memTile_ty, name="inB")

    # Output C
    #outC = ObjectFifo(C_memTile_ty, name="outC", obj_type=[C_ty]) #unnecessary

    outCA = ObjectFifo(CA_memTile_ty, name="outCA")

    
    #########
    #########
    tensor_ty2 = np.ndarray[(N,), np.dtype[bfloat16]]
    tile_ty2 = np.ndarray[(n,), np.dtype[bfloat16]]
    A_ty2 = np.ndarray[(n,), np.dtype[bfloat16]]
    B_ty2 = np.ndarray[(n,), np.dtype[bfloat16]]
    C_ty2 = np.ndarray[(n,), np.dtype[bfloat16]]
    
    C_memTile_ty2 = np.ndarray[(n,), np.dtype[bfloat16]]

    # Output C2
    outC2 = ObjectFifo(C_memTile_ty2, name="outC2")
    
    # Task for the cores to perform
    def core_fn(of_a, of_b, of_ca, eltwise_mul):
        for _ in range_(tiles):
            elem_ca = of_ca.acquire(1)
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            eltwise_mul(elem_in_a, elem_in_b, elem_ca)
            of_a.release(1)
            of_b.release(1)
            of_ca.release(1)
            
#    def core_fn2(of_a2, of_b2, of_c2, eltwise_mul):
#        for _ in range_(tiles):
#            elem_out2 = of_c2.acquire(1)
#            elem_in_a2 = of_a2.acquire(1)
#            elem_in_b2 = of_b2.acquire(1)
#            eltwise_mul(elem_in_a2, elem_in_b2, elem_out2)
#            of_a2.release(1)
#            of_b2.release(1)
#            of_c2.release(1)
            
    def core_fn2(of_a2, of_c2, eltwise_mul):
        for _ in range_(tiles):
            elem_out2 = of_c2.acquire(1)
            elem_in_a2 = of_a2.acquire(1)
            elem_in_b2 = elem_in_a2 #this is prob wrong
            eltwise_mul(elem_in_a2, elem_in_b2, elem_out2)
            of_a2.release(1)
            of_c2.release(1)

            
    # Create workers to perform the task
    workers = []
    workers.append(
        Worker(
            core_fn,
            fn_args=[
                inA.cons(),
                inB.cons(),
                #outC.prod(),
                #outCA_fifos[i].prod(),
                outCA.prod(),
                eltwise_mul_bf16_vector,
            ],
            placement=Tile(1, 2+0),
        )
    )
        
    workers.append(
        Worker(
            core_fn2,
            fn_args=[
                #inA_fifos2[i].cons(), #instead of using the input A2, we use CA
                outCA.cons(),
                #inB_fifos2[i].cons(),
                outC2.prod(),
                relu_mul_bf16_vector,
            ],
            placement=Tile(2, 2+0),
        )
    )


    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty, tensor_ty2, tensor_ty2) as (A, B, C, A2, C2):
        B2=A2
        rt.start(*workers)
        
        tg = rt.task_group()
        rt.fill(inA.prod(), A, task_group=tg, placement=Tile(1, 0))
        rt.fill(inB.prod(), B, task_group=tg, placement=Tile(1, 0))
        #rt.drain(outC.cons(), C, task_group=tg, placement=Tile(1, 0), wait=True)
        rt.finish_task_group(tg)

        tg2 = rt.task_group()
        #rt.fill(inA2.prod(), A2, task_group=tg2, placement=Tile(2, 0))
        #rt.fill(inB2.prod(), B2, task_group=tg2, placement=Tile(2, 0))
        rt.drain(outC2.cons(), C2, task_group=tg2, placement=Tile(2, 0), wait=True)
        rt.finish_task_group(tg2)
        

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())

dev = NPU2()
try:
    trace_size = 0 if (len(sys.argv) != 3) else int(sys.argv[2])
except ValueError:
    print("Argument is not an integer")
module = my_aaa(dev, trace_size)
print(module)
