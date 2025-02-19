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
    n = 1024
    N_div_n = N // n

    n_cores = 2
    tiles = N_div_n // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # Type used in the tile memory
    A_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(n,), np.dtype[bfloat16]]
    C_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # Type used in the memory tile which aggregates across the 2 cores
    A_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    B_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    C_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]

    # AIE Core Function declarations
    eltwise_mul_bf16_vector = Kernel(
        "eltwise_mul_bf16_vector", "mul.o", [tile_ty, tile_ty, tile_ty]
    )

    # AIE-array data movement with object fifos
    # Input A
    inA = ObjectFifo(A_memTile_ty, name="inA")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(A_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    inA_fifos = inA.cons().split(
        of_offsets,
        obj_types=[A_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
        placement=Tile(1, 1),
    )

    # Input B
    inB = ObjectFifo(B_memTile_ty, name="inB")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(B_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    inB_fifos = inB.cons().split(
        of_offsets,
        obj_types=[B_ty] * n_cores,
        names=[f"memB{i}" for i in range(n_cores)],
        placement=Tile(1, 1),
    )

    # Output C
    outC = ObjectFifo(C_memTile_ty, name="outC")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(C_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    outC_fifos = outC.prod().join(
        of_offsets,
        obj_types=[C_ty] * n_cores,
        names=[f"memC{i}" for i in range(n_cores)],
        placement=Tile(1, 1),
    )
    
    
    
    ######33
    #########
    #########
    tensor_ty2 = np.ndarray[(N,), np.dtype[bfloat16]]
    tile_ty2 = np.ndarray[(n,), np.dtype[bfloat16]]
    
    A_ty2 = np.ndarray[(n,), np.dtype[bfloat16]]
    B_ty2 = np.ndarray[(n,), np.dtype[bfloat16]]
    C_ty2 = np.ndarray[(n,), np.dtype[bfloat16]]

    # Type used in the memory tile which aggregates across the 2 cores
    A_memTile_ty2 = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    B_memTile_ty2 = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    C_memTile_ty2 = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]

    # AIE Core Function declarations
    #eltwise_mul_bf16_vector2 = Kernel(
    #    "eltwise_mul_bf16_vector", "mul.o", [tile_ty2, tile_ty2, tile_ty2]
    #)

    # AIE-array data movement with object fifos
    # Input A
    inA2 = ObjectFifo(A_memTile_ty2, name="inA2")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(A_memTile_ty2)) // n_cores) * i
        for i in range(n_cores)
    ]
    inA_fifos2 = inA2.cons().split(
        of_offsets,
        obj_types=[A_ty2] * n_cores,
        names=[f"memA2{i}" for i in range(n_cores)],
        placement=Tile(2, 1),
    )

    # Input B
    inB2 = ObjectFifo(B_memTile_ty2, name="inB2")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(B_memTile_ty2)) // n_cores) * i
        for i in range(n_cores)
    ]
    inB_fifos2 = inB2.cons().split(
        of_offsets,
        obj_types=[B_ty2] * n_cores,
        names=[f"memB2{i}" for i in range(n_cores)],
        placement=Tile(2, 1),
    )

    # Output C
    outC2 = ObjectFifo(C_memTile_ty2, name="outC2")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(C_memTile_ty2)) // n_cores) * i
        for i in range(n_cores)
    ]
    outC_fifos2 = outC2.prod().join(
        of_offsets,
        obj_types=[C_ty2] * n_cores,
        names=[f"memC2{i}" for i in range(n_cores)],
        placement=Tile(2, 1),
    )
    
    
    
    
    
    
    
    
    
    
    
    

    # Task for the cores to perform
    def core_fn(of_a, of_b, of_c, eltwise_mul):
        for _ in range_(tiles):
            elem_out = of_c.acquire(1)
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            eltwise_mul(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)
            
    def core_fn2(of_a2, of_b2, of_c2, eltwise_mul):
        for _ in range_(tiles):
            elem_out2 = of_c2.acquire(1)
            elem_in_a2 = of_a2.acquire(1)
            elem_in_b2 = of_b2.acquire(1)
            eltwise_mul(elem_in_a2, elem_in_b2, elem_out2)
            of_a2.release(1)
            of_b2.release(1)
            of_c2.release(1)

    # Create workers to perform the task
    workers = []
    for i in range(n_cores):
        workers.append(
            Worker(
                core_fn,
                fn_args=[
                    inA_fifos[i].cons(),
                    inB_fifos[i].cons(),
                    outC_fifos[i].prod(),
                    eltwise_mul_bf16_vector,
                ],
                placement=Tile(1, 2+i),
            )
        )
        
    for i in range(n_cores):
        workers.append(
            Worker(
                core_fn2,
                fn_args=[
                    inA_fifos2[i].cons(),
                    inB_fifos2[i].cons(),
                    outC_fifos2[i].prod(),
                    eltwise_mul_bf16_vector,
                ],
                placement=Tile(2, 2+i),
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
        rt.drain(outC.cons(), C, task_group=tg, placement=Tile(1, 0), wait=True)
        rt.finish_task_group(tg)

        tg2 = rt.task_group()
        rt.fill(inA2.prod(), A2, task_group=tg2, placement=Tile(2, 0))
        rt.fill(inB2.prod(), B2, task_group=tg2, placement=Tile(2, 0))
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
