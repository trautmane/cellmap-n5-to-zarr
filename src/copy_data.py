import os
import zarr
import dask.array as da

from dask.distributed import Client
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster
import time 
import numpy as np


def copy_arrays_data(src_dest_info, zs, max_dask_chunk_num):
    for src_group, dest_group in src_dest_info:
        src_group = zarr.open_group(src_group, mode = 'r') 
        zarrays = src_group.arrays(recurse = True)
        
        for item in zarrays:
            start_time = time.time()
            arr_src = item[1]
            #the chunk sizing of a dask array has a very big impact on computation performance
            # for example: for ~10TB dataset, use ~300MB chunk size.
            darray = da.from_array(arr_src, chunks=optimal_dask_chunksize(arr_src, max_dask_chunk_num))
            dataset = zarr.open_array(store = zs, path = os.path.join(dest_group.lstrip("/"), arr_src.path.lstrip("/")), mode = 'a')
            da.store(darray, dataset, lock = False)
            copy_time = time.time() - start_time
            print(f"({copy_time}s) copied {arr_src.name} to {dest_group}")

def cluster_compute(scheduler, num_cores):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if scheduler == "lsf":
                num_cores = 30
                cluster = LSFCluster( cores=num_cores,
                        processes=1,
                        memory=f"{15 * num_cores}GB",
                        ncpus=num_cores,
                        mem=15 * num_cores,
                        walltime="48:00"
                        )
                cluster.scale(num_cores)
            elif scheduler == "local":
                    cluster = LocalCluster()

            with Client(cluster) as cl:
                text_file = open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w")
                text_file.write(str(cl.dashboard_link))
                text_file.close()
                cl.compute(function(*args, **kwargs), sync=True)
        return wrapper
    return decorator

# raise dask warning
def chunk_num_warning(darr):
    chunk_num =  da.true_divide(darr.shape, darr.chunksize).prod()
    if (chunk_num > pow(10, 5)):
        
        log_file_path = os.path.join(os.getcwd(), "warnings")
        os.mkdir(log_file_path)

        warning_file = open(os.path.join(log_file_path, "dask_warning" + ".txt"), "a")
        warning_file.write("Warning: dask array contains more than 100,000 chunks.")
        warning_file.close()  

# calculate automatically what chunk size scaling we should have in order to avoid having a complex dask computation graph. 
def optimal_dask_chunksize(arr, max_dask_chunk_num):
    #calculate number of chunks within a zarr array.
    chunk_num= np.prod([arr_dim / chunk_dim for arr_dim, chunk_dim in zip(arr.shape, arr.chunks)])
    print(f"chunk_num: {chunk_num}")

    scaling = 1
    while chunk_num > max_dask_chunk_num:
        print(f"chunk_num: {chunk_num}")
        chunk_num = chunk_num / scaling
        scaling +=1
    print(f"scaling factor for {arr.name} is {scaling}")
    print(f'chunks: {tuple(dim * scaling for dim in arr.chunks)}')
    return tuple(dim * scaling for dim in arr.chunks) 