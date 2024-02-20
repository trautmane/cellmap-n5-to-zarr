import os
import zarr
import dask.array as da

from dask.distributed import Client
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster
import time 
import numpy as np


def copy_arrays_data(src_dest_info, zs, max_dask_chunk_num, comp):
    for src_obj, dest_group in src_dest_info:

        if isinstance(src_obj, zarr.core.Array):
            zarrays = [(src_obj.basename, src_obj)]
        else:
            zarrays = src_obj.arrays(recurse = True)
        
        for item in list(zarrays):
            start_time = time.time()
            arr_src = item[1]

            #the chunk sizing of a dask array has a very big impact on computation performance
            # for example: for ~10TB dataset, use ~300MB chunk size.
            darray = da.from_array(arr_src, chunks=optimal_dask_chunksize(arr_src, max_dask_chunk_num))
            
            if isinstance(src_obj, zarr.core.Array):
                dataset = zarr.open(store = zs, path = dest_group,
                                    mode='w', shape=arr_src.shape, chunks=arr_src.chunks, dtype=arr_src.dtype, compressor=comp)
            else:
                arr_path = arr_src.path.replace(src_obj.path, '')
                dataset = zarr.open(store =zs, path=os.path.join(dest_group.lstrip("/"), arr_path.lstrip("/")),
                                    mode='w', shape=arr_src.shape, chunks=arr_src.chunks, dtype=arr_src.dtype, compressor=comp)

            da.store(darray, dataset, lock = False)
            copy_time = time.time() - start_time
            print(f"({copy_time}s) copied {arr_src.name} to {dest_group}")

def cluster_compute(scheduler, num_cores):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if scheduler == "lsf":
                num_cores = 40
                cluster = LSFCluster( cores=num_cores,
                        processes=1,
                        memory=f"{15 * num_cores}GB",
                        ncpus=num_cores,
                        mem=15 * num_cores,
                        walltime="48:00",
                        death_timeout = 240.0,
                        local_directory = "/scratch/zubovy/"
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
    if isinstance(arr, zarr.core.Array):
        chunk_dims = arr.chunks
    else:
        chunk_dims = arr.chunksize
    chunk_num= np.prod(arr.shape)/np.prod(chunk_dims) 
    
    # 1. Scale up chunk size (chunksize approx = 1GB)
    scaling = 1
    while np.prod(chunk_dims)*arr.itemsize*pow(scaling, 3)/pow(10, 6) < 300 :
        scaling += 1

    # 3. Number of chunks should be < 50000
    while (chunk_num / pow(scaling,3)) > max_dask_chunk_num:
        scaling +=1

    # 2. Make sure that chunk dims < array dims
    while any([ch_dim > 3*arr_dim/4 for ch_dim, arr_dim in zip(tuple(dim * scaling for dim in chunk_dims), arr.shape)]):#np.prod(chunks)*arr.itemsize*pow(scaling,3) > arr.nbytes:
        scaling -=1

    if scaling == 0:
        scaling = 1

    return tuple(dim * scaling for dim in chunk_dims) 
