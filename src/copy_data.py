import os
import zarr
import dask.array as da

from dask.distributed import Client
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster

def copy_arrays_data(root_src, src_dest_info, zs):
    for src_group, dest_group in src_dest_info:

        zarrays = root_src[src_group].arrays(recurse = True)
        
        for item in zarrays:
            arr_src = item[1]
            darray = da.from_array(arr_src, chunks=arr_src.chunks)
            dataset = zarr.open_array(store = zs, path = os.path.join(dest_group.lstrip("/"), arr_src.path.lstrip("/")), mode = 'a')
            da.store(darray, dataset, lock = False)
            print(f"copied {arr_src.name} to {dest_group}")


def cluster_compute(scheduler, num_cores):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if scheduler == "lsf":
                num_cores = 8
                cluster = LSFCluster( cores=num_cores,
                        processes=1,
                        memory=f"{15 * num_cores}GB",
                        ncpus=num_cores,
                        mem=15 * num_cores,
                        walltime="01:00"
                        )
                cluster.scale(num_cores)
            elif scheduler == "local":
                    cluster = LocalCluster()

            with Client(cluster) as cl:        
                    cl.compute(function(*args, **kwargs), sync=True)
        return wrapper
    return decorator
