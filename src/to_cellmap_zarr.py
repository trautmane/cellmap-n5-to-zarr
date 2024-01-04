import zarr
import os
import click
from numcodecs import Zstd
from pathlib import Path

import zarr_attrs_multiscales as to_ngff
import copy_data as cd
#from copy_data import cluster_compute
import cellmap_layout as cml

import os
import zarr
import dask.array as da

@click.command()
@click.option('--src', '-s', type=click.Path(exists = True))
@click.option("--dest", '-d', type=click.Path())
@click.option("--dataset", '-ds', type=click.STRING)
@click.option('--mtype', '-mt', default = "em", type=click.STRING)
@click.option('--gtruth', '-gt', default = "", type=click.STRING)
@click.option('--inf', '-i' , default = "", type=click.STRING)
@click.option('--masks', '-m', default = "", type=click.STRING)
@click.option('--lm', '-lm', default = "", type=click.STRING)
@click.option('--num_cores', '-c', default = 8, type=click.INT)
@click.option('--scheduler', '-s', default = "local", type=click.STRING)
@click.option('--cname', "-cn", default = "zstd", type=click.STRING)
@click.option('--clevel', '-cl', default = 6, type=click.INT)
@click.option('--shuffle', '-sh' , default = 0, type=click.INT)
@click.option('--max_dask_chunk_num', '-maxchnum' , default = 50000, type=click.INT)
def cli(src, dest, dataset, mtype, gtruth, inf, masks, lm, num_cores, scheduler, cname, clevel, shuffle, max_dask_chunk_num):
    compressor = Zstd(level=clevel)

    #figure out the layout of an output .zarr file. 
    recon_groups = cml.get_store_info(src, mtype, dataset, inference = inf, groundtruth = gtruth, masks = masks, lm = lm)

    #copy groups and arrays info to an output zarr file.  
    root_dest, src_dest_info, zs = cml.create_cellmap_tree(recon_groups, dest, compressor)

    #add ome ngff multiscale metadata, if applicable
    to_ngff.normalize_to_ngff(root_dest)
    
    #copy input .n5 arrays data to corresponding arrays in the output .zarr file.
    copy_arrays_data = cd.cluster_compute(scheduler, num_cores)(cd.copy_arrays_data)
    copy_arrays_data(src_dest_info, zs, max_dask_chunk_num)

    

if __name__ == '__main__':
    cli()
