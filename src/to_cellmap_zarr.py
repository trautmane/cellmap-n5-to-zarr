import zarr
import os
import click
from numcodecs import Blosc
from pathlib import Path

import zarr_attrs_multiscales as to_ngff
import copy_data as cd
#from copy_data import cluster_compute
import cellmap_layout as cml


# src = '../../test_data/input/crop1.zarr'
# dest = '../../test_data/output/output1.zarr'

# inference = ""
# groundtruth = ""
# masks = ""
# fibsem = "/"
# m_type = "em"
# comp = Blosc(cname="lz4", clevel=5, shuffle=1)

@click.command()
@click.option('--src', '-s', type=click.Path(exists = True))
@click.option("--dest", '-d', type=click.Path())
@click.option("--dataset", '-ds', type=click.STRING)
@click.option('--mtype', '-mt', default = "em", type=click.STRING)
@click.option('--gtruth', '-gt', default = "", type=click.STRING)
@click.option('--inf', '-i' , default = "", type=click.STRING)
@click.option('--masks', '-m', default = "", type=click.STRING)
@click.option('--num_cores', '-c', default = 8, type=click.INT)
@click.option('--scheduler', '-s', default = "local", type=click.STRING)
@click.option('--cname', "-cn", default = "zstd", type=click.STRING)
@click.option('--clevel', '-cl', default = 9, type=click.INT)
@click.option('--shuffle', '-sh' , default = 0, type=click.INT)
def cmzarr(src, dest, dataset, mtype, gtruth, inf, masks, num_cores, scheduler, cname, clevel, shuffle):
    compressor = Blosc(cname=cname, clevel=clevel, shuffle=shuffle)


    root_src, recon_groups = cml.get_store_info(src, mtype, dataset, inference = inf, groundtruth = gtruth, masks = masks)

    root_dest, src_dest_info, zs = cml.create_cellmap_tree(recon_groups, dest, root_src, compressor)
    to_ngff.normalize_to_ngff(root_dest)
 
    copy_arrays_data = cd.cluster_compute(scheduler, num_cores)(cd.copy_arrays_data)
    copy_arrays_data( root_src, src_dest_info, zs)

    

if __name__ == '__main__':
    cmzarr()