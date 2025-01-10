import sys
import click
from numcodecs import Zstd

import zarr_attrs_multiscales as to_ngff
import copy_data as cd
import cellmap_layout as cml

import zarr

@click.command()
@click.option('--src', '-s', type=click.Path(exists = True))
@click.option("--dest", '-d', type=click.Path())
@click.option('--mtype', '-mt', default = "em", type=click.STRING)
@click.option('--gtruth', '-gt', default = "", type=click.STRING)
@click.option('--inf', '-i' , default = "", type=click.STRING)
@click.option('--masks', '-m', default = "", type=click.STRING)
@click.option('--lm', '-lm', default = "", type=click.STRING)
@click.option('--num_workers', '-w', default = 100, type=click.INT)
@click.option('--cluster', '-c', default = '', type=click.STRING,
              help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'")
@click.option('--clevel', '-cl', default = 6, type=click.INT)
@click.option('--max_dask_chunk_num', '-maxchnum' , default = 50000, type=click.INT)
@click.option('--dry', default = False, type=click.BOOL)
@click.option('--lsf_runtime_limit', default = '48:00', type=click.STRING,
              help="The runtime limit of the LSF job in [hour:]minute form.")
@click.option('--lsf_project_name', default = None, type=click.STRING,
              help="The LSF project to bill.  Omit if your lab or project group can be billed by default.")
@click.option('--lsf_worker_log_dir', default = None, type=click.STRING,
              help="The path of the parent directory for all LSF worker logs.  Omit if you want worker logs to be emailed to you.")
def cli(src,
        dest,
        mtype,
        gtruth,
        inf,
        masks,
        lm,
        num_workers,
        cluster,
        clevel,
        max_dask_chunk_num,
        dry,
        lsf_runtime_limit,
        lsf_project_name,
        lsf_worker_log_dir):
    
    if cluster == '':
        print('Did not specify which instance of the dask client to use!')
        sys.exit(0)
    
    compressor = Zstd(level=clevel)

    #figure out the layout of an output .zarr file. 
    recon_groups = cml.get_store_info(src, mtype, inference = inf, groundtruth = gtruth, masks = masks, lm = lm)

    #copy groups and arrays info to an output zarr file.  
    root_dest, src_dest_info, zs = cml.create_cellmap_tree(recon_groups, dest, compressor)
    
    #add ome ngff multiscale metadata, if applicable
    for item in src_dest_info:
        if isinstance(item[0], zarr.hierarchy.Group): 
            dest_group = root_dest[item[1]]    
            to_ngff.normalize_to_ngff(dest_group)
    
    #copy input .n5 arrays data to corresponding arrays in the output .zarr file.
    if not dry:
        lsf_job_extra_directives = []
        if lsf_project_name is not None:
            lsf_job_extra_directives.append(f"-P {lsf_project_name}")

        copy_arrays_data = cd.cluster_compute(cluster,
                                              num_workers,
                                              lsf_runtime_limit,
                                              lsf_worker_log_dir,
                                              lsf_job_extra_directives)(cd.copy_arrays_data)

        copy_arrays_data(src_dest_info, zs, max_dask_chunk_num, compressor)
   

if __name__ == '__main__':
    cli()
