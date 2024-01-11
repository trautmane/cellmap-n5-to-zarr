import pydantic_zarr as pz
import zarr
import os
import re
from fibsem_tools.io.zarr import n5_spec_unwrapper

def get_store_info(src, m_type, inference, groundtruth, masks, lm):
    #em data
    if src:
        store_path, path = split_n5_input(src)
        store = open_store(store_path)
        root_src = zarr.open(store = store, path = path, mode = 'r')

        fibsem_dtypes = {}
        #if input is a .zarr array
        if isinstance(root_src, zarr.core.Array):
            fibsem_dtypes["fibsem-" + str(root_src.dtype)] = os.path.join(os.path.abspath(os.sep), src.strip(" /"))
        else:
            #if input is a group that contains a set of groups, and we want to preserve the group hierarchy           
            # if list(root_src.group_keys()):
            #     for group in root_src.group_keys():                  
            #         fibsem_dtypes["fibsem-" + str(list(root_src[group].arrays(recurse= True))[0][1].dtype)] = os.path.join(os.path.abspath(os.sep), src.lstrip(" /"))
            # # if input is a group 
            # else:
            fibsem_dtypes["fibsem-" + str(list(root_src.arrays(recurse= True))[0][1].dtype)] = os.path.join(os.path.abspath(os.sep), src.lstrip(" /"))
    else:
        fibsem_dtypes = { 'fibsem' : ""}
        
    labels = {"inference" : inference, "groundtruth" : groundtruth , "masks" : masks}
    #light microscopy data
    light_m = {"lm" : lm}

    recon_groups = list(zip([m_type, "labels", ""], map(drop_empty_group, [fibsem_dtypes, labels, light_m])))

    return recon_groups

def create_cellmap_tree(recon_groups, dest, comp):
    
        dest_path = os.path.join(os.path.abspath(os.sep), dest.lstrip("/"))
        zs = zarr.storage.NestedDirectoryStore(dest_path)  
        root = zarr.open(zs, mode = 'a')
        if 'recon-1' in root.group_keys():
            recon = root['recon-1']
        else:
            recon = root.create_group('recon-1')

        src_dest_info = copy_arrays_info(recon_groups, zs, recon, comp)
     
        return root, src_dest_info, zs  

# tree structure: recon_group/base_group(em, labels)/parent(fibsem-dtype, groundtruth, ...)/siblings
def copy_arrays_info(groups_src, store_dest, base_group_out, comp):
    src_dest_info = []
    for group in groups_src:
        parent_name = group[0]
        siblings = group[1]

        if siblings:
            
            for item in siblings.keys():
                
                src_store, path = split_n5_input(os.path.join(os.path.abspath(os.sep), siblings[item].lstrip("/")))
                src_obj = zarr.open(store=src_store, path=path, mode = 'r')

                if parent_name == "":
                    dest_path = os.path.join(base_group_out.name, item)
                else: 
                    dest_path = os.path.join(base_group_out.name, parent_name, item)
                
                if isinstance(src_obj, zarr.core.Array):
                    dest_path =  os.path.join(dest_path, src_obj.basename)    

                src_dest_info.append((src_obj, dest_path))
                copy_n5_tree(src_obj, store_dest, dest_path, comp) 

    return src_dest_info

def copy_n5_tree(n5_root, z_store, path, comp):

    if isinstance(n5_root, zarr.core.Array):
        spec_n5 = pz.ArraySpec.from_zarr(n5_root)
    else:
        spec_n5 = pz.GroupSpec.from_zarr(n5_root)

    # transform N5 GroupSpec into Zarr-compatible GroupSpec
    spec_zarr = n5_spec_unwrapper(spec_n5)
    return  spec_zarr.to_zarr(z_store, path=path)

def drop_empty_group(group_dict):
    for k in group_dict.copy():
        if group_dict[k] == "":
            del group_dict[k]
    return group_dict

def open_store(path):
    if os.path.splitext(path)[-1] == ".zarr":
        store = zarr.storage.NestedDirectoryStore(path)
    else:
        store = zarr.n5.N5Store(path)
    return store

def split_n5_input(src):
    split = re.split("\.n5", src)
    return "".join([split[0], ".n5"]), split[1]

