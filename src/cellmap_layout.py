import pydantic_zarr as pz
import zarr
import os

def get_store_info(src, m_type, fibsem, inference, groundtruth, masks, lm):
    #em data
    if src:
        if os.path.splitext(src)[-1] == ".zarr":
            store = zarr.storage.NestedDirectoryStore(src)
        else:
            store = zarr.n5.N5Store(src)

        root_src = zarr.open_group(store, mode = 'r')

        fibsem_dtypes = {}
        if fibsem:
            fibsem_dtypes["fibsem-" + str(list(root_src[fibsem].arrays(recurse= True))[0][1].dtype)] = os.path.join(os.path.abspath(os.sep), src.strip(" /"), fibsem) 
        else:
            for group in root_src[fibsem].group_keys():
                fibsem_dtypes["fibsem-" + str(list(root_src[group].arrays(recurse= True))[0][1].dtype)] = os.path.join(os.path.abspath(os.sep), src.lstrip(" /"))
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

# tree structure: recon_group/base_group/parent/siblings
def copy_arrays_info(groups_src, store_dest, base_group_out, comp):
    src_dest_info = []
    for group in groups_src:
        parent_name = group[0]
        siblings = group[1]

        if siblings:
            
            for item in siblings.keys():
                
                src_store = zarr.n5.N5Store(os.path.join(os.path.abspath(os.sep), siblings[item].lstrip("/")))
                src_group = zarr.open_group(src_store, mode = 'r')

                if parent_name == "":
                    dest_path = os.path.join(base_group_out.name, item)
                else: 
                    dest_path = os.path.join(base_group_out.name, parent_name, item)

                src_dest_info.append((src_group, dest_path))

                copy_n5_tree(src_group, store_dest, dest_path, comp)
            
    return src_dest_info

# d=groupspec.to_dict(),  
def normalize_groupspec(d, comp):
    for k,v in d.items():
        if k == "compressor":
            d[k] = comp.get_config()

        elif k == 'dimension_separator':
            d[k] = '/'
        elif isinstance(v,  dict):
            normalize_groupspec(v, comp)

def copy_n5_tree(n5_root, z_store, path, comp):
    spec_n5 = pz.GroupSpec.from_zarr(n5_root)
    spec_n5_dict = spec_n5.dict()
    normalize_groupspec(spec_n5_dict, comp)
    spec_n5 = pz.GroupSpec(**spec_n5_dict)
    return spec_n5.to_zarr(z_store, path=path)

def drop_empty_group(group_dict):
    for k in group_dict.copy():
        if group_dict[k] == "":
            del group_dict[k]
    return group_dict
