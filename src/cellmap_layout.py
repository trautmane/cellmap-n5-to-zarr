import pydantic_zarr as pz
import zarr
import os



def get_store_info(src, m_type, fibsem, inference, groundtruth, masks ):
    if os.path.splitext(src)[-1] == ".zarr":
        store = zarr.storage.NestedDirectoryStore(src)
    else:
        store = zarr.n5.N5Store(src)

    root_src = zarr.open_group(store, mode = 'r' )

    fibsem_dtypes = {}
    if root_src[fibsem].group_keys():
        fibsem_dtypes["fibsem-" + str(list(root_src[fibsem].arrays(recurse= True))[0][1].dtype)] = root_src[fibsem].name
    else:
        for group in root_src[fibsem].group_keys():
            fibsem_dtypes["fibsem-" + str(list(root_src[group].arrays(recurse= True))[0][1].dtype)] = group
        
    labels = {"inference" : inference, "groundtruth" : groundtruth , "masks" : masks}

    recon_groups = list(zip([m_type, "labels"], map(drop_empty_group, [fibsem_dtypes, labels])))
    return root_src, recon_groups

def create_cellmap_tree(recon_groups, dest, root_src, comp):
    
        dest_path = os.path.join(os.path.abspath(os.sep), dest.lstrip("/"))
        zs = zarr.storage.NestedDirectoryStore(dest_path)  
        root = zarr.open(zs, mode = 'a')
        recon = root.create_group('recon-1')

        src_dest_info = copy_arrays_info(recon_groups, root_src, zs, recon, comp)
     
        return root, src_dest_info, zs  

# tree structure: recon_group/base_group/parent/siblings
def copy_arrays_info(groups_src, root_src, store_dest, base_group_out, comp):
    src_dest_info = []
    for group in groups_src:

        parent_name = group[0]
        siblings = group[1]

        if siblings:

            parent_group = base_group_out.create_group(parent_name)

            for item in siblings.keys():
                g_src = siblings[item]
                path = os.path.join(parent_group.name, item)
                src_dest_info.append((g_src, path))
                copy_n5_tree(root_src[g_src], store_dest, path, comp)
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
