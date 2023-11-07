import zarr 
import os
import json
import os
from operator import itemgetter
import natsort

def apply_ngff_template(zgroup):
    
    f_zattrs_template = open('src/zarr_attrs_template.json')
    z_attrs = json.load(f_zattrs_template)
    f_zattrs_template.close()

    junits = open('src/unit_names.json')
    unit_names = json.load(junits)
    junits.close()

    units_list = []

    for unit in zgroup.attrs['units']:
        if unit in unit_names.keys():
            units_list.append(unit_names[unit])
        else:
            units_list.append(unit)

    #populate .zattrs
    z_attrs['multiscales'][0]['axes'] = [{"name": axis, 
                                          "type": "space",
                                           "unit": unit} for (axis, unit) in zip(zgroup.attrs['axes'], 
                                                                                 units_list)]
    z_attrs['multiscales'][0]['version'] = '0.4'
    z_attrs['multiscales'][0]['name'] = zgroup.name
    z_attrs['multiscales'][0]['coordinateTransformations'] = [{"type": "scale",
                    "scale": [1.0, 1.0, 1.0]}, {"type" : "translation", "translation" : [1.0, 1.0, 1.0]}]
    
    return z_attrs

def normalize_to_ngff(zgroup):
    group_keys = zgroup.keys()

    for key in group_keys:
        if isinstance(zgroup[key], zarr.hierarchy.Group):

            normalize_to_ngff(zgroup[key])
            if 'scales' in zgroup[key].attrs.asdict():
                zattrs = apply_ngff_template(zgroup[key])
                zarrays = zgroup[key].arrays(recurse=True)

                unsorted_datasets = []
                for arr in zarrays:
                    unsorted_datasets.append(ome_dataset_metadata(arr[1], zgroup[key]))

                #1.apply natural sort to organize datasets metadata array for different resolution degrees (s0 -> s10)
                #2.add datasets metadata to the ngff template
                zattrs['multiscales'][0]['datasets'] = natsort.natsorted(unsorted_datasets, key=itemgetter(*['path']))
                zgroup[key].attrs['multiscales'] = zattrs['multiscales']


def ome_dataset_metadata(n5arr, group):
    text_file = open(os.path.join(os.getcwd(), "attrs", "_".join(str(n5arr.name).split("/")) + ".txt"), "w")
    text_file.write(str(sorted(n5arr.attrs)))
    text_file.close()
    arr_attrs_n5 = n5arr.attrs['transform']
    dataset_meta =  {
                    "path": os.path.relpath(n5arr.path, group.path),
                    "coordinateTransformations": [{
                        'type': 'scale',
                        'scale': arr_attrs_n5['scale']},{
                        'type': 'translation',
                        'translation' : arr_attrs_n5['translate']
                    }]}
    
    return dataset_meta