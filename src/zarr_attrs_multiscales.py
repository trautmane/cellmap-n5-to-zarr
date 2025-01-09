import zarr 
import json
import os
from operator import itemgetter
import natsort
import re

def apply_ngff_template(zgroup):

    src_dir = os.path.dirname(os.path.abspath(__file__))

    f_zattrs_template = open(f'{src_dir}/zarr_attrs_template.json')
    z_attrs = json.load(f_zattrs_template)
    f_zattrs_template.close()

    junits = open(f'{src_dir}/unit_names.json')
    unit_names = json.load(junits)
    junits.close()

    units_list = []

    for unit in zgroup.attrs['units']:
        if unit in unit_names.keys():
            units_list.append(unit_names[unit])
        else:
            units_list.append(unit)

    #try:
    axes_reverse = zgroup.attrs['axes'][::-1]
    #except KeyError:
    #    pass
    #populate .zattrs
    z_attrs['multiscales'][0]['axes'] = [{"name": axis, 
                                          "type": "space",
                                           "unit": unit} for (axis, unit) in zip(axes_reverse, 
                                                                                 units_list)]
    z_attrs['multiscales'][0]['version'] = '0.4'
    z_attrs['multiscales'][0]['name'] = zgroup.name
    z_attrs['multiscales'][0]['coordinateTransformations'] = [
                {
                    "scale": [
                        1.0,
                        1.0,
                        1.0
                    ],
                    "type": "scale"
                }
            ]
    # delete legacy n5 attrs
    if 'axes' in z_attrs:
        del z_attrs['axes']
    
    if 'pixelResolution' in z_attrs:
        del z_attrs['pixelResolution']
    
    if 'scales' in z_attrs:
        del z_attrs['scales']
        
    if 'units' in z_attrs:
        del z_attrs['units']
        
    if 'translate' in z_attrs:
        del z_attrs['translate']
        
    if 'scales' in z_attrs:
        del z_attrs['scales']
        
    if 'ordering' in z_attrs:
        del z_attrs['ordering']
    
    return z_attrs

def normalize_to_ngff(zgroup):
    group_keys = list(zgroup.group_keys())
    group_keys.append("")
    
    for key in group_keys:
        if isinstance(zgroup[key], zarr.hierarchy.Group):

            if key != "":
                normalize_to_ngff(zgroup[key])
            if 'scales' in zgroup[key].attrs.asdict():
                zattrs = apply_ngff_template(zgroup[key])
                zarrays = list(zgroup[key].arrays(recurse=True))

                unsorted_datasets = []
                for arr in zarrays:
                    dataset_meta = ome_dataset_metadata(arr[1], zgroup[key], int(re.findall(r'\d+', arr[0])[0]))
                    unsorted_datasets.append(dataset_meta)
                    
                #1.apply natural sort to organize datasets metadata array for different resolution degrees (s0 -> s10)
                #2.add datasets metadata to the ngff template
                #print(natsort.natsorted(unsorted_datasets, key=itemgetter(*['path'])))
                zattrs['multiscales'][0]['datasets'] = natsort.natsorted(unsorted_datasets, key=itemgetter(*['path']))
                zgroup[key].attrs['multiscales'] = zattrs['multiscales']


def ome_dataset_metadata(n5arr, group, i):
    scale = [scale * dim for scale, dim in zip(group.attrs['scales'][i], group.attrs['pixelResolution']['dimensions'])]
    reverse_scale = scale[::-1]
    translation = [s0*(scale - 1)/2 for s0, scale in zip(group.attrs['pixelResolution']['dimensions'], group.attrs['scales'][i])]
    reverse_translation = translation[::-1]
    dataset_meta =  {
                    "path": os.path.relpath(n5arr.path, group.path),
                    "coordinateTransformations": [{
                        'type': 'scale',
                        'scale': reverse_scale
                        },
                        {
                        'type': 'translation',
                        'translation' : reverse_translation
                        }
                    ]}
    
    return dataset_meta