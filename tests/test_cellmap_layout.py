import pytest
import dask.array as da
import numpy as np
import zarr
from numcodecs import Zstd
import pydantic_zarr as pz
import os
from fibsem_tools.io.zarr import n5_spec_unwrapper



import cellmap_layout as cml 

@pytest.fixture(scope='session')
def filepaths(tmp_path_factory):
    path = tmp_path_factory.mktemp('test_data', numbered=False)
    input = path / 'input/test_file.n5'
    output = path / 'output/test_file_new1.zarr'

    populate_n5file(input)
    return (input, output)



#test file
def populate_n5file(input):
    store = zarr.N5Store(input)
    root = zarr.group(store = store, overwrite = True) 
    paths = ['render/branch_0/data', 'render/branch_0/data1/data1_lvl1/data1_lvl2',
             'render/branch_1/data2', 'render/branch_2/data3/data3_lvl1/data3_lvl2']
    datasets = []
    for path in paths:
        n5_data = zarr.create(store=store, 
                                path=path, 
                                shape = (100,100, 100),
                                chunks=10,
                                dtype='float32', compressor=Zstd(level=4))
        n5_data[:] = 42 * np.random.rand(100,100, 100)
        datasets.append(n5_data)

    test_metadata_n5 = {"pixelResolution":{"dimensions":[4.0,4.0,4.0],
                        "unit":"nm"},
                        "ordering":"C",
                        "scales":[[1,1,1],[2,2,2],[4,4,4],[8,8,8],[16,16,16],
                                  [32,32,32],[64,64,64],[128,128,128],[256,256,256],
                                  [512,512,512],[1024,1024,1024]],
                        "axes":["z","y","x"],
                        "units":["nm","nm","nm"],
                        "translate":[-2519,-2510,1]}
    for i in range(3):
        root[f'render/branch_{i}'].attrs.update(test_metadata_n5)
        
    res_params = [(4.0, 2.0), (8.0, 0.0), (16.0, 4.0), (32.0, 8.0)]
     
    for (data, res_param) in zip(datasets, res_params):
            transform = {
            "axes": [
                "z",
                "y",
                "x"
            ],
            "ordering": "C",
            "scale": [
                res_param[0],
                res_param[0],
                res_param[0]
            ],
            "translate": [
                res_param[1],
                res_param[1],
                res_param[1]
            ],
            "units": [
                "nm",
                "nm",
                "nm"
            ]}
            data.attrs['transform'] = transform

@pytest.fixture
def n5_data(filepaths):
    populate_n5file(filepaths[0])
    store_n5 = zarr.N5Store(filepaths[0])
    n5_root = zarr.open_group(store_n5, mode = 'r')
    zarr_arrays = sorted(n5_root.arrays(recurse=True))
    return (filepaths[0], n5_root, zarr_arrays)

def test_copy_n5_tree(n5_data, filepaths):
     z_store = zarr.NestedDirectoryStore(filepaths[1])
     n5_root = n5_data[1]
     spec_n5 = pz.GroupSpec.from_zarr(n5_root) 
     spec_zarr = n5_spec_unwrapper(spec_n5)
     try:
        spec_zarr.to_zarr(z_store, path="/") 
        assert True
     except:
        assert False
     
    