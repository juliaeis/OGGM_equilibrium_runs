import multiprocessing
import xarray as xr
import os

def read_single_glacier(path):
    ds = xr.open_dataset(path)
    return ds

if __name__ == '__main__':

    REGION = str(os.environ.get('REGION')).zfill(2)
    WORKING_DIR = os.environ.get("WORKDIR")
    READING_DIR = os.environ.get("READDIR")

    read_dir = os.path.join(READING_DIR, 'region_'+REGION)
    files = [os.path.join(read_dir, file) for file in os.listdir(read_dir) if not file.endswith('m.nc')]

    p = multiprocessing.Pool()
    result = p.map(read_single_glacier, files)
    ds = xr.merge(result)

    ds.to_netcdf(os.path.join(WORKING_DIR, REGION+'_equilibrium.nc'))