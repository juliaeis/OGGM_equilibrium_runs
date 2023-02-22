
import xarray as xr
import os


if __name__ == '__main__':

    WORKING_DIR = os.environ.get("WORKDIR")
    READING_DIR = os.environ.get("READDIR")
    REGION = str(os.environ.get('REGION')).zfill(2)

    read_dir = os.path.join(READING_DIR, 'region_'+REGION,'*.nc')
    print('start region '+REGION)
    ds = xr.open_mfdataset(read_dir, parallel=True)
    ds.to_netcdf(os.path.join(WORKING_DIR, REGION+'_equilibrium.nc'))
    print('region '+REGION+' DONE')
    
