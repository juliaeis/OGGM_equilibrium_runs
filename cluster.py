# Libs
import os
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import random
from functools import partial
from time import gmtime, strftime
# Locals
import oggm
import oggm.cfg as cfg
from oggm import tasks, workflow, utils
from oggm.workflow import execute_entity_task
from oggm.core.flowline import equilibrium_stop_criterion, FileModel

def compile_gcm_output(gdirs, list, years, results):

    fp = os.path.join(cfg.PATHS['working_dir'], gdirs[0].rgi_region + '_equilibrium.nc')
    if os.path.exists(fp): os.remove(fp)

    ds = xr.Dataset()

    # Global attributes
    ds.attrs['description'] = 'OGGM model output'
    ds.attrs['oggm_version'] = oggm.__version__
    ds.attrs['calendar'] = '365-day no leap'
    ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # Coordinates
    ds.coords['rgi_id'] = ('rgi_id', [gd.rgi_id for gd in gdirs])
    ds['rgi_id'].attrs['description'] = 'RGI glacier identifier'
    ds.coords['gcm'] = ('gcm', list)
    ds['gcm'].attrs['description'] = ' CMIP6 scenario'
    ds.coords['year'] = ('year', years)
    ds['year'].attrs['description'] = 'central year of random climate used to create the equilibrium glacier'

    # Variables
    ds['equilibrium'] = (('rgi_id', 'gcm', 'year'), results)
    ds['equilibrium'].attrs['description'] = 'total glacier volume of equilibrium glacier'
    ds['equilibrium'].attrs['units'] = 'km 3'

    ds.to_netcdf(fp)
    return ds


def read_cmip6_data(path, gdirs, reset=False):
    l = []
    for file in os.listdir(os.path.join(path,'tas')):
        if file.endswith('.nc'):
            name = file.split('_')[0]
            suffix = name.split('.')[2]
            l.append(suffix)
            tas_file = os.path.join(path, 'tas', file)
            pr_file = os.path.join(path, 'pr', name + '_pr.nc')
            if reset:
                execute_entity_task(tasks.process_cmip_data, gdirs, filesuffix=suffix, fpath_temp=tas_file,
                                    fpath_precip=pr_file)
    return l

def equilibrium_runs_yearly(gdir, gcm_list,years):
    f = partial(equilibrium_stop_criterion, n_years_specmb=100, spec_mb_threshold=10)
    eq = np.zeros((len(gcm_list),len(years)))

    for i, gcm in enumerate(gcm_list):
        for j,yr in enumerate(years):
            random.seed(yr)
            seed = random.randint(0, 2000)
            if yr == 1866:
                tasks.run_random_climate(gdir, climate_filename='gcm_data', climate_input_filesuffix=gcm, y0=yr,
                                         nyears=2000, unique_samples=True, output_filesuffix=gcm+'_'+str(yr), seed=seed)
            else:
                fp = gdir.get_filepath('model_geometry', filesuffix=gcm+'_'+str(yr-1))
                fmod = FileModel(fp)
                no_nan_yr = fmod.volume_m3_ts().dropna().index[-1]
                fmod.run_until(no_nan_yr)
                print(fmod)
                mod = tasks.run_random_climate(gdir, climate_filename='gcm_data', climate_input_filesuffix=gcm, y0=1866,
                                         nyears=2000, unique_samples=True, output_filesuffix=gcm+'_'+str(yr),
                                         stop_criterion=f, seed=seed, init_model_fls = fmod.fls)
                eq[i, j] = mod.volume_km3
    return eq

if __name__ == '__main__':

    # Initialize OGGM and set up the default run parameters
    cfg.initialize()

    ON_CLUSTER = True

    # Local paths
    if ON_CLUSTER:
        WORKING_DIR = os.environ.get("WORKDIR")
        cfg.PATHS['working_dir'] = WORKING_DIR
        OUT_DIR = os.environ.get("OUTDIR")
        REGION = str(os.environ.get('REGION')).zfill(2)
        print(REGION)
        cmip6_path = os.path.join(os.environ.get("PROJDIR"),'cmip6')
    else:
        cfg.PATHS['working_dir'] = os.path.join('run_CMIP6')
        cmip6_path = 'cmip6'

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = True

    # store model geometry
    cfg.PARAMS['store_model_geometry'] = True

    # How many grid points around the glacier?
    # Make it large if you expect your glaciers to grow large
    cfg.PARAMS['border'] = 160

    # Go - initialize glacier directories
    gdirs = workflow.init_glacier_regions(['RGI60-11.00897','RGI60-11.00779'], from_prepro_level=3, reset=False)
    #gdirs = workflow.init_glacier_regions()

    # read (reset=False) or process cmip6 data (reset=True)
    gcm_list = read_cmip6_data(cmip6_path, gdirs, reset=True)
    years = range(1866, 1999)

    res = execute_entity_task(equilibrium_runs_yearly, gdirs, gcm_list=gcm_list, years=years)
    ds = compile_gcm_output(gdirs, gcm_list,years, res)

