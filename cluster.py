# Libs
import os
import matplotlib.pyplot as plt
import xarray as xr
import time
import pandas as pd
import numpy as np
import random
import geopandas as gpd
from functools import partial
from time import gmtime, strftime
import logging
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
    ds['equilibrium'] = (('rgi_id', 'gcm', 'year'), [res[0] for res in results])
    ds['equilibrium'].attrs['description'] = 'total glacier volume of equilibrium glacier'
    ds['equilibrium'].attrs['units'] = 'km 3'

    ds['equilibrium_area'] = (('rgi_id', 'gcm', 'year'), [res[1] for res in results])
    ds['equilibrium_area'].attrs['description'] = 'total glacier area of equilibrium glaciers'
    ds['equilibrium_area'].attrs['units'] = 'km 2'

    ds['run_time'] = (('rgi_id', 'gcm', 'year'), [res[2] for res in results])
    ds['run_time'].attrs['description'] = 'total runtime for glacier'
    ds['run_time'].attrs['units'] = 'sec.'

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
    eq_vol = np.zeros((len(gcm_list), len(years)))
    eq_area = np.zeros((len(gcm_list), len(years)))
    t_array = np.zeros((len(gcm_list), len(years)))
    for i, gcm in enumerate(gcm_list):
        for j, yr in enumerate(years):
            random.seed(yr)
            seed = random.randint(0, 2000)
            t0 = time.time()
            if yr == 1866:
                mod = tasks.run_random_climate(gdir, climate_filename='gcm_data', climate_input_filesuffix=gcm, y0=yr,
                                         nyears=2000, unique_samples=True, output_filesuffix=gcm + '_' + str(yr),
                                         seed=seed)
            else:
                fp = gdir.get_filepath('model_geometry', filesuffix=gcm + '_' + str(yr - 1))
                fmod = FileModel(fp)
                no_nan_yr = fmod.volume_m3_ts().dropna().index[-1]
                fmod.run_until(no_nan_yr)
                mod = tasks.run_random_climate(gdir, climate_filename='gcm_data', climate_input_filesuffix=gcm, y0=yr,
                                               nyears=2000, unique_samples=True, output_filesuffix=gcm + '_' + str(yr),
                                               stop_criterion=f, seed=seed, init_model_fls=fmod.fls)
            eq_vol[i, j] = mod.volume_km3
            eq_area[i, j] = mod.area_km2
            t_array[i,j] = time.time()-t0
    logging.warning(gdir.rgi_id+' finished')

    return eq_vol, eq_area, t_array

if __name__ == '__main__':

    # Initialize OGGM and set up the default run parameters
    cfg.initialize()
    cfg.set_logging_config(logging_level='WARNING')
    ON_CLUSTER = True

    # Local paths
    if ON_CLUSTER:
        WORKING_DIR = os.environ.get("WORKDIR")
        cfg.PATHS['working_dir'] = WORKING_DIR
        OUT_DIR = os.environ.get("OUTDIR")
        REGION = str(os.environ.get('REGION')).zfill(2)
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

    # RGI file
    path = utils.get_rgi_region_file(REGION, version='61')
    rgidf = gpd.read_file(path)
    rgidf = rgidf.sort_values('Area', ascending=False)

    # exclude non-landterminating glaciers
    rgidf = rgidf[rgidf.TermType == 0]
    rgidf = rgidf[rgidf.Connect != 2]

    # select classes by area to make a check for the running time
    cat1 = pd.cut(rgidf.Area[rgidf.Area > 1], bins=10, labels=range(10)).astype('float')
    cat2 = pd.cut(rgidf.Area[rgidf.Area < 1], bins=30, labels=range(10, 40)).astype('float')
    rgidf.loc[cat1.index, 'category'] = cat1
    rgidf.loc[cat2.index, 'category'] = cat2

    rgidf = rgidf.drop_duplicates(subset='category', keep='last')

    # Go - initialize glacier directories
    gdirs = workflow.init_glacier_regions(rgidf, from_prepro_level=3, reset=False)
    #gdirs = workflow.init_glacier_regions()

    # read (reset=False) or process cmip6 data (reset=True)
    gcm_list = read_cmip6_data(cmip6_path, gdirs, reset=True)
    years = range(1866, 1999)

    res = execute_entity_task(equilibrium_runs_yearly, gdirs, gcm_list=gcm_list, years=years)
    ds = compile_gcm_output(gdirs, gcm_list,years, res)

