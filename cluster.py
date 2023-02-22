# Libs
import sys
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
import math
import logging
# Locals
import oggm
import oggm.cfg as cfg
from oggm import tasks, workflow, utils
from oggm.workflow import execute_entity_task
from oggm.core.flowline import equilibrium_stop_criterion, FileModel

def compile_gcm_output(gdirs, gcm_list, results):

    dir = os.path.join(cfg.PATHS['working_dir'], 'region_'+gdirs[0].rgi_region)
    utils.mkdir(dir)
    fp = os.path.join(dir, 'equilibrium_'+gdirs[0].rgi_id+'.nc')

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
    ds.coords['gcm'] = ('gcm', gcm_list)
    ds['gcm'].attrs['description'] = ' CMIP6 scenario'
    ds.coords['year'] = ('year', range(1866,2020))
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


def process_cmip6_data(path, gdirs, gcms, reset=False):
    for file in os.listdir(os.path.join(path,'tas')):
        name = file.split('_')[0]
        suffix = name.split('.')[2]
        if suffix in gcms:
            tas_file = os.path.join(path, 'tas', file)
            pr_file = os.path.join(path, 'pr', name + '_pr.nc')
            if reset:
                execute_entity_task(tasks.process_cmip_data, gdirs, filesuffix=suffix, fpath_temp=tas_file, fpath_precip=pr_file)

def equilibrium_runs_yearly(gdir, gcm_list, n_years, invert_years=False):
    logging.warning(gdir.rgi_id+' started')
    f = partial(equilibrium_stop_criterion, n_years_specmb=100, spec_mb_threshold=10)
    # maximum 2019-1866=154 years
    eq_vol = np.zeros((len(gcm_list), 154))*np.nan
    eq_area = np.zeros((len(gcm_list), 154))*np.nan
    t_array = np.zeros((len(gcm_list), 154))*np.nan

    #create dataset that merges all model_diagnostic files of this glacier
    diag_ds = xr.Dataset()

    for i, gcm in enumerate(gcm_list):
        if gcm != 'CRU':
            climate_filename='gcm_data'
            input_suffix=gcm
        else:
            climate_filename='climate_historical'
            input_suffix=None

        c = xr.open_dataset(gdir.get_filepath(climate_filename, filesuffix=input_suffix))
        years =  range(c.time.to_series().iloc[0].year + 16, c.time.to_series().iloc[-1].year - 14)
        
        if invert_years:
            years = years[::-1]
            
        for k,yr in enumerate(years):
            random.seed(yr)
            seed = random.randint(0, 2000)
            t0 = time.time()
            try:
                # in the first year, we don't use the stopping criteria to make sure, we really end up in an equilibrium state
                if k == 0:
                    mod = tasks.run_random_climate(gdir, climate_filename=climate_filename, climate_input_filesuffix=input_suffix, y0=yr,
                                             nyears=n_years, unique_samples=True, output_filesuffix=gcm + '_' + str(yr),
                                             seed=seed)
                # for all other years the previous equilibrium state as the initial condition and we use the stopping criteria
                else:
                    fp = gdir.get_filepath('model_geometry', filesuffix=gcm + '_' + str(years[k-1]))
                    fmod = FileModel(fp)
                    no_nan_yr = fmod.volume_m3_ts().dropna().index[-1]
                    fmod.run_until(no_nan_yr)
                    mod = tasks.run_random_climate(gdir, climate_filename=climate_filename, climate_input_filesuffix=input_suffix, y0=yr,
                                                   nyears=n_years, unique_samples=True, output_filesuffix=gcm + '_' + str(yr),
                                                   stop_criterion=f, seed=seed, init_model_fls=fmod.fls)
                    # if run was sucessfull, we don't need the file for init_mod any more --> remove file
                    os.remove(fp)
                j = list(range(1866,2020)).index(yr)
                eq_vol[i, j] = mod.volume_km3
                eq_area[i, j] = mod.area_km2
                t_array[i,j] = time.time()-t0
            
            except Exception as e:
                print('Failed in'+gcm+' at year '+str(yr),'with Error:'+str(e))
                break

            # read, merge and delete the current model_diagnotics file
            try:
                dp = gdir.get_filepath('model_diagnostics', filesuffix=gcm + '_' + str(yr))
                diag = xr.open_dataset(dp)
                diag = diag.expand_dims(['gcm', 'year'])
                diag.coords['gcm'] = ('gcm', [gcm])
                diag.coords['year'] = ('year', [yr])
                diag_ds = xr.merge([diag_ds, diag])
                os.remove(dp)
            except:
                pass
        logging.warning(gcm + ' done')
    # Global attributes
    diag_ds.attrs['description'] = 'OGGM model output'
    diag_ds.attrs['oggm_version'] = oggm.__version__
    diag_ds.attrs['calendar'] = '365-day no leap'
    diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    diag_ds.to_netcdf(os.path.join(gdir.dir, 'model_diagnostics_merged.nc'))

    logging.warning(gdir.rgi_id+' finished')

    return eq_vol, eq_area, t_array


if __name__ == '__main__':

    # Initialize OGGM and set up the default run parameters
    cfg.initialize()
    cfg.set_logging_config(logging_level='WARNING')
 
    REPEAT_FAILED=True

    # Local paths

    WORKING_DIR = os.environ.get("WORKDIR")
    cfg.PATHS['working_dir'] = WORKING_DIR
    OUT_DIR = os.environ.get("OUTDIR")
    REGION = str(os.environ.get('REGION')).zfill(2)
    JOB_NR = float(os.environ.get('JOB_NR'))
    cmip6_path = os.path.join(os.environ.get("PROJDIR"),'cmip6_select')

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = True

    # store model geometry
    cfg.PARAMS['store_model_geometry'] = True
    
    # climate settings
    cfg.PARAMS['climate_qc_months'] = 0
    cfg.PARAMS['baseline_climate'] = 'CRU'
    cfg.PARAMS['use_tstar_calibration'] = False  # This is new and is false per default but still
    cfg.PARAMS['use_winter_prcp_factor'] = False
    cfg.PARAMS['prcp_scaling_factor'] = 2.5  # for CRU
    cfg.PARAMS['hydro_month_nh'] = 1
    cfg.PARAMS['hydro_month_sh'] = 1
    cfg.PARAMS['min_mu_star'] = 20
    cfg.PARAMS['max_mu_star'] = 600
    
    # set border parameter
    cfg.PARAMS['border'] = 240
    # link to the preprocessed directories
    prepro_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/CRU/elev_bands/qc0/pcp2.5/match_geod_pergla/'
    
    if not REPEAT_FAILED:
        
        # RGI file
        path = utils.get_rgi_region_file(REGION, version='61')
        rgidf = gpd.read_file(path)
        rgidf = rgidf.sort_values('Area', ascending=True)

        # exclude non-landterminating glaciers
        rgidf = rgidf[rgidf.TermType == 0]
        rgidf = rgidf[rgidf.Connect != 2]  

        # path to the statistic file
        url = os.path.join(prepro_url, 'RGI62/b_'+str(cfg.PARAMS['border'])+'/L5/summary/')

        # exculde glaciers that failed during preprocessing
        fpath = utils.file_downloader(url + f'glacier_statistics_{REGION}.csv')
        stat = pd.read_csv(fpath, index_col=0, low_memory=False)
        rgidf = rgidf[~rgidf.RGIId.isin(stat.error_task.dropna().index)].reset_index()
        
        #select glacier by JOB_NR
        rgi_id = rgidf.iloc[[JOB_NR]].index
        gcm_list = ['CRU', 'CanESM5', 'NorESM2-MM', 'FGOALS-f3-L', 'GISS-E2-2-H', 'BCC-CSM2-MR', 'MRI-ESM2-0',
                    'E3SM-1-1', 'CESM2', 'MPI-ESM1-2-HR', 'ACCESS-CM2', 'EC-Earth3', 'IPSL-CM6A-LR-INCA', 'MIROC6']
    else:
        # read file with failed glaciers
        failed = pd.read_csv('../run_reverse.txt',index_col=0)
        #filter for THIS region
        failed = failed[failed.index.str.startswith('RGI60-'+REGION)]
        # select glacier by JOB_NR
        rgi_id = failed.iloc[[JOB_NR]].index
        gcm_list = failed[failed>0].iloc[[JOB_NR]].dropna(axis=1).columns.to_numpy()

    # Go - initialize glacier directories
    gdirs = workflow.init_glacier_regions(rgi_id, from_prepro_level=5, prepro_base_url=prepro_url)

    # process cmip6 data
    process_cmip6_data(cmip6_path, gdirs, gcms=gcm_list, reset=True)

    n_years = 2000
    if REGION in ['01', '03', '04', '05', '06', '07', '09', '17']:
        n_years = 5000
    res = execute_entity_task(equilibrium_runs_yearly, gdirs, gcm_list=gcm_list, n_years=n_years, invert_years=True)
    ds = compile_gcm_output(gdirs, gcm_list, res)

