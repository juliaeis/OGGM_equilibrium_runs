if __name__ == '__main__':

    # Initialize OGGM and set up the default run parameters
    cfg.initialize()

    ON_CLUSTER = True

    # Local paths
    if ON_CLUSTER:
        WORKING_DIR = os.environ.get("S_WORKDIR")
        cfg.PATHS['working_dir'] = WORKING_DIR
        OUT_DIR = os.environ.get("OUTDIR")
        REGION = str(os.environ.get('REGION')).zfill(2)
        print('REGION')
    else:
        cfg.PATHS['working_dir'] = os.path.join('run_CMIP6')

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

    print(gdirs)