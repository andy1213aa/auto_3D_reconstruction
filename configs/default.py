"""Default Hyperparameter configuration"""

import ml_collections


def get_config():
    """Get the default Hypermarameter configuration"""
    config = ml_collections.ConfigDict()
    # System setting (No need to change)
    config.tmp = 'tmp'  # don't change cause Wrap app is fixed with file name.
    config.reconstruction_feature_path = f'{config.tmp}/feature.txt'  #no need to change

    #Process tmp folder
    config.images_folder = '/media/aaron/work/ITRI_SSTC/S100/FY112_FRP/code/FRP/2Dto3D/E041'
    config.output_folder = '/home/aaron/Desktop/test'

    # Meshroom
    config.meshroom_batch = '/home/aaron/Downloads/app/Meshroom-2021.1.0-av2.4.0-centos7-cuda10.2/meshroom_batch'
    config.meshroom_input = config.images_folder
    config.meshroom_pipline = 'photogrammetry'

    #Tracing
    config.trace_vis = False

    # R3DS
    config.canonical_feature_path = './4lips_REYE_REyebrow_LEYE_LEyebrow.txt'
    config.wrap_cmd = '/home/aaron/Downloads/app/R3DS_Wrap_3.4.8_Linux/Wrap3Cmd'
    config.wrap_default = './default.wrap'
    config.wrap_output = f'{config.output_folder}/Wrap'

    #General
    config.keep_intermediate_result = True

    return config