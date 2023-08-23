from absl import logging
from absl import flags
from absl import app
from ml_collections import config_flags
import os
from utlis.parsing_json import parsing_SFM_json
from utlis.post_process import process_feature_txt, export_r3ds_format
from ray_tracing.main import main as ray_trace
import shutil

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the hyperparameter configuration.',
    lock_config=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    cfg = FLAGS.config

    #Clear tmp folder
    if os.path.exists(f'./{cfg.tmp}'):
        shutil.rmtree(f'./{cfg.tmp}')

    # Meshroom Dense Reconstruction
    meshroom_command = f'{cfg.meshroom_batch} -i {cfg.meshroom_input} -p {cfg.meshroom_pipline} -o {os.getcwd()}/{cfg.tmp} --cache {os.getcwd()}/{cfg.tmp}/MeshroomCache'
    os.system(meshroom_command)

    # Tracing Algorithm
    views = parsing_SFM_json(cfg)
    for view in views:

        res, intersect_flag = ray_trace(
            width=view.width,
            height=view.height,
            R=view.R,
            t=view.t,
            focal=view.focal,
            princpt=view.princpt,
            pixel_size_mm=view.px_size,
            obj_pth=f'{cfg.tmp}/texturedMesh.obj',
            img_pth=view.img_pth,
            render_type='feature_marking',
            visualize=cfg.trace_vis,
        )
        if intersect_flag.count(True) <= 0.95 * len(intersect_flag):
            continue
        #Post-process feature file to make sure the length is same.
        shutil.copy(cfg.canonical_feature_path,
                    f'./{cfg.tmp}/{cfg.canonical_feature_path}')
        export_r3ds_format(res, cfg.reconstruction_feature_path)
        process_feature_txt(c_path=f'./{cfg.tmp}/{cfg.canonical_feature_path}',
                            flag=intersect_flag)
        break

    # R3DS Wrap
    r3ds_command = f'{cfg.wrap_cmd} compute {cfg.wrap_default}'
    os.system(r3ds_command)

    #Copy wrap output from tmp
    if not os.path.exists(cfg.wrap_output):
        os.mkdir(cfg.wrap_output)

    shutil.copy(f'./{cfg.tmp}/tmp.obj', f'{cfg.wrap_output}/output.obj')
    shutil.copy(f'./{cfg.tmp}/tmp.jpg', f'{cfg.wrap_output}/output.jpg')

    #If need median result
    if cfg.keep_intermediate_result:
        shutil.copytree(f'./{cfg.tmp}',
                        f'{cfg.output_folder}/intermediate_result')

    shutil.rmtree(f'./{cfg.tmp}')


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    app.run(main)