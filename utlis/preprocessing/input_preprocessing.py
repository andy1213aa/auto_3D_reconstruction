import os
from absl import logging
import shutil


def input_preprocessing(cfg):

    if os.path.isdir(cfg.input):
        shutil.copytree(cfg.input, f'./{cfg.tmp}/frames')
        return True
    elif os.path.isfile(cfg.input):
        video_extensions = ['.mp4', '.avi', '.mkv']

        if any(cfg.input.lower().endswith(ext) for ext in video_extensions):

            #RVM remove background
            rvm_cmd = f'python RobustVideoMatting/inference.py \
                --variant resnet50 \
                --checkpoint "RobustVideoMatting/rvm_resnet50.pth" \
                --device cuda \
                --input-source "{cfg.input}" \
                --downsample-ㄙㄟratio 0.25 \
                --output-type video \
                --output-composition "./{cfg.tmp}/input_preprocessing.mp4" \
                --output-video-mbps 21 \
                --seq-chunk 12'

            os.system(rvm_cmd)

            # ffmpeg split video into frames
            ffmpeg_cmd = f'ffmpeg -i "./{cfg.tmp}/input_preprocessing.mp4" \
                -vf "transpose=1,fps={cfg.split_fps}" \
                {cfg.split_output_folder}/frame_%04d.png'

            os.system(ffmpeg_cmd)

            return True
        else:
            logging.info(
                'The file path is not: 1) folder of images, 2) video in mp4/avi/mkv.'
            )
            return False
    else:
        logging.info('The path is not exist')
        return False
