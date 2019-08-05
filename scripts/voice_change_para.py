import argparse
import glob
import multiprocessing
import re
from functools import partial
from pathlib import Path

import librosa
import numpy
import re

from yukarin import AcousticConverter
from yukarin.config import create_from_json as create_config
from yukarin.f0_converter import F0Converter
from yukarin.utility.json_utility import save_arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=Path)
parser.add_argument('--model_iteration', type=int)
parser.add_argument('--config_path', type=Path)
parser.add_argument('--input_scale', type=float, default=1.0)
parser.add_argument('--threshold', type=float)
parser.add_argument('--output_sampling_rate', type=int)
parser.add_argument('--input_statistics', type=Path)
parser.add_argument('--target_statistics', type=Path)
parser.add_argument('--output_dir', type=Path, default='./output/')
parser.add_argument('--disable_dataset_test', action='store_false')
parser.add_argument('--dataset_wave_dir', type=Path)
parser.add_argument('--test_wave_file', type=Path)
parser.add_argument('--gpu', type=int)
arguments = parser.parse_args()

model_dir: Path = arguments.model_dir
model_iteration: int = arguments.model_iteration
config_path: Path = arguments.config_path
input_scale: float = arguments.input_scale
threshold: float = arguments.threshold
output_sampling_rate: int = arguments.output_sampling_rate
input_statistics: Path = arguments.input_statistics
target_statistics: Path = arguments.target_statistics
output_dir: Path = arguments.output_dir
disable_dataset_test: bool = arguments.disable_dataset_test
dataset_wave_dir: Path = arguments.dataset_wave_dir
test_wave_file: Path = arguments.test_wave_file
gpu: int = arguments.gpu

output_dir.mkdir(exist_ok=True)


def _extract_number(f):
    s = re.findall("\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(model_dir: Path, iteration: int = None, prefix: str = 'predictor_'):
    if iteration is None:
        paths = model_dir.glob('*.npz')
        model_path = list(sorted(paths, key=_extract_number))
    else:
        fn = prefix + '{}.npz'.format(iteration)
        model_path = model_dir / fn
    return model_path


def process(p_in: Path):
    try:
        #print(f'Loaded acoustic converter model "{p_in}"')
    
        # acoustic converter
        config = create_config(config_path)
        
        # f0 converter
        if input_statistics is not None:
            f0_converter = F0Converter(input_statistics=input_statistics, target_statistics=target_statistics)
        else:
            f0_converter = None
        
        acoustic_converter = AcousticConverter(
            config,
            str(p_in.absolute()),
            gpu=gpu,
            f0_converter=f0_converter,
            out_sampling_rate=output_sampling_rate,
        )
        
        # input wave
        w_in = acoustic_converter.load_wave(test_wave_file)
        w_in.wave *= input_scale

        # input feature
        f_in = acoustic_converter.extract_acoustic_feature(w_in)
        f_in_effective, effective = acoustic_converter.separate_effective(wave=w_in, feature=f_in, threshold=threshold)

        # convert
        f_out = acoustic_converter.convert_loop(f_in_effective)
        f_out = acoustic_converter.combine_silent(effective=effective, feature=f_out)
        f_out = acoustic_converter.decode_spectrogram(f_out)

        # save
        sampling_rate = acoustic_converter.out_sampling_rate
        
        result = re.match('^\D*_(\d+)$', str(p_in.stem))
        
        if result :
            it = result.group(1)
        else :
            return
        
        frame_period = acoustic_converter.config.dataset.acoustic_param.frame_period
        wave = f_out.decode(sampling_rate=sampling_rate, frame_period=frame_period)
        outpath = str(output_dir / (test_wave_file.stem + '_{}.wav'.format(it)))
        print(outpath)
        librosa.output.write_wav(y=wave.wave, path=outpath, sr=wave.sampling_rate)
    except:
        import traceback
        traceback.print_exc()


def main():
    save_arguments(arguments, output_dir / 'arguments.json')
    
    models = _get_predictor_model_path(model_dir)
    
    paths_test = [val for val in models]
    
    process_partial = partial(process)
    if gpu is None:
        list(multiprocessing.Pool().map(process_partial, models))
    else:
        list(map(process_partial, paths_test))


if __name__ == '__main__':
    main()
