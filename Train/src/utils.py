import subprocess
from pathlib import Path
import logging
import os
import sys
import shutil
import yaml
import socket
from shlex import quote
from datetime import datetime
from glob import glob
import torch
import os
import random
from tensorboardX import SummaryWriter
from einops import repeat
from contextlib import contextmanager
import time
import yacs
from yacs.config import CfgNode as CN
import habitat
import numpy as np
import torch
import clip
import PIL
import lmdb
from torchvision import transforms
from habitat_baselines.config.default import get_config as get_habitat_config

logger = logging.getLogger(__name__)

from habitat.tasks.nav.instance_image_nav_task import (
    InstanceImageGoalSensor
)
from src.sensor import (
    ImageGoalSensorV2,
    GibsonImageGoalFeatureSensor,
    QueriedImageSensor
)
from habitat.config import Config


def load_config(config_path):
    conf = CN()
    # Task need to be RandomSample/TrainVQVAE/TrainWorldModel
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False

    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.DemonstrationBatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineDemonstrationBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.UseDemonstration = False

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf


def compare_config(cc, pc):
    def flat_map(src, target=None, prefix=""):
        if target is None:
            target = {}
        for k, v in src.items():
            if type(v) is dict:
                flat_map(v, target, prefix + k + ".")
            else:
                target[prefix + k] = v
        return target

    # load previous config and find unmatch config
    cc = yaml.load(str(cc), Loader=yaml.FullLoader)
    pc = yaml.load(str(pc), Loader=yaml.FullLoader)
    unmatch_config = {k: v for k, v in flat_map(pc).items() if flat_map(cc)[k] != v}
    return unmatch_config


def save_sh(run_dir, run_type):
    with open(run_dir / 'code/run_{}_{}.sh'.format(run_type, socket.gethostname()), 'w') as f:
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('mkdir -p {}\n'.format(run_dir / 'code/unpack'))
        f.write('tar -C {} -xzvf {}\n'.format(run_dir / 'code/unpack', run_dir / 'code/code_*.tar.gz'))
        f.write('cd {}\n'.format(run_dir / 'code/unpack'))
        f.write('patch -p1 < ../dirty.patch\n')
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('cp -r -f {} {}\n'.format(run_dir / 'code/unpack/*', quote(os.getcwd())))
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')


def save_config(run_dir, config, type):
    if config is not None:
        F = open(run_dir / 'config_of_{}.yaml'.format(type), 'w')
        F.write(str(config))
        F.close()


def get_random_rundir(exp_dir: Path, prefix: str = 'run', suffix: str = ''):
    if exp_dir.exists():
        runs = glob(str(exp_dir / '*_run*'))
        num_runs = len([r for r in runs if (prefix + '_') in r])
    else:
        num_runs = 0
    dt = datetime.now().strftime('%Y%m%d-%H%M%S')
    rundir = prefix + '_' + 'run{}'.format(num_runs) + '_' + suffix
    return (exp_dir / rundir)


def pack_code(run_dir: str):
    run_dir = Path(run_dir) / 'code'
    if not run_dir.exists():
        run_dir.mkdir()
    if os.path.isdir(".git"):
        HEAD_commit_id = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True
        )
        tar_name = f'code_{HEAD_commit_id.stdout[:-1]}.tar.gz'
        subprocess.run(
            ['git', 'archive', '-o', str(run_dir / tar_name), 'HEAD'],
            check=True,
        )
        diff_process = subprocess.run(
            ['git', 'diff', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True,
        )
        if diff_process.stdout:
            logger.warning('Working tree is dirty. Patch:\n%s', diff_process.stdout)
            with (run_dir / 'dirty.patch').open('w') as f:
                f.write(diff_process.stdout)
    else:
        logger.warning('.git does not exist in current dir')


def change_habitat_config(config):
    with habitat.config.read_write(config):
        if ImageGoalSensorV2.cls_uuid in config.habitat.task.sensors:
            goalsensoruuid = ImageGoalSensorV2.cls_uuid
        elif QueriedImageSensor.cls_uuid in config.habitat.task.sensors:
            goalsensoruuid = QueriedImageSensor.cls_uuid
        elif InstanceImageGoalSensor.cls_uuid + "_sensor" in config.habitat.task.sensors:
            goalsensoruuid = InstanceImageGoalSensor.cls_uuid
        else:
            assert False, "Do not specifit goal sensor"

        config.habitat.task.distance_to_view = Config()
        config.habitat.task.distance_to_view.type = "DistanceToView"
        config.habitat.task.distance_to_view.goalsensoruuid = goalsensoruuid

        config.habitat.task.view_match = Config()
        config.habitat.task.view_match.type = "ViewMatch"
        config.habitat.task.view_match.view_weight = 0.5
        config.habitat.task.view_match.angle_threshold = 25.0
        config.habitat.task.view_match.goalsensoruuid = goalsensoruuid

        config.habitat.task.view_angle = Config()
        config.habitat.task.view_angle.type = "ViewAngle"
        config.habitat.task.view_angle.goalsensoruuid = goalsensoruuid
    return config


def get_config(exp_config, opts, run_type, model_dir, overwrite, note, debug, global_rank):
    exp_config = exp_config.split(',')
    exp_config = [path if '.yaml' in path else f'src/config/{path}.yaml' for path in exp_config]

    config = get_habitat_config(exp_config, opts)
    config = change_habitat_config(config)
    config.defrost()

    if model_dir == None:
        model_dir = 'results/official'

    config.habitat_baselines.checkpoint_folder = os.path.join(model_dir, 'ckpts')
    if 'habitat_baselines.eval_ckpt_path_dir' not in opts:
        config.habitat_baselines.eval_ckpt_path_dir = os.path.join(model_dir, 'ckpts')
    config.habitat_baselines.tensorboard_dir = os.path.join(model_dir, 'tb')
    config.habitat_baselines.video_dir = os.path.join(model_dir, "video")

    if debug:
        ds_type = config.habitat.environment.type
        if ds_type == "gibson":
            scene = getattr(config.habitat, 'debug_scenes', ['Adrian'])
        elif ds_type == "mp3d":
            scene = getattr(config.habitat, 'debug_scenes', ['pRbA3pwrgk9'])
        elif ds_type == "hm3d":
            scene = getattr(config.habitat, 'debug_scenes', ['00001-UVdNNRcVyV1'])
        else:
            scene = getattr(config.habitat, 'debug_scenes', ['NRsmXFcVTbN'])

        logger.warning('Debug using 1 scene!')
        config.habitat.dataset.content_scenes = scene
        config.habitat_baselines.log_interval = 1

    if global_rank == 0:
        if overwrite:
            logger.warning('Warning! overwrite is specified!\nCurrent model dir will be removed!')
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=True)

        run_dir = get_random_rundir(exp_dir=Path(model_dir), prefix=run_type, suffix=note)

        config.habitat_baselines.log_file = os.path.join(run_dir, 'log.txt')
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(config.habitat_baselines.tensorboard_dir, exist_ok=True)
        os.makedirs(config.habitat_baselines.checkpoint_folder, exist_ok=True)

        pack_code(run_dir)
        save_sh(run_dir, run_type)
        save_config(run_dir, config, run_type)

    config.freeze()
    return config


def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])


class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    # Task need to be RandomSample/TrainVQVAE/TrainWorldModel
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False

    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.DemonstrationBatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineDemonstrationBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.UseDemonstration = False

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf
