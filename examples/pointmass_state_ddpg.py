import shutil
from functools import partial
import dm2gym
import dm_control

from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
# from rlpyt.envs.gym import pixel_make as gym_make
from rlpyt.envs.gym import make, gymlike_make as gym_make, gymlike_make
from rlpyt.algos.qpg.ddpg import DDPG
from rlpyt.agents.qpg.ddpg_agent import DdpgAgent
from rlpyt.algos.qpg.sac import SAC
# from rlpyt.algos.qpg.sac_agent import SacAgent
from rlpyt.models.qpg.conv2d import MuConv2dModel, QofMuConv2dModel

from rlpyt.models.qpg.mlp import MuMlpModel, QofMuMlpModel


# from rlpyt.agents.qpg.image_ddpg_agent import ImageDdpgAgent
# from rlpyt.models.qpg.conv2d import MuConv2dModel, QofMuConv2dModel

from rlpyt.utils.logging.context import logger_context


from sacred import Experiment
from sacred.observers import FileStorageObserver
from rlpyt.utils.logging.sacred_observers import FileStorageObserverWithExUuid

LOG_DIR = 'pointmass'

ex = Experiment("ddpg_PointMassImage")

@ex.config
def base_cfg():
    uuid = 'ddpg_agent'
    cfg = dict(
        sampler_fn = 'CpuSampler',
        workers_cpus = [1],
    )
    """DDPG changes according to Lillicrap paper"""
    env_id = "dm2gym:Point_massEasy-v0"

    sampler_kwargs = dict(
#         batch_T = 50,
        batch_T = 50,
        #batch_B = 16,
        batch_B = 16,
        max_decorrelation_steps = 0,
        eval_n_envs = 3,
        eval_max_steps = int(1e53),
        eval_max_trajectories = 10,
    )

    runner_kwargs = dict(
        n_steps = 1e6,
        log_interval_steps = 1e4,
    )
    run_id = 1
    cuda_idx = None
    snapshot_mode = "last"
    name = "ddpg_" + env_id
    algo_kwargs = dict(
        batch_size = 512,
        target_update_tau=0.001,
        replay_ratio=32,
        )
    
    q_model_kwargs = dict(
            #observation_shape=(4,),
            hidden_sizes=[200],
            #action_latent_size=64,
            #action_size=2,
        )
    mu_model_kwargs = dict(
            #observation_shape=(4,),
            hidden_sizes=[200],
            #action_size=2
        )

    v_model_kwargs = dict(
            img_channels=3,
            latent_size=200,
            action_size=2,
        )
            # model_kwargs = dict(fc_sizes = [200,200], channels = [16,32,32], kernel_sizes=[8,4,4], strides=[4,2,2], paddings=[0,0,0])
    agent_kwargs = dict(
        QModelCls=QofMuMlpModel,
        ModelCls=MuMlpModel,
        model_kwargs=mu_model_kwargs,
        q_model_kwargs=q_model_kwargs
    )

    # agent_kwargs = dict(
    #     QModelCls=QofMuConv2dModel,
    #     ModelCls=MuConv2dModel,
    #     model_kwargs=mu_model_kwargs,
    #     q_model_kwargs=q_model_kwargs
    # )

@ex.main
def build_and_train(cfg, env_id, sampler_kwargs, runner_kwargs, uuid, cuda_idx, snapshot_mode, name, algo_kwargs, agent_kwargs, seed=None):
    sampler = eval(cfg['sampler_fn'])(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        **sampler_kwargs
    )

    # algo = SAC(**agent_kwargs)
    # agent = SacAgent(**agent_kwargs)
    algo = DDPG(**algo_kwargs)
    # agent = ImageDdpgAgent(**agent_kwargs)
    agent = DdpgAgent(**agent_kwargs)
    print(cfg)
    if cfg['sampler_fn'] == 'CpuSampler':
        affinity = dict(workers_cpus=cfg['workers_cpus'])
    else:
        print("serialsampler")
        affinity = dict(cuda_idx=cuda_idx)

    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        **runner_kwargs,
        affinity=affinity,
        seed=seed
    )

    config = dict(env_id=env_id)

    with logger_context(LOG_DIR, uuid, name, config, snapshot_mode):
        runner.train()


if __name__ == '__main__':
    assert LOG_DIR, 'log_dir cannot be empty'
    print(f"Found {LOG_DIR}--removing {LOG_DIR}...")
    shutil.rmtree(LOG_DIR)
    ex.observers.append(FileStorageObserverWithExUuid(LOG_DIR))
    ex.run_commandline()