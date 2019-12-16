from functools import partial
import dm2gym
import dm_control

from dm_control import suite
for domain_name, task_name in suite.ALL_TASKS:
    ID = f'{domain_name.capitalize()}{task_name.capitalize()}-v0'
    print(domain_name, task_name, ID)

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

# from rlpyt.agents.qpg.image_ddpg_agent import ImageDdpgAgent
# from rlpyt.models.qpg.conv2d import MuConv2dModel, QofMuConv2dModel

from rlpyt.utils.logging.context import logger_context


            # self,
            # in_channels,
            # channels,
            # kernel_sizes,
            # strides,
            # paddings=None,
            # nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            # use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            # head_sizes=None,  # Put an MLP head on top.



from sacred import Experiment
from sacred.observers import FileStorageObserver

LOG_DIR = 'pointmass'

ex = Experiment("ddpg_PointMassImage")
ex.observers.append(FileStorageObserver(LOG_DIR))

@ex.config
def base_cfg():
    cfg = dict(
        sampler_fn = 'CpuSampler',
        workers_cpus = [1],
    )
    """DDPG changes according to Lillicrap paper"""
    env_id = "dm2gym:Point_massEasy-v0"

    sampler_kwargs = dict(
        batch_T = 50,
        batch_B = 1,
        max_decorrelation_steps = 0,
        eval_n_envs = 3,
        eval_max_steps = int(1e53),
        eval_max_trajectories = 50,
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
        batch_size = 256,
        target_update_tau=0.001,
        replay_ratio=32,
        )
    q_model_kwargs = dict(
            img_channels=3,
            latent_size=200,
            action_latent_size=64,
            # action_size=4,
        )
    mu_model_kwargs = dict(
            img_channels=3,
            latent_size=200,
            # action_size=4
        )

    v_model_kwargs = dict(
            img_channels=3,
            latent_size=200,
            # action_size=4
        )
            # model_kwargs = dict(fc_sizes = [200,200], channels = [16,32,32], kernel_sizes=[8,4,4], strides=[4,2,2], paddings=[0,0,0])
    agent_kwargs = dict(
        QModelCls=QofMuConv2dModel,
        ModelCls=MuConv2dModel,
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
def build_and_train(cfg, env_id, sampler_kwargs, runner_kwargs, run_id, cuda_idx, snapshot_mode, name, algo_kwargs, agent_kwargs, seed=None):
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
    log_dir = "experiments"

    with logger_context(log_dir, run_id, name, config, snapshot_mode):
        runner.train()

if __name__ == '__main__':
    ex.run_commandline()