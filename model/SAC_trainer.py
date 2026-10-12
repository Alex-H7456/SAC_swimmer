from torchrl.collectors import Collector
from torchrl.objectives import SACLoss
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives.utils import SoftUpdate
from torch import optim
from torchrl.record.loggers import CSVLogger
from torchrl.trainers.algorithms import SACTrainer
from tensordict import TensorDictBase
import logging
#Use Pytorch Defaults
BATCH_FRAMES = 1000
LR = 3e-4
BUFFER_STORAGE = 100000
POLYAK_FACTOR = 0.995
TOTAL_FR = 1000000
FR_SKIP = 1
OPTIM_STEPS = 100
BATCH_SIZE = 256
LOG_DIR = "logs"

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
            self, env, actor, qvalue_network, 
            total_frames=TOTAL_FR,
            batch_frames=BATCH_FRAMES, 
            optim_steps=OPTIM_STEPS,
            lr = LR,
            polyak_factor = POLYAK_FACTOR,
            fr_skip = FR_SKIP,
            log_dir = LOG_DIR,
            batch_size = BATCH_SIZE,
            episode_return_callback=None
            ):
        self._episode_return_callback = episode_return_callback
        self._optim_steps = optim_steps
        self._total_frames = total_frames
        self._batch_frames = batch_frames
        self.fr_skip = fr_skip
        self._collector = Collector(
            env, actor, frames_per_batch=batch_frames, total_frames=total_frames
        )
        self._loss_module = SACLoss(actor, qvalue_network)
        self._optimizer = optim.Adam(self._loss_module.parameters(), lr=lr)
        self._replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(BUFFER_STORAGE),
            batch_size=batch_size,
        )
        self._target_net_updater = SoftUpdate(self._loss_module, eps=polyak_factor)
        self._logger = CSVLogger(
            exp_name="sac_training",
            log_dir=log_dir,
        )
        self._trainer = self._create_trainer()

    def _create_trainer(self):
        trainer = SACTrainer(
            collector=self._collector,
            total_frames=self._total_frames,
            frame_skip=self.fr_skip,
            optim_steps_per_batch=self._optim_steps,
            loss_module=self._loss_module,
            optimizer=self._optimizer,
            replay_buffer=self._replay_buffer,
            target_net_updater=self._target_net_updater,
            logger=self._logger,
            enable_logging=True,
            progress_bar=True,
            episode_reward_key="reward_sum",
            log_interval=self._batch_frames - 1,
        )
        if self._episode_return_callback is not None:
            trainer.register_hook("batch_process", self._capture_episode_returns)
        return trainer

    def _capture_episode_returns(self, batch: TensorDictBase) -> TensorDictBase:
        done = batch["next", "done"].squeeze(-1)
        returns = batch["next", "reward_sum"].squeeze(-1)[done]
        for value in returns.detach().cpu().tolist():
            self._episode_return_callback(float(value))
        return batch

    def train(self):
        logger.info(f"Starting SAC training for {self._total_frames:,} frames.")
        self._trainer.train()
        logger.info("SAC training complete.")
