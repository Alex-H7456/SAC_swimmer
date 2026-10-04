from torchrl.collectors import Collector
from torchrl.objectives import SACLoss
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives.utils import SoftUpdate
from torch import optim
from torchrl.record.loggers import CSVLogger
from torchrl.trainers.algorithms import SACTrainer
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
            self, env, actor, qvalue_network
            ):
        self._collector = Collector(env, actor, frames_per_batch=BATCH_FRAMES)
        self._loss_module = SACLoss(actor, qvalue_network)
        self._optimizer = optim.Adam(self._loss_module.parameters(), lr=LR)
        self._replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(BUFFER_STORAGE),
            batch_size=BATCH_SIZE,
        )
        self._target_net_updater = SoftUpdate(self._loss_module, eps=POLYAK_FACTOR)
        self._logger = CSVLogger(
            exp_name="sac_training",
            log_dir=LOG_DIR,
        )
        self._trainer = self._create_trainer()

    def _create_trainer(self):
        trainer = SACTrainer(
            collector=self._collector,
            total_frames=TOTAL_FR,
            frame_skip=FR_SKIP,
            optim_steps_per_batch=OPTIM_STEPS,
            loss_module=self._loss_module,
            optimizer=self._optimizer,
            replay_buffer=self._replay_buffer,
            target_net_updater=self._target_net_updater,
            logger=self._logger,
            enable_logging=True,
            progress_bar=True,
            episode_reward_key="reward_sum",
            log_interval=BATCH_FRAMES - 1,
        )
        return trainer

    def train(self):
        logger.info(f"Starting SAC training for {TOTAL_FR:,} frames.")
        self._trainer.train()
        logger.info("SAC training complete.")
