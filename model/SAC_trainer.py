from torchrl.collectors import Collector
from torchrl.objectives import SACLoss
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives.utils import SoftUpdate
from torch import optim
from torchrl.trainers.algorithms import SACTrainer

#Use Pytorch Defaults
BATCH_FRAMES = 1000
LR = 3e-4
BUFFER_STORAGE = 100000
POLYAK_FACTOR = 0.995
TOTAL_FR = 1000000
FR_SKIP = 1
OPTIM_STEPS = 100



class trainer:
    def __init__(
            self, env, policy, actor_network, qvalue_network
            ):
        self._collector = Collector(env, policy, frames_per_batch=BATCH_FRAMES)
        self._loss_module = SACLoss(actor_network, qvalue_network)
        self._optimizer = optim.Adam(self._loss_module.parameters(), lr=LR)
        self._replay_buffer = ReplayBuffer(storage=LazyTensorStorage(BUFFER_STORAGE))
        self._target_net_updater = SoftUpdate(self._loss_module, eps=POLYAK_FACTOR)

# Create and run trainer
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
        )
        return trainer

    def train_agent(self):
        trainer = self._create_trainer()
        trainer.train()
