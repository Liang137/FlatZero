import torch
import numpy as np
import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExpCallback(TrainerCallback):

    def __init__(self, trainer, test_dataset):
        super().__init__()
        self.trainer = trainer
        self.every_steps = 100
        self.num_samples = 100
        self.delta = 1e-4
        self.test_dataset = test_dataset

    
    @torch.no_grad()
    def compute_train_loss(self, model):
        total_loss = 0.0
        train_dataloader = self.trainer.get_train_dataloader()

        for inputs in train_dataloader:
            total_loss += self.trainer.zo_forward(model, inputs, False).item()
        total_loss /= len(train_dataloader)
        
        return total_loss


    @torch.no_grad()
    def on_step_begin(self, args, state, control, **kwargs):
        global_step = state.global_step
        if global_step % self.every_steps != 0:
            return

        model = kwargs.get("model", None)
        if model is None:
            return
        model.eval()
        
        current_loss = self.compute_train_loss(model)
        perturb_loss = 0.0

        for _ in range(self.num_samples):
            random_seed = np.random.randint(1000000000)

            model = self.trainer.efficient_perturb_parameters(model, random_seed, scaling_factor=self.delta, notinFlat=False)
            perturb_loss += self.compute_train_loss(model)
            model = self.trainer.efficient_perturb_parameters(model, random_seed, scaling_factor=-self.delta, notinFlat=False)

        perturb_loss /= self.num_samples
        exp_sharp = 2 * (perturb_loss - current_loss) / (self.delta ** 2)

        output = self.trainer.evaluate(eval_dataset=self.test_dataset)
        metrics = output.metrics
        objective = self.trainer.dev_objective(metrics)

        logger.info(f"[ExpCallback] step={state.global_step}, train_loss={current_loss}, exp_sharp={exp_sharp}, test_acc={objective}")
