import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import os


class ScheduledOptim:
    '''tricky scheduler from an article'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class BERTTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader=None,
        lr= 1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        log_freq=100,
        device='cuda:0',
        checkpoint_dir='checkpoints_mlm_nsp',
        train_task="mlm", # mlm or nsp
        ):

        self.device = device
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.checkpoint_dir = checkpoint_dir
        self.train_task = train_task
        
        # setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert_model.model_dim, n_warmup_steps=warmup_steps
            )

        # using Negative Log Likelihood Loss function for predicting the masked_token and next_sentence
        ignore_index = -100 if self.train_task == "mlm" else 0
        self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        self.log_freq = log_freq
        print("total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def save_checkpoint(self, epoch):
        # save the model checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"bert_model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.bert_model.state_dict(),
        }, checkpoint_path)

        print(f"checkpoint saved at {checkpoint_path}")

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        self.model.to(torch.device(self.device))
        self.model.bert_model.to(torch.device(self.device))

        avg_loss = 0.0

        mode = "train" if train else "test"
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()


        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"epoch{epoch}_mode_{mode}"):
            data = {key: value.to(self.device) for key, value in data.items()}

            # forward and loss computation based on the training task
            if self.train_task == "nsp":
                next_sent_output = self.model.forward(data["bert_input"].to(self.device), data["segment_label"].to(self.device))
                loss = self.criterion(next_sent_output, data["is_next"])
            elif self.train_task == "mlm":
                mask_lm_output = self.model.forward(data["input_ids"].to(self.device), None)
                loss = self.criterion(mask_lm_output.transpose(1, 2), data["labels"])


            # backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

        print(f"epoch {epoch}, {mode}: avg_loss={avg_loss / len(data_loader)}")

        # naive saiving policy
        if mode == "train":
            self.save_checkpoint(epoch)