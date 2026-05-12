import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import evaluate

class Trainer:
    def __init__(self, model, optimizer, device, train_stats, step_interval, save_model_dir):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_history = []
        self.extr_test_history = []
        self.gen_test_history = []
        self.model_dir = save_model_dir
        self.train_stats = train_stats
        
        # Ensure stats are detached so no gradients flow through them
        self.mean_node_dv = self.train_stats[2][0].detach()
        self.mean_node_disp = self.train_stats[3][0].detach()
        self.std_node_dv = self.train_stats[2][1].detach()
        self.std_node_disp = self.train_stats[3][1].detach()
        
        # Initialize best test loss and best epoch.
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.best_epoch = None        
        self.step_interval = step_interval
        
        # Internal loss tracking
        self.loss = 0.0

    def train(self, train_graph, accumulate=False, accumulation_steps=1):
        """
        Modified train method to support gradient accumulation.
        
        Parameters:
        -----------
        train_graph : torch_geometric.data.Data
        accumulate  : bool (True if we are still summing gradients, False if we step)
        accumulation_steps : int (Total mini-batches per simulated batch, usually 16 or 8)
        """
        self.model.train()
        
        # 1. Move data to device once
        train_graph = train_graph.to(self.device)
        
        # 2. Forward Pass
        pred_node_dvel, pred_node_disp = self.model(train_graph)

        # 3. Target
        actual_node_dvel = train_graph.y_dv.float()
        actual_node_disp = train_graph.y_dx.float()

        # Detaching stats again here is a safety measure to ensure the graph stays leaf-like
        pred_node_dvel_ = (pred_node_dvel - self.mean_node_dv) / self.std_node_dv
        pred_node_disp_ = (pred_node_disp - self.mean_node_disp) / self.std_node_disp
        actual_node_dvel_ = (actual_node_dvel - self.mean_node_dv) / self.std_node_dv
        actual_node_disp_ = (actual_node_disp - self.mean_node_disp) / self.std_node_disp

        # 4. Compute Loss and Scale it
        # We MUST divide by accumulation_steps so the total gradient is the AVERAGE of the big batch
        raw_loss = F.mse_loss(pred_node_disp_, actual_node_disp_) + F.mse_loss(pred_node_dvel_, actual_node_dvel_)
        scaled_loss = raw_loss / accumulation_steps
        
        # 5. Backward Pass (Accumulates gradients in .grad buffers)
        scaled_loss.backward()

        # 6. Optimizer Step (Only when accumulation cycle is finished)
        if not accumulate:
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Record the non-scaled loss for logging
        self.loss = raw_loss.cpu().detach().numpy()
        self.train_history.append(self.loss)

    def test(self, test_loader, mode='val', epoch=None):
        self.model.eval()
        mean_pos_error = evaluate(test_loader, self.model, self.device)
        
        if mode == 'val':
            self.val_loss_pos = mean_pos_error
            self.gen_test_history.append(self.val_loss_pos)
            if epoch is not None and self.val_loss_pos < self.best_val_loss:
                self.best_val_loss = self.val_loss_pos
                self.best_epoch = epoch
                self.save_model(epoch)
        if mode == 'test':
            self.test_loss_pos = mean_pos_error

    def save_model(self, iteration):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # Clean up old best model to save disk space
        if self.best_model_path and os.path.exists(self.best_model_path):
            try:
                os.remove(self.best_model_path)
            except:
                pass
        self.best_model_path = os.path.join(self.model_dir, 
                                 f'Val_Loss_{self.val_loss_pos:.5f}mm_iter{iteration}.pth')
        torch.save(self.model.state_dict(), self.best_model_path)