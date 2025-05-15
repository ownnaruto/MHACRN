import copy
import datetime
import os
import torch
import numpy as np
from torchinfo import summary
from .STFRunner import STFRunner 
import sys 
from tqdm import tqdm 
import time 

sys.path.append("..")

import matplotlib.pyplot as plt     
from lib.utils import print_log
from lib.metrics import RMSE_MAE_MAPE

class MHACRNRUNNER(STFRunner):
    def __init__(
        self,
        cfg: dict,
        device,
        scaler,
        log=None,
    ):
        super().__init__(cfg, device, scaler, log)

        self.cfg = cfg
        self.device = device
        self.scaler = scaler
        self.log = log

        self.clip_grad = cfg.get("clip_grad")

        self.batches_seen = 0

    def train_one_epoch(self, model, trainset_loader, optimizer, scheduler, criterion, epoch=None, save_path=None, last_epoch=None):
        model.train()
        batch_loss_list = []
        
        if epoch >= last_epoch:
            
            use_cl = self.cfg.get('use_cl')
            warm_epoch = self.cfg.get('warm_epoch', 0)
            cl_step_size = self.cfg.get('cl_step_size')
            out_steps = self.cfg.get('out_steps')
            
            if use_cl and epoch >= warm_epoch:
                target_length = (epoch - warm_epoch) // cl_step_size + 1
                
                target_length = min(target_length ,out_steps)        
                
                if (epoch - warm_epoch) % cl_step_size == 0 :
                    print_log(f'CL target length = {target_length}', log=self.log)
            else:
                target_length = out_steps 
            
            
            for x_batch, y_batch in tqdm(trainset_loader, ncols=80): 
                x = x_batch.to(self.device)
                y = y_batch.to(self.device) 
                y_true = y[..., [0]]
                y_cov = y[..., 1:]
                output = model(x, y_cov, self.scaler.transform(y_true), self.batches_seen)
                y_pred = self.scaler.inverse_transform(output)

                self.batches_seen += 1

                    
                loss = criterion(
                            y_pred[:, :target_length, ...], 
                            y_true[:, :target_length, ...]
                )
                
                
                batch_loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                optimizer.step()

            if save_path:
                save = os.path.join(save_path, f'epoch_{epoch}.pt')
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, save) 
            epoch_loss = np.mean(batch_loss_list)
        else:
            epoch_loss = 0 
        scheduler.step()
        for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], self.cfg.get('min_lr_rate', 1e-4))
        return epoch_loss 

    @torch.no_grad()
    def eval_model(self, model, valset_loader, criterion):
        model.eval()
        batch_loss_list = []
        for x_batch, y_batch in valset_loader:
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)

            y_true = y[..., [0]]
            y_cov = y[..., 1:]

            output = model(x, y_cov)
            y_pred = self.scaler.inverse_transform(output)

            loss = criterion(y_pred, y_true)     
            batch_loss_list.append(loss.item()) 

        return np.mean(batch_loss_list)

    @torch.no_grad()
    def predict(self, model, loader):
        model.eval()
        y_list = []
        out_list = []

        for x_batch, y_batch in loader:
            x = x_batch.to(self.device)
            y = y_batch.to(self.device) 

            y_true = y[..., [0]]
            y_cov = y[..., 1:]

            output = model(x, y_cov)
            y_pred = self.scaler.inverse_transform(output)

            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()
            out_list.append(y_pred)
            y_list.append(y_true)

        out = np.vstack(out_list).squeeze() 
        y = np.vstack(y_list).squeeze()

        return y, out

    def model_summary(self, model, dataloader):
        x_shape = next(iter(dataloader))[0].shape
        y_cov_shape = next(iter(dataloader))[1][..., 1:].shape
        return summary(
            model,
            [x_shape, y_cov_shape],
            verbose=0,  
            device=self.device,
        )
    
    def train(
        self,
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=200,
        early_stop=10,
        compile_model=False,
        verbose=1,
        plot=False,
        save=None, 
        last_epoch=-1,
    ):
        if torch.__version__ >= "2.0.0" and compile_model:
            model = torch.compile(model)
            
        wait = 0
        min_val_loss = np.inf

        train_loss_list = []
        val_loss_list = []

        
        for epoch in range(max_epochs):
        
            train_loss = self.train_one_epoch(
                model, trainset_loader, optimizer, scheduler, criterion,
                epoch=epoch, last_epoch = last_epoch
            )
            train_loss_list.append(train_loss)

            if epoch + 1 < last_epoch:
                continue 
            val_loss = self.eval_model(model, valset_loader, criterion)
            val_loss_list.append(val_loss)
            
            if (epoch + 1) % verbose == 0:
                print_log(
                    datetime.datetime.now(),
                    "Epoch",
                    epoch + 1,
                    'lr = %.5f'% optimizer.param_groups[0]['lr'],
                    " \tTrain Loss = %.5f" % train_loss,
                    "Val Loss = %.5f" % val_loss,
                    log=self.log,
                )

            if val_loss < min_val_loss:
                wait = 0
                min_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= early_stop:
                    break

        model.load_state_dict(best_state_dict)
        train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(
            *self.predict(model, trainset_loader)
        )
        val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*self.predict(model, valset_loader))

        out_str = f"Early stopping at epoch: {epoch+1}\n"
        out_str += f"Best at epoch {best_epoch+1}:\n"
        out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
        out_str += "Train MAE = %.5f, RMSE = %.5f, MAPE = %.5f\n" % (
            train_mae,
            train_rmse,
            train_mape,
        )
        out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
        out_str += "Val MAE = %.5f, RMSE = %.5f, MAPE = %.5f" % (
            val_mae,
            val_rmse,
            val_mape,
        )
        print_log(out_str, log=self.log)

        if plot:
            plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
            plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
            plt.title("Epoch-Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        
        if save:
            torch.save(best_state_dict, save) 
            
        return model

    def test_model(self, model, testset_loader, save_pred=None):
        model.eval() 
            
        print_log("--------- Test ---------", log=self.log)

        start = time.time()
        y_true, y_pred = self.predict(model, testset_loader)
        
        if save_pred:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            path =  os.path.join(save_pred, f'{now}.npy')
            np.save(path, y_pred)
            
        end = time.time()

        out_steps = y_pred.shape[1]

        rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
        out_str = "All Steps (1-%d) MAE = %.5f, RMSE = %.5f, MAPE = %.5f\n" % (
            out_steps,
            mae_all,
            rmse_all,
            mape_all,
        )

        for i in range(out_steps):
            rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
            out_str += "Step %d MAE = %.5f, RMSE = %.5f, MAPE = %.5f\n" % (
                i + 1,
                mae,
                rmse,
                mape,
            )

        print_log(out_str, log=self.log, end="")
        print_log("Inference time: %.2f s" % (end - start), log=self.log)
        if save_pred:
            print_log(f"Prediction Save in {path}")