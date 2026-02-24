import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from untils.initialize import *
from config import args
import untils.data_set as data_set
import models.GCDLNet

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''
torch.cuda.empty_cache()


def setup_logger(log_path):
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    return logger


class EarlyStopping:
    def __init__(self, patience=args.patience, verbose=False, delta=args.delta):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        print(f"val_loss={val_loss}")
        score = -val_loss
        if math.isnan(score):
            return 
        
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'GCDLNet_latest.pth'))
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), os.path.join(path, 'GCDLNet_best.pth'))
        self.val_loss_min = val_loss


def main():
    device = args.device
    print(f"Current working directory: {os.getcwd()}")
    
    log_path = os.path.abspath('./models/GCDLNet.log')
    print(f"Log path: {log_path}")
    logger = setup_logger(log_path)

    train_ys = [os.path.join(args.train_label_dir, f) for f in os.listdir(args.train_label_dir) if f.endswith('.npy')]
    train_xs = [os.path.join(args.train_img_dir, f) for f in os.listdir(args.train_label_dir) if f.endswith('.npy')]
    valid_ys = [os.path.join(args.test_label_dir, f) for f in os.listdir(args.test_label_dir) if f.endswith('.npy')]
    valid_xs = [os.path.join(args.test_img_dir, f) for f in os.listdir(args.test_label_dir) if f.endswith('.npy')]
            
    train_dataset = data_set.DatasetCustom(x_paths=train_xs, y_paths=train_ys)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    
    valid_dataset = data_set.DatasetCustom(x_paths=valid_xs, y_paths=valid_ys)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True)
    
    print(f"Dataset size: {len(train_dataset)}")
    
    net = models.GCDLNet.GCDLNet(os.path.join(args.save_model, "mean_model.pth")).to(device)
    
    latest_model_path = os.path.join(args.save_model, 'GCDLNet_latest.pth')
    if os.path.exists(latest_model_path):
        state_dict = torch.load(latest_model_path, map_location=device)
        net.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pre-trained model.")

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion_train = args.train_loss
    criterion_valid = args.valid_loss
    
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        net.train()
        train_epoch_loss = []
        
        for inputs, targets in tqdm(train_loader, total=len(train_loader), desc=f"\033[31mEpoch {epoch + 1}, training:\033[0m"):
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            
            outputs = net(inputs)
            optimizer.zero_grad()
            loss = criterion_train(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            
        avg_train_loss = np.average(train_epoch_loss)
        print(f"\nepoch={epoch}/{args.epochs}, loss={avg_train_loss}")
        logging.info(f"epoch={epoch}/{args.epochs}, loss={avg_train_loss}")
        train_epochs_loss.append(avg_train_loss)

        valid_epoch_loss = []
        net.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader, total=len(valid_loader), desc=f"\033[31mEpoch {epoch + 1}, validation:\033[0m"):
                torch.cuda.empty_cache()
                inputs = inputs.to(torch.float32).to(device)
                targets = targets.to(torch.float32).to(device)
                
                outputs = net(inputs)
                loss = criterion_valid(outputs, targets)
                
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
                
        avg_valid_loss = np.average(valid_epoch_loss)
        valid_epochs_loss.append(avg_valid_loss)
        
        early_stopping(valid_epochs_loss[-1], model=net, path=args.save_model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    os.makedirs('./loss', exist_ok=True)
    np.save('./loss/train_loss.npy', np.array(train_loss))
    np.save('./loss/valid_loss.npy', np.array(valid_loss))
    np.save('./loss/train_epochs_loss.npy', np.array(train_epochs_loss))
    np.save('./loss/valid_epochs_loss.npy', np.array(valid_epochs_loss))

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss[1:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.savefig('./Loss.png')
    plt.show()


if __name__ == '__main__':
    main()