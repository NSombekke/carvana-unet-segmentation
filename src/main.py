import argparse
import torch
import os

from tqdm import tqdm
from torch.optim import Adam
from datetime import datetime
from copy import deepcopy
from functools import partialmethod

from utils import set_seed, save_model
from dataset import get_dataloaders, download_datasets
from model import get_model
from loss import get_loss_func
from transform import get_transforms
    
def train(model, loss_func, optimizer, train_dl, val_dl, device, args):
    start_epoch = 0
    best_epoch = 0
    best_model = None
    best_loss = float('inf')
    model_id = f"{args.model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    model_dir = os.path.join(args.output_dir, model_id)
    os.makedirs(model_dir)
    
    if args.model_path:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']   
        best_loss = checkpoint['best_loss']
    
    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        train_loss = train_loop(model, loss_func, optimizer, train_dl, device)
        val_loss = val_loop(model, loss_func, val_dl, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = deepcopy(model)
        print(f"Epoch: {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss}")
        save_model(os.path.join(model_dir, f"{model_id}_{epoch}.pth"), model, optimizer, best_loss, epoch)
    save_model(os.path.join(model_dir, f"{model_id}_best{best_epoch}.pth"), best_model, optimizer, best_loss, best_epoch)

def train_loop(model, loss_func, optimizer, train_dl, device):
    model.train()
    train_loss = 0
    for image, mask in tqdm(train_dl):
        image, mask = image.to(device), mask.to(device)
        pred = model(image)
        loss = loss_func(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_dl)

def val_loop(model, loss_func, val_dl, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for image, mask in tqdm(val_dl):
            image, mask = image.to(device), mask.to(device)
            pred = model(image)
            loss = loss_func(pred, mask)
            val_loss += loss.item()
    return val_loss / len(val_dl)

def test(model, test_dl, device, args):
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    model.eval()
    with torch.no_grad():
        for image in test_dl:
            image = image.to(device)
            pred = model(image)

def main(args):
    if args.download:
        download_datasets(args.data_dir)
    set_seed(args.seed)
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=args.no_progress)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using {device}")
    model = get_model(args.model_name)
    model.to(device)
    loss_func = get_loss_func(args.loss_func)
    optimizer = Adam(model.parameters(), lr=args.lr)
    transforms = get_transforms(args.input_size)
    train_dl, val_dl, test_dl = get_dataloaders(args.data_dir, args.batch_size, args.num_workers, transforms)
    if args.eval:
        test(model, test_dl, device, args)
    else:
        train(model, loss_func, optimizer, train_dl, val_dl, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carvana Masking Challenge using Pytorch - Training")
    parser.add_argument("--data_dir", type=str, help="data directory", default="../data")
    parser.add_argument("--output_dir", type=str, help="output directory", default="../output")
    parser.add_argument("--model_path", type=str, help="model path for continue training or evaluation", default="")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--num_workers", type=int, help="number of workers", default=2)
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.0001)
    parser.add_argument("--loss_func", type=str, help="loss function", default="dice")
    parser.add_argument("--model_name", type=str, help="model name", default="unet")
    parser.add_argument("--cpu", action="store_true", help="use cpu")
    parser.add_argument("--input_size", type=tuple, help="input size", default=(320, 480))
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--download", action="store_true", help="download datasets")
    parser.add_argument("--eval", action="store_true", help="evaluate model")
    parser.add_argument("--no_progress", action="store_true", help="show no tqdm progress bar")
    args = parser.parse_args()
    
    main(args)