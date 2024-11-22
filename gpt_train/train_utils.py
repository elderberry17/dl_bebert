import torch
from tqdm import tqdm
import os

def save_checkpoint(model, checkpoints_dir, epoch):
    # save the model checkpoint
    checkpoint_path = os.path.join(checkpoints_dir, f"gpt_model_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)

    print(f"checkpoint saved at {checkpoint_path}")


def train_val_loop(model, train_loader, val_loader, optimizer, num_epochs, checkpoints_dir="checkpoints_clm"):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"epoch {epoch}:", end=' ')
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = val_epoch(model, val_loader)
        print(f"avg train loss = {train_loss} | avg val loss = {val_loss}")
    
        if val_loss < best_val_loss:
            print(f"val loss dropped from {best_val_loss} to {val_loss}. saving the model...")
            save_checkpoint(model, checkpoints_dir, epoch)
            best_val_loss = val_loss

def train_epoch(model, loader, optimizer):
    total_loss = 0

    model.train()
    for batch in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        _, loss = model(batch['input_ids'], batch['attention_mask'], mode="train")
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss
    


def val_epoch(model, loader):
    total_loss = 0

    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        with torch.no_grad():
            _, loss = model(batch['input_ids'], batch['attention_mask'], mode="train")
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss