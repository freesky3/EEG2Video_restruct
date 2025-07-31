'''
train the model and save it
'''
import torch
from torch import nn


def loss_acc(batch, model, criterion, device):
    '''Forward a batch through the model'''
    input, labels = batch
    input = input.to(device)
    labels = labels.to(device)

    output = model(input)

    loss = criterion(output, labels)

    preds = output.argmax(1)
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


from tqdm import tqdm


def valid(data_loader, model, criterion, device): 
    '''Validate the model on the validation set'''
    model.eval()
    running_loss = 0
    running_accuracy = 0
    pbar = tqdm(total = len(data_loader.dataset), ncols=0, desc='Valid')

    for i, batch in enumerate(data_loader):
        with torch.no_grad():
            loss, accuracy = loss_acc(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()
        pbar.update(data_loader.batch_size)
        pbar.set_postfix(
			loss=f"{running_loss / (i+1):.2f}",
			accuracy=f"{running_accuracy / (i+1):.2f}",
		)
    pbar.close()
    model.train()

    return running_accuracy / len(data_loader)



import numpy as np
from fix_seed import set_seed
from dataloader import get_dataloader
from models import glfnet
import os
from dotenv import load_dotenv

load_dotenv("variable.env")

# set hyperparameters
def config():
    configutation = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'data_path': os.getenv("EEG_VP_train_watching_data_path"),
        'save_path': os.getenv("EEG_VP_train_save_path"),
        'seed': 42, 
        'valid_steps': 2000,
        'save_steps': 10000, 
        'total_steps': 70000
    }
    return configutation

def main(
        batch_size,
        learning_rate,
        data_path,
        save_path,
        seed,
        valid_steps,
        save_steps, 
        total_steps
): 
    
    # set seed to ensure reproducibility
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    # get dataloader
    train_loader, valid_loader = get_dataloader(data_path, batch_size, num_workers=0)
    train_iterator = iter(train_loader)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model = glfnet(out_dim=50, emb_dim=64, input_dim=310)
    model.apply(init_weights)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200 * len(train_loader))


    pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit='step')
    # start training
    for step in range(total_steps):
        try: 
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        
        loss, accuracy = loss_acc(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(
			loss=f"{batch_loss:.2f}",
			accuracy=f"{batch_accuracy:.2f}",
			step=step + 1,
		)

        # Validate
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader, model, criterion, device)
            best_accuracy = 0

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                bet_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit='step')
        
        # Save model
        if (step + 1) % save_steps == 0 and bet_state_dict is not None:
            torch.save(bet_state_dict, os.path.join(save_path, f"model_{step+1}.pth"))
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

        pbar.close()



if __name__ == '__main__':
    main(**config())