import torch as t
import tqdm
import Model.Loss
import torch.optim as optim

assert t.cuda.is_available()
device="cuda"


# Dataloader
epoch=10


for e in range(epoch):
    for im in tqdm(dataloader):
        im.to(device)

        # Loss Calculation

        # Update (step)





    # epoch sonu bastırma

