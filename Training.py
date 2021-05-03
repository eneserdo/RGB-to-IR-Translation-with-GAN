import torch as t
import tqdm
import Model.Loss
import torch.optim as optim




def main():
    lambda_D = 1
    lambda_FM = 1
    lambda_P = 1

    assert t.cuda.is_available()
    device = "cuda"

    # Dataloader
    epoch = 10


    for e in range(epoch):
        for im in tqdm(dataloader):
            im.to(device)

            # Loss Calculation

            # Update (step)

        # epoch sonu bastırma ve kaydetme


if __name__ == '__main__':
    main()
