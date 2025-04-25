import math
import os
import time

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from training.src.model_v2 import env
from training.src.model_v2.dataset.tile_dataset_v2 import TileDatasetV2
from training.src.model_v2.model_progress_utils import mean_average_precision


def train(model, optimizer, criterion, scheduler, epochs = 10):
    train_dataset = TileDatasetV2("./model_v2/dataset/train")
    train_dataloader = DataLoader(train_dataset, batch_size=env.BATCH_SIZE, shuffle=True)
    val_dataset = TileDatasetV2("./model_v2/dataset/valid")
    val_dataloader = DataLoader(val_dataset, batch_size=env.BATCH_SIZE, shuffle=True)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    tempdir = "./temp"
    os.makedirs(tempdir, exist_ok=True)
    best_model_path = os.path.join(tempdir, "best_model.pt")

    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
                no_of_batches = math.ceil(len(val_dataset) / env.BATCH_SIZE)
                all_preds = torch.zeros((no_of_batches, env.BATCH_SIZE, env.S, env.S, env.N, env.C + 5))
                all_targets = torch.zeros((no_of_batches, env.BATCH_SIZE, env.S, env.S, env.N, env.C + 5))

            running_loss = 0.0
            i = 0
            best_mAP = 0.0

            for inputs, targets in tqdm(dataloaders[phase]):
                inputs = inputs.to(env.DEVICE)
                targets = targets.to(env.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    else:
                        all_preds[i] = outputs
                        all_targets[i] = targets
                        i += 1

                running_loss += loss.item() * inputs.size(0)

            if phase == "train":
                torch.save(model.state_dict(), tempdir + f"/model_5.{epoch+1}.pt")
                scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            # if phase == "val":
                # # all_preds = all_preds.view(-1, env.S, env.S, env.N, env.C + 5)
                # # all_targets = all_targets.view(-1, env.S, env.S, env.N, env.C + 5)
                # mAP = mean_average_precision(all_preds.to("cpu"), all_targets.to("cpu"), val_dataset)
                # print(f"mAP: {mAP.item():.4f}")
                #
                # if mAP > best_mAP:
                #     best_mAP = mAP.item()
                #     torch.save(model.state_dict(), best_model_path)
                #     print("Best model saved!")

    training_time = time.time() - start
    print(f"Training complete in {training_time:.2f} seconds")
    # print(f"Best mAP: {best_mAP:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    return model

