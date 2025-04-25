import torch

from tqdm import tqdm
from training.src.model_v2 import env
from training.src.model_v2.darknet_backbone import YoloV2
from training.src.model_v2.dataset.tile_dataset_v2 import TileDatasetV2
from training.src.model_v2.model_progress_utils import check_model_accuracy, cal_epoch_acc


def test_model(path):
    # Test model accuracy - IOU not taken into account
    model = YoloV2()
    model.load_state_dict(torch.load(path))
    model = model.to(env.device)
    model.eval()

    test_dataset = TileDatasetV2("./dataset/test")
    # test_dataset = TileDatasetV2("./dataset/valid")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    preds, targets = [], []

    for inputs, target in tqdm(test_dataloader):
        inputs = inputs.to(env.device)
        target = target.to(env.device)

        with torch.no_grad():
            preds.append(model(inputs)[0])
        targets.append(target[0])
    preds, targets = torch.stack(preds), torch.stack(targets)


    # model_accuracy = check_model_accuracy(preds, targets, thres=0.5, dataset=test_dataset)
    # cal_epoch_acc(*model_accuracy)

    # model_accuracy = check_model_accuracy(preds, targets, thres=0.55)
    # cal_epoch_acc(*model_accuracy)

    model_accuracy = check_model_accuracy(preds, targets, thres=0.6)
    cal_epoch_acc(*model_accuracy)

    # model_accuracy = check_model_accuracy(preds, targets, thres=0.8)
    # cal_epoch_acc(*model_accuracy)

if __name__ == "__main__":
    test_model("../temp/model_4.4.pt")
