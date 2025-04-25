import torch

from training.src.model_v2 import env
from training.src.model_v2.darknet_backbone import YoloV2
from training.src.model_v2.dataset.tile_dataset_v2 import TileDatasetV2


class YoloV2Loss(torch.nn.Module):
    def __init__(self):
        super(YoloV2Loss, self).__init__()

        self.lambda_noobj = torch.tensor(1.3).to(env.DEVICE)
        self.lambda_obj = torch.tensor(1.0).to(env.DEVICE)
        self.lambda_class = torch.tensor(3.3).to(env.DEVICE)
        self.lambda_coord = torch.tensor(6.0).to(env.DEVICE)

        self.binary_loss = torch.nn.BCEWithLogitsLoss()
        self.logistic_loss = torch.nn.CrossEntropyLoss()
        self.regression_loss = torch.nn.MSELoss()

    def forward(self, predictions, targets):
        obj_mask = targets[..., 4] == 1
        noobj_mask = targets[..., 4] == 0

        if noobj_mask.any():
            noobj_loss = self.binary_loss(
                predictions[noobj_mask][[..., 4]], targets[noobj_mask][[..., 4]]
            )
        else:
            noobj_loss = torch.tensor(0.0, device=predictions.device)

        if obj_mask.any():
            obj_loss = self.binary_loss(
                predictions[obj_mask][[..., 4]], targets[obj_mask][[..., 4]]
            )

            pred_wh = torch.exp(predictions[obj_mask][..., 2:4])
            pred_coord = torch.cat((torch.sigmoid(predictions[obj_mask][..., 0:2]), pred_wh), dim=-1)
            coord_loss = self.regression_loss(
                pred_coord, targets[obj_mask][..., 0:4]
            )

            class_loss = self.logistic_loss(
                predictions[obj_mask][..., 5:], targets[obj_mask][..., 5:]
            )
        else:
            obj_loss = torch.tensor(0.0, device=predictions.device)
            coord_loss = torch.tensor(0.0, device=predictions.device)
            class_loss = torch.tensor(0.0, device=predictions.device)

        loss = (
            self.lambda_coord * coord_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_obj * obj_loss +
            self.lambda_class * class_loss
        )

        return loss


if __name__ == "__main__":
    loss = YoloV2Loss()
    model = YoloV2()

    img, target = TileDatasetV2("./dataset/train")[101]
    target = target.unsqueeze(dim=0)
    pred = model(img.unsqueeze_(dim=0))
    print(loss(pred, target))
