import torch

from training.src.model_v2 import env
from training.src.model_v2.darknet_backbone import YoloV2
from training.src.model_v2.training_model import train
from training.src.model_v2.yolo_loss import YoloV2Loss

print(env.DEVICE)
print(torch.cuda.mem_get_info())

model = YoloV2()
model.load_state_dict(torch.load("./temp/model_5.0.pt"))
model = model.to(env.DEVICE)
criterion = YoloV2Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
model = train(model, optimizer, criterion, exp_lr_scheduler)
