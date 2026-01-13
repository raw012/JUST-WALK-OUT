# This file aims to implement the loss function of the model
import torch
from utils_bbox import bbox_iou

class YOLOLoss(torch.nn.Module):
    def __init__(self, S=7, B=2, C=3, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, target):
        """
        preds: [N, S, S, B*5 + C]
        target: [N, S, S, 5 + C]
        """
        N = preds.size(0)
        preds = preds.view(N, self.S, self.S, self.B, 5)

        obj_mask = target[..., 0].unsqueeze(-1)
        tgt_box = target[..., 1:5].unsqueeze(3)

        ious = bbox_iou(preds[..., :4], tgt_box)
        best_iou, best_idx = ious.max(dim=3, keepdim=True)

        responsible = torch.zeros_like(ious)
        responsible.scatter_(3, best_idx, 1)
        responsible = responsible.unsqueeze(-1)

        # Coord loss
        coord_loss = self.lambda_coord * torch.sum(
            responsible * obj_mask.unsqueeze(3) *
            (preds[..., :2] - tgt_box[..., :2])**2
        )

        size_loss = self.lambda_coord * torch.sum(
            responsible * obj_mask.unsqueeze(3) *
            (torch.sqrt(preds[..., 2:4].clamp(1e-6)) -
             torch.sqrt(tgt_box[..., 2:4]))**2
        )

        # Confidence loss
        conf_loss_obj = torch.sum(
            responsible * obj_mask.unsqueeze(3) *
            (preds[..., 4:5] - best_iou.unsqueeze(-1))**2
        )

        conf_loss_noobj = self.lambda_noobj * torch.sum(
            (1 - responsible) * (1 - obj_mask.unsqueeze(3)) *
            preds[..., 4:5]**2
        )

        # Class loss
        class_loss = torch.sum(
            obj_mask * (preds[..., self.B*5:] - target[..., 5:])**2
        )

        return (coord_loss + size_loss + conf_loss_obj +
                conf_loss_noobj + class_loss) / N
