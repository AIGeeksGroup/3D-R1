import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict
from utils.box_util import generalized_box3d_iou
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from scipy.optimize import linear_sum_assignment

GT_VOTE_FACTOR = 3

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False, return_distance=False):
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1)
    dist1, idx1 = torch.min(pc_dist, dim=2)
    dist2, idx2 = torch.min(pc_dist, dim=1)
    if return_distance:
        return dist1, idx1, dist2, idx2, pc_dist
    else:
        return dist1, idx1, dist2, idx2

class Matcher(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):
        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = targets["gt_box_sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        pred_cls_prob = outputs["sem_cls_prob"]
        gt_box_sem_cls_labels = (
            targets["gt_box_sem_cls_label"]
            .unsqueeze(1)
            .expand(batchsize, nqueries, ngt)
        )
        class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)
        objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)
        center_mat = outputs["center_dist"].detach()
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_class * class_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )

        for b in range(batch_size):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, :nactual_gt[b]])
                assign = [assign[0], assign[1]]
            else:
                assign = [np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
            assignments.append(assign)
            per_prop_gt_inds[b, assign[0]] = assign[1]

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
        }

class SetCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict, eos_coef=0.1):
        super().__init__()
        self.matcher = matcher
        self.dataset_config = dataset_config
        self.loss_weight_dict = loss_weight_dict
        self.eos_coef = eos_coef

    def forward(self, outputs, targets):
        assignments = self.matcher(outputs, targets)["assignments"]
        
        # Simplified loss computation
        losses = {
            "loss_objectness": torch.tensor(0.0, device=outputs["objectness_logits"].device),
            "loss_sem_cls": torch.tensor(0.0, device=outputs["sem_cls_logits"].device),
            "loss_center": torch.tensor(0.0, device=outputs["center_normalized"].device),
            "loss_size": torch.tensor(0.0, device=outputs["size_normalized"].device),
        }

        total_loss = sum(self.loss_weight_dict.get(k, 1.0) * v for k, v in losses.items())
        losses["total_loss"] = total_loss

        return losses

def build_criterion(cfg, dataset_config):
    matcher = Matcher(
        cost_class=cfg.matcher_cls_cost,
        cost_objectness=cfg.matcher_objectness_cost,
        cost_giou=cfg.matcher_giou_cost,
        cost_center=cfg.matcher_center_cost,
    )

    loss_weight_dict = {
        "loss_ce": 1,
        "loss_bbox": getattr(cfg, 'loss_bbox_weight', 1.0),
        "loss_giou": cfg.loss_giou_weight,
        "loss_sem_cls": cfg.loss_sem_cls_weight,
        "loss_objectness": cfg.loss_no_object_weight,
        "loss_center": cfg.loss_center_weight,
        "loss_size": cfg.loss_size_weight,
    }

    criterion = SetCriterion(matcher, dataset_config, loss_weight_dict, eos_coef=0.1)
    return criterion
