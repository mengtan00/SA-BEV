import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS
from .view_transformer import LSSViewTransformerBEVDepth, Mlp, SELayer, ASPP, DepthNet
from torch.cuda.amp.autocast_mode import autocast

class SABlock(nn.Module):
    """ Spatial attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x, y):
        return torch.mul(self.conv(x), self.attention(y))

class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self,  channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.depth2sem = SABlock(channels, channels)
        self.sem2depth = SABlock(channels, channels)

    def forward(self, depth, sem):
        depth_new = depth + self.sem2depth(sem, depth)
        sem_new = sem + self.depth2sem(depth, sem)
        return depth_new, sem_new

class TaskHead(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(TaskHead, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.decoder = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1), 
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ) 
        self.head = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, return_feat=True):
        if return_feat:
            feat = self.decoder(feat)
            return self.head(feat), feat 
        return self.head(self.decoder(feat))

class TaskFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TaskFPN, self).__init__()
        self.reduce_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.self_attention = SABlock(out_channels, out_channels)
    
    def forward(self, feat0, feat1):
        feat0 = self.reduce_conv(F.interpolate(feat0, scale_factor=2, mode='bilinear'))
        feat0_new = feat0 + self.self_attention(feat1, feat0)
        return feat0_new
            

class MSCThead(nn.Module):
    def __init__(self,
                 in_channels=[512, 512],
                 mid_channels=[512, 256],
                 depth_channels=118,
                 semantic_channels=2,
                 context_channels=80,
                ):
        super(MSCThead, self).__init__()
        # preprocess
        self.reduce_conv0 =  nn.Sequential(
                nn.Conv2d(in_channels[0], mid_channels[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mid_channels[0]), nn.ReLU(inplace=True))
        self.reduce_conv1 =  nn.Sequential(
                nn.Conv2d(in_channels[1], mid_channels[1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mid_channels[1]), nn.ReLU(inplace=True))
        self.bn = nn.BatchNorm1d(27)
        self.scale0_mlp = Mlp(27, mid_channels[0], mid_channels[0])
        self.scale1_mlp = Mlp(27, mid_channels[1], mid_channels[1])
        self.scale0_se = SELayer(mid_channels[0])
        self.scale1_se = SELayer(mid_channels[1])
        self.aspp = ASPP(mid_channels[0], mid_channels[0])
        # stage one
        self.depth_head0= TaskHead(mid_channels[0], mid_channels[0], depth_channels)
        self.semantic_head0 = TaskHead(mid_channels[0], mid_channels[0], semantic_channels)
        self.context_conv0 = nn.Sequential(
            nn.Conv2d(mid_channels[0], mid_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[0]), 
            nn.ReLU(inplace=True)
        ) 
        # combine information
        self.mtd = MultiTaskDistillationModule(mid_channels[0])
        self.depth_fpn = TaskFPN(mid_channels[0], mid_channels[1])
        self.semantic_fpn = TaskFPN(mid_channels[0], mid_channels[1])
        self.context_fpn = TaskFPN(mid_channels[0], mid_channels[1])
        # stage two
        self.depth_head1 = TaskHead(mid_channels[1], mid_channels[1], depth_channels)
        self.semantic_head1 = TaskHead(mid_channels[1], mid_channels[1], semantic_channels)
        self.context_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels[1], mid_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[1]), 
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels[1], context_channels, kernel_size=1, stride=1, padding=0)
        ) 
    
    @autocast(False)
    def forward(self, x, mlp_input):
        # preprocess
        B, N, C, H, W = x[0].shape
        scale0_feat = x[0].view(B * N, C, H, W).float()
        B, N, C, H, W = x[1].shape
        scale1_feat = x[1].view(B * N, C, H, W).float()
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        scale0_feat = self.reduce_conv0(scale0_feat)
        scale1_feat = self.reduce_conv1(scale1_feat)
        scale0_se = self.scale0_mlp(mlp_input)[..., None, None]
        scale1_se = self.scale1_mlp(mlp_input)[..., None, None]
        scale0_feat = self.scale0_se(scale0_feat, scale0_se)
        scale1_feat = self.scale1_se(scale1_feat, scale1_se)
        scale0_feat = self.aspp(scale0_feat)
        # stage one
        depth0, depth_feat = self.depth_head0(scale0_feat)
        semantic0, semantic_feat = self.semantic_head0(scale0_feat)
        context_feat = self.context_conv0(scale0_feat)
        # combine information
        depth_feat, semantic_feat = self.mtd(depth_feat, semantic_feat)
        depth_feat = self.depth_fpn(depth_feat, scale1_feat)
        semantic_feat = self.semantic_fpn(semantic_feat, scale1_feat)
        context_feat = self.context_fpn(context_feat, scale1_feat)
        # stage two
        depth1 = self.depth_head1(depth_feat, return_feat=False)
        semantic1 = self.semantic_head1(semantic_feat, return_feat=False)
        context1 = self.context_conv1(context_feat)

        return (depth1, semantic1, context1, depth0, semantic0)

@NECKS.register_module()
class SABEVPool(LSSViewTransformerBEVDepth):

    def __init__(self, loss_depth_weight=3.0, loss_semantic_weight=25, depthnet_cfg=dict(),
                 depth_threshold=1, semantic_threshold=0.25, **kwargs):
        super(SABEVPool, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.loss_semantic_weight = loss_semantic_weight
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.out_channels, self.D + 2, **depthnet_cfg)
        # self.depth_net = MSCThead(in_channels, mid_channels, self.D, 2, self.out_channels)
        self.depth_threshold = depth_threshold / self.D
        self.semantic_threshold = semantic_threshold

    def get_downsampled_gt_depth_and_semantic(self, gt_depths, gt_semantics):
        # remove point not in depth range
        gt_semantics[gt_depths < self.grid_config['depth'][0]] = 0
        gt_semantics[gt_depths > self.grid_config['depth'][1]] = 0
        gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
        gt_depths[gt_depths > self.grid_config['depth'][1]] = 0
        gt_semantic_depths = gt_depths * gt_semantics

        B, N, H, W = gt_semantics.shape
        gt_semantics = gt_semantics.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantics = gt_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantics = gt_semantics.view(
            -1, self.downsample * self.downsample)
        gt_semantics = torch.max(gt_semantics, dim=-1).values
        gt_semantics = gt_semantics.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_semantics = F.one_hot(gt_semantics.long(),
                              num_classes=2).view(-1, 2).float()

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_depths = (gt_depths -
                     (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.D + 1).view(
                                  -1, self.D + 1)[:, 1:].float()
        gt_semantic_depths = gt_semantic_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_semantic_depths = gt_semantic_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantic_depths = gt_semantic_depths.view(
            -1, self.downsample * self.downsample)
        gt_semantic_depths =  torch.where(gt_semantic_depths == 0.0,
                                    1e5 * torch.ones_like(gt_semantic_depths),
                                    gt_semantic_depths)
        gt_semantic_depths = (gt_semantic_depths - (self.grid_config['depth'][0] - 
                            self.grid_config['depth'][2])) / self.grid_config['depth'][2] 
        gt_semantic_depths = torch.where(
                    (gt_semantic_depths < self.D + 1) & (gt_semantic_depths >= 0.0),
                    gt_semantic_depths, torch.zeros_like(gt_semantic_depths)).long()                           
        soft_depth_mask = gt_semantics[:,1] > 0
        gt_semantic_depths = gt_semantic_depths[soft_depth_mask]
        gt_semantic_depths_cnt = gt_semantic_depths.new_zeros([gt_semantic_depths.shape[0], self.D+1])
        for i in range(self.D+1):
            gt_semantic_depths_cnt[:,i] = (gt_semantic_depths == i).sum(dim=-1)
        gt_semantic_depths = gt_semantic_depths_cnt[:,1:] / gt_semantic_depths_cnt[:,1:].sum(dim=-1, keepdim=True)
        gt_depths[soft_depth_mask] = gt_semantic_depths
  
        return gt_depths, gt_semantics

    @force_fp32()
    def get_depth_and_semantic_loss(self, depth_labels, depth_preds, semantic_labels, semantic_preds):
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        semantic_preds = semantic_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        semantic_weight = torch.zeros_like(semantic_labels[:,1:2])
        semantic_weight = torch.fill_(semantic_weight, 0.1)
        semantic_weight[semantic_labels[:,1] > 0] = 0.9

        depth_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[depth_mask]
        depth_preds = depth_preds[depth_mask]
        semantic_labels = semantic_labels[depth_mask]
        semantic_preds = semantic_preds[depth_mask]
        semantic_weight = semantic_weight[depth_mask]

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ) * semantic_weight).sum() / max(0.1, semantic_weight.sum())

            pred = semantic_preds
            target = semantic_labels
            alpha = 0.25
            gamma = 2
            pt = (1 - pred) * target + pred * (1 - target)
            focal_weight = (alpha * target + (1 - alpha) *
                            (1 - target)) * pt.pow(gamma)
            semantic_loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
            semantic_loss = semantic_loss.sum() / max(1, len(semantic_loss))
        return self.loss_depth_weight * depth_loss, self.loss_semantic_weight * semantic_loss


    def voxel_pooling_prepare_v2(self, coor, kept):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).to(coor).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1)
        coor = torch.cat((coor, batch_idx), 1)
        # filter out points that are outside box
        kept = kept.view(num_points)
        kept &= (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]
        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat, kept):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor, kept)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            dummy += feat.mean() * 0
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def view_transform(self, img_feat_shape, input, depth, tran_feat, kept, paste_idx, bda_paste):
        B, N, C, H, W = img_feat_shape
        coor = self.get_lidar_coor(*input[1:7])
        if paste_idx is None:
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W), kept)
        else:
            paste_idx0 = [a[0] for a in paste_idx]
            coor0 = bda_paste.view(B, 1, 1, 1, 1, 3,
                          3).matmul(coor[paste_idx0].unsqueeze(-1)).squeeze(-1)
            bev_feat = self.voxel_pooling_v2(
                coor0, depth.view(B, N, self.D, H, W)[paste_idx0],
                tran_feat.view(B, N, self.out_channels, H, W)[paste_idx0], 
                kept.view(B, N, self.D, H, W)[paste_idx0])

            paste_idx1 = [a[1] for a in paste_idx]
            coor1 = bda_paste.view(B, 1, 1, 1, 1, 3,
                          3).matmul(coor[paste_idx1].unsqueeze(-1)).squeeze(-1)
            bev_feat += self.voxel_pooling_v2(
                coor1, depth.view(B, N, self.D, H, W)[paste_idx1],
                tran_feat.view(B, N, self.out_channels, H, W)[paste_idx1], 
                kept.view(B, N, self.D, H, W)[paste_idx1])    
        return bev_feat

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input, paste_idx, bda_paste) = input[:10]
        x = x[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        # x = x.float()
        # mlp_input = mlp_input.float()
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        semantic_digit = x[:, self.D:self.D + 2]
        tran_feat = x[:, self.D + 2:self.D + 2 + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        semantic = semantic_digit.softmax(dim=1)
        kept = (depth >= self.depth_threshold) * (semantic[:,1:2] >= self.semantic_threshold)
        return self.view_transform(input[0][0].shape, input, depth, tran_feat, kept, paste_idx, bda_paste), \
            (depth, semantic)

    def get_loss(self, img_preds, gt_depth, gt_semantic):
        depth, semantic = img_preds
        depth_labels, semantic_labels = \
            self.get_downsampled_gt_depth_and_semantic(gt_depth, gt_semantic)
        loss_depth, loss_semantic = \
            self.get_depth_and_semantic_loss(depth_labels, depth, semantic_labels, semantic)
        return loss_depth, loss_semantic


@NECKS.register_module()
class SABEVPoolwithMSCT(SABEVPool):

    def __init__(self, head_in_channels, head_mid_channels,
                loss_depth_weight=3.0, loss_semantic_weight=25, 
                 depth_threshold=1, semantic_threshold=0.25, **kwargs):
        super(SABEVPoolwithMSCT, self).__init__(**kwargs)
        self.depth_net = MSCThead(head_in_channels, head_mid_channels, 
                                  self.D, 2, self.out_channels)

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input, paste_idx, bda_paste) = input[:10]
        
        out = self.depth_net(x, mlp_input)
        depth = out[0].softmax(dim=1)
        semantic = out[1].softmax(dim=1)
        tran_feat = out[2]
        kept = (depth >= self.depth_threshold) * (semantic[:,1:2] >= self.semantic_threshold)
        return self.view_transform(input[0][1].shape, input, depth, tran_feat, kept, paste_idx, bda_paste), \
            (out[3], out[4], out[0], out[1])
    
    def get_loss(self, img_preds, gt_depth, gt_semantic):
        depth0 = F.interpolate(img_preds[0], scale_factor=2, mode='bilinear').softmax(1)
        semantic0 = F.interpolate(img_preds[1], scale_factor=2, mode='bilinear').softmax(1)
        depth1 = img_preds[2].softmax(1)
        semantic1 = img_preds[3].softmax(1)
        depth_labels, semantic_labels = \
            self.get_downsampled_gt_depth_and_semantic(gt_depth, gt_semantic)
        loss_depth0, loss_semantic0 = \
            self.get_depth_and_semantic_loss(depth_labels, depth0, semantic_labels, semantic0)
        loss_depth1, loss_semantic1 = \
            self.get_depth_and_semantic_loss(depth_labels, depth1, semantic_labels, semantic1)
        loss_depth = (loss_depth0 + loss_depth1) / 2
        loss_semantic = (loss_semantic0 + loss_semantic1) / 2
        return loss_depth, loss_semantic
