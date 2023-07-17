import torch
import torch.nn.functional as F
import numpy as np
from mmcv.runner import force_fp32

from mmdet.models import DETECTORS
from .centerpoint import CenterPoint
from .bevdet import BEVDepth4D
from torch.cuda.amp import autocast
from ...datasets.pipelines.loading import LoadAnnotationsBEVDepth

@DETECTORS.register_module()
class SABEV(BEVDepth4D):

    def __init__(self, use_bev_paste=True, bda_aug_conf=None, **kwargs):
        super(SABEV, self).__init__(**kwargs)
        self.use_bev_paste = use_bev_paste
        if use_bev_paste:
            self.loader = LoadAnnotationsBEVDepth(bda_aug_conf, None, is_train=True)

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input, paste_idx, bda_paste, img_metas=None):
        x = self.image_encoder(img)
        bev_feat, img_preds = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input, paste_idx, bda_paste])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, img_preds

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = list(self.img_backbone(imgs))
        if self.with_img_neck:
            x[1] = self.img_neck(x[1:])
            if type(x) in [list, tuple]:
                x = x[:2]
        for i in range(2):
            _, output_dim, ouput_H, output_W = x[i].shape
            x[i] = x[i].view(B, N, output_dim, ouput_H, output_W)
        return x[:2][::-1]

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.view(B, self.num_frame, N, 3, 3),
            trans.view(B, self.num_frame, N, 3),
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        if len(inputs) == 9:
            paste_idx = inputs[7]
            bda_paste = inputs[8]
        else:
            paste_idx = None
            bda_paste = None
        return imgs, rots, trans, intrins, post_rots, post_trans, bda, paste_idx, bda_paste

    def extract_img_feat(self, img, img_metas, **kwargs):

        imgs, rots, trans, intrins, post_rots, post_trans, bda, paste_idx, bda_paste = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        key_frame = True  # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, post_tran in zip(
                imgs, rots, trans, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input, paste_idx, bda_paste)
                if key_frame:
                    bev_feat, img_preds = self.prepare_bev_feat(*inputs_curr, img_metas)
                else:
                    with torch.no_grad():
                        bev_feat, _ = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
            bev_feat_list.append(bev_feat)
            key_frame = False

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], img_preds

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, img_preds = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, img_preds)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        if self.use_bev_paste:
            B = len(gt_bboxes_3d)
            paste_idx = []
            for i in range(B):
                for j in range(i, i + 1):
                    if j+1>=B: j-=B
                    paste_idx.append([i,j+1])
            
            gt_boxes_paste = []
            gt_labels_paste = []
            bda_mat_paste = []
            for i in range(len(paste_idx)):
                gt_boxes_tmp = []
                gt_labels_tmp = []
                for j in paste_idx[i]:
                    gt_boxes_tmp.append(gt_bboxes_3d[j])
                    gt_labels_tmp.append(gt_labels_3d[j])
                gt_boxes_tmp = torch.cat([tmp.tensor for tmp in gt_boxes_tmp], dim=0)
                gt_labels_tmp = torch.cat(gt_labels_tmp, dim=0)
                rotate_bda, scale_bda, flip_dx, flip_dy = self.loader.sample_bda_augmentation()
                gt_boxes_tmp, bda_rot = self.loader.bev_transform(gt_boxes_tmp.cpu(), rotate_bda, scale_bda, flip_dx, flip_dy)
                gt_boxes_tmp = gt_bboxes_3d[0].new_box(gt_boxes_tmp.cuda())
                bda_mat_paste.append(bda_rot.cuda())
                gt_boxes_paste.append(gt_boxes_tmp)
                gt_labels_paste.append(gt_labels_tmp)
            gt_bboxes_3d = gt_boxes_paste
            gt_labels_3d = gt_labels_paste
            img_inputs.append(paste_idx)
            img_inputs.append(torch.stack(bda_mat_paste))
            
        img_feats, pts_feats, img_preds = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']
        gt_semantic = kwargs['gt_semantic']
        loss_depth, loss_semantic = \
            self.img_view_transformer.get_loss(img_preds, gt_depth, gt_semantic)
        losses = dict(loss_depth=loss_depth, loss_semantic=loss_semantic)
        with autocast(False):
            losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list