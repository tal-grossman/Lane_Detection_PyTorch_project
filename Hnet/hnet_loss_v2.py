import enum

import torch
import numpy as np
from torch.nn.modules.loss import _Loss

from hnet_utils import hnet_transformation, PRE_H, POLY_FIT_ORDER

# Use GPU if available, else use CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

COEFF_PENALTY_DEG_TO_WEIGHTS = {
    2: torch.tensor([50, 20, 0]).to(device),
    3: torch.tensor([50, 20, 2, 0]).to(device)
}


class REG_TYPE(enum.IntEnum):
    NONE = 0
    COEFFICIENTS_L1 = 1
    COEFFICIENTS_L2 = 2


class PreTrainHnetLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, transformation_coefficient):
        return self._pre_train_loss(transformation_coefficient)

    @staticmethod
    def _pre_train_loss(transformation_coefficient):
        """
        :param transformation_coefficient: the 6 params from the HNet
        :return: the loss
        """
        # todo, I add the dim=1 to get the mean of the batch, but I'm not sure if it's correct
        # if it wasn't there, I would get scalar after the norm and the mean would be the same
        pre_loss = torch.mean(torch.norm((transformation_coefficient - PRE_H) / PRE_H, dim=1))
        return pre_loss


class HetLoss(_Loss):

    def __init__(self, regularization_type: REG_TYPE = REG_TYPE.NONE):
        super(HetLoss, self).__init__()
        self.regularization_type = regularization_type

    def forward(self, input_pts, transformation_coefficient):
        return self._hnet_loss(input_pts, transformation_coefficient)

    def _hnet_loss(self, input_pts, transformation_coefficient):
        # assert not torch.isnan(transformation_coefficient).any(), "transformation_coefficient is nan"
        # todo: handle case where transformation_coefficient is nan
        if torch.isnan(transformation_coefficient).any():
            print("in _hnet_loss(): transformation_coefficient is nan")
            print(transformation_coefficient)
            raise RuntimeError

        batch_size = input_pts.shape[0]
        single_frame_losses = []
        for i in range(batch_size):
            frame_input_pts = input_pts[i]
            frame_transformation_coefficient = transformation_coefficient[i]
            frame_loss = self.hnet_single_frame_loss(frame_input_pts, frame_transformation_coefficient,
                                                     poly_fit_order=POLY_FIT_ORDER, debug_idx = i)
            single_frame_losses.append(frame_loss)

            residual = frame_transformation_coefficient - PRE_H
            print(f'in _hnet_loss(): frame_loss: {frame_loss}, hnet_res - pre_h: {residual.tolist()}')

        loss = torch.mean(torch.stack(single_frame_losses))

        # print(f'in _hnet_loss(): batch loss: {loss}')
        # print(f'transformation_coefficient: {transformation_coefficient}')

        return loss

    def hnet_single_frame_loss(self, input_pts, transformation_coefficient, poly_fit_order: int = 2,
                               debug_idx: int = None):
        """
        :param input_pts: the points of the lane of a single image, shape: [k, 3] (k is the number of points)
        :param transformation_coefficient: the 6 params from the HNet, shape: [1, 6]
        :param poly_fit_order: the order of the polynomial
        :return single_frame_loss: the loss of the single frame
        """

        valid_pts_reshaped, H, preds_transformation_back, _, poly_coeffs = hnet_transformation(input_pts,
                                                                                               transformation_coefficient,
                                                                                               poly_fit_order)
        # compute loss between back-transformed polynomial fit and gt_pts
        single_frame_err = valid_pts_reshaped[0, :] - preds_transformation_back[0, :]
        single_frame_sq_err = torch.pow(single_frame_err, 2)
        single_frame_loss = torch.mean(single_frame_sq_err)

        if self.regularization_type == REG_TYPE.COEFFICIENTS_L1:
            single_frame_loss += torch.mean(torch.abs(torch.mul(COEFF_PENALTY_DEG_TO_WEIGHTS[poly_fit_order], poly_coeffs.transpose(0,1))))
        elif self.regularization_type == REG_TYPE.COEFFICIENTS_L2:
            single_frame_loss += torch.mean(torch.mul(COEFF_PENALTY_DEG_TO_WEIGHTS[poly_fit_order],
                                                      torch.pow(poly_coeffs.transpose(0,1), 2)))

        # if single_frame_loss > 1000:
        # PRE_H_MAT = torch.zeros(3, 3, device=device)
        # PRE_H_MAT[0, 0] = PRE_H[0]  # a
        # PRE_H_MAT[0, 1] = PRE_H[1]  # b
        # PRE_H_MAT[0, 2] = PRE_H[2]  # c
        # PRE_H_MAT[1, 1] = PRE_H[3]  # d
        # PRE_H_MAT[1, 2] = PRE_H[4]  # e
        # PRE_H_MAT[2, 1] = PRE_H[5]  # f
        # PRE_H_MAT[2, 2] = 1
        # print(f'in hnet_single_frame_loss(): single_frame_loss: {single_frame_loss}')
        # print(f'H: {H}')
        # print(f'H - PRE_H_MAT: {H - PRE_H_MAT}')

        # print(f'INDEX_DEBUG: {debug_idx}')

        return single_frame_loss
