import enum

import torch
import numpy as np
from torch.nn.modules.loss import _Loss

from Hnet.hnet_utils import hnet_transformation, PRE_H

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
        pre_loss = torch.mean(torch.norm(
            (transformation_coefficient - PRE_H) / PRE_H, dim=1))
        return pre_loss


class HetLoss(_Loss):

    def __init__(self, regularization_type: REG_TYPE = REG_TYPE.NONE):
        super(HetLoss, self).__init__()
        self.regularization_type = regularization_type

    def forward(self, input_pts, transformation_coefficient, poly_fit_order: int = 3, debug_images=None):
        return self._hnet_loss(input_pts, transformation_coefficient, poly_fit_order, debug_images)

    def _hnet_loss(self, input_pts, transformation_coefficient, poly_fit_order: int = 3, debug_images=None):
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
            debug_image = debug_images[i] if debug_images is not None else None
            frame_loss = self.hnet_single_frame_loss(frame_input_pts, frame_transformation_coefficient,
                                                     poly_fit_order=poly_fit_order, debug_idx=i, debug_image=debug_image)
            single_frame_losses.append(frame_loss)

            residual = frame_transformation_coefficient - PRE_H
            # print(f'in _hnet_loss(): frame_loss: {frame_loss}, hnet_res - pre_h: {residual.tolist()}')

        loss = torch.mean(torch.stack(single_frame_losses))

        # print(f'in _hnet_loss(): batch loss: {loss}')
        # print(f'transformation_coefficient: {transformation_coefficient}')

        return loss

    def hnet_single_frame_loss(self, input_pts, transformation_coefficient, poly_fit_order: int = 2,
                               debug_idx: int = None, debug_image=None):
        """
        :param input_pts: the points of the lane of a single image, shape: [k, 3] (k is the number of points)
        :param transformation_coefficient: the 6 params from the HNet, shape: [1, 6]
        :param poly_fit_order: the order of the polynomial
        :return single_frame_loss: the loss of the single frame
        """
        hnet_tranformation_hnet_poly2_ret = hnet_transformation(input_pts,transformation_coefficient,
                                                                                               poly_fit_order)
        valid_pts_reshaped = hnet_tranformation_hnet_poly2_ret["valid_pts_reshaped"]
        H = hnet_tranformation_hnet_poly2_ret["H"]
        preds_transformation_back = hnet_tranformation_hnet_poly2_ret["preds_transformation_back"]
        pts_projects_normalized = hnet_tranformation_hnet_poly2_ret["pts_projects_normalized"]
        poly_coeffs = hnet_tranformation_hnet_poly2_ret["w"]
        hnet_poly2_preds = hnet_tranformation_hnet_poly2_ret["preds"]
        pts_projects = hnet_tranformation_hnet_poly2_ret["pts_projects"]

        # compute loss between back-transformed polynomial fit and gt_pts
        single_frame_err = valid_pts_reshaped[0,
                                              :] - preds_transformation_back[0, :]
        single_frame_sq_err = torch.pow(single_frame_err, 2)
        single_frame_loss = torch.mean(single_frame_sq_err)

        ### debug ###
        
        # hnet_tranformation_hnet_poly3_ret = hnet_transformation(input_pts,transformation_coefficient, 3)
        # valid_pts_reshaped_poly3 = hnet_tranformation_hnet_poly3_ret["valid_pts_reshaped"]
        # H_poly3 = hnet_tranformation_hnet_poly3_ret["H"]
        # preds_transformation_back_poly3 = hnet_tranformation_hnet_poly3_ret["preds_transformation_back"]
        # pts_projects_normalized_poly3 = hnet_tranformation_hnet_poly3_ret["pts_projects_normalized"]
        # poly_coeffs_poly3 = hnet_tranformation_hnet_poly3_ret["w"]
        # hnet_poly3_preds = hnet_tranformation_hnet_poly3_ret["preds"]

        # single_frame_err_poly3 = valid_pts_reshaped_poly3[0,
        #                                         :] - preds_transformation_back_poly3[0, :]
        # single_frame_sq_err_poly3 = torch.pow(single_frame_err_poly3, 2)
        # single_frame_loss_poly3 = torch.mean(single_frame_sq_err_poly3)


        # hnet_tranformation_pre_h_poly2_ret = hnet_transformation(input_pts, PRE_H, 2)
        # valid_pts_reshaped_pre_h = hnet_tranformation_pre_h_poly2_ret["valid_pts_reshaped"]
        # H_pre_h = hnet_tranformation_pre_h_poly2_ret["H"]
        # preds_transformation_back_pre_h = hnet_tranformation_pre_h_poly2_ret["preds_transformation_back"]
        # pts_projects_normalized_pre_h = hnet_tranformation_pre_h_poly2_ret["pts_projects_normalized"]
        # poly_coeffs_pre_h = hnet_tranformation_pre_h_poly2_ret["w"]
        # pre_h_poly2_preds = hnet_tranformation_pre_h_poly2_ret["preds"]
        # pts_projects_pre_h = hnet_tranformation_pre_h_poly2_ret["pts_projects"]

        # single_frame_err_pre_h = valid_pts_reshaped_pre_h[0,
        #                                         :] - preds_transformation_back_pre_h[0, :]
        # single_frame_sq_err_pre_h = torch.pow(single_frame_err_pre_h, 2)
        # single_frame_loss_pre_h = torch.mean(single_frame_sq_err_pre_h)


        # hnet_tranformation_pre_h_poly3_ret = hnet_transformation(input_pts, PRE_H, 3)
        # valid_pts_reshaped_pre_h_poly3 = hnet_tranformation_pre_h_poly3_ret["valid_pts_reshaped"]
        # H_pre_h_poly3 = hnet_tranformation_pre_h_poly3_ret["H"]
        # preds_transformation_back_pre_h_poly3 = hnet_tranformation_pre_h_poly3_ret["preds_transformation_back"]
        # pts_projects_normalized_pre_h_poly3 = hnet_tranformation_pre_h_poly3_ret["pts_projects_normalized"]
        # poly_coeffs_pre_h_poly3 = hnet_tranformation_pre_h_poly3_ret["w"]
        # pre_h_poly3_preds = hnet_tranformation_pre_h_poly3_ret["preds"]

        # single_frame_err_pre_h_poly3 = valid_pts_reshaped_pre_h_poly3[0,
        #                                         :] - preds_transformation_back_pre_h_poly3[0, :]
        # single_frame_sq_err_pre_h_poly3 = torch.pow(single_frame_err_pre_h_poly3, 2)
        # single_frame_loss_pre_h_poly3 = torch.mean(single_frame_sq_err_pre_h_poly3)

        
        # import matplotlib.pyplot as plt
        # type_to_color_dict = {"gt": "green", "hnet_poly2": "blue", "hnet_poly3": "orange", "pre_h_poly2": "red", "pre_h_poly3": "purple"}
        # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # label_hnet_poly2 = f"hnet_poly2, loss: {single_frame_loss.item():.3f}"
        # label_pre_h_poly2 = f"pre_h_poly2, loss: {single_frame_loss_pre_h.item():.3f}"
        # label_hnet_poly3 = f"hnet_poly3, loss: {single_frame_loss_poly3.item():.3f}"
        # label_pre_h_poly3 = f"pre_h_poly3, loss: {single_frame_loss_pre_h_poly3.item():.3f}"


        # # First subplot: Image with Lanes Drawing
        # axes[0].imshow(debug_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8), cmap='viridis')
        # axes[0].scatter(valid_pts_reshaped[0, :].cpu().numpy(), valid_pts_reshaped[1, :].cpu().numpy(), s=5, 
        #         label='Valid Points', color=type_to_color_dict["gt"])
        # axes[0].scatter(preds_transformation_back[0, :].cpu().detach().numpy(), 
        #         preds_transformation_back[1, :].cpu().detach().numpy(), 
        #         s=8, label=label_hnet_poly2, color=type_to_color_dict['hnet_poly2'], alpha=0.5)
        # axes[0].scatter(preds_transformation_back_pre_h[0, :].cpu().detach().numpy(), 
        #         preds_transformation_back_pre_h[1, :].cpu().detach().numpy(), 
        #         s=8, label=label_pre_h_poly2, color=type_to_color_dict['pre_h_poly2'], alpha=0.5)
        # axes[0].scatter(preds_transformation_back_poly3[0, :].cpu().detach().numpy(),
        #         preds_transformation_back_poly3[1, :].cpu().detach().numpy(),
        #         s=8, label=label_hnet_poly3, color=type_to_color_dict['hnet_poly3'], alpha=0.5)
        # axes[0].scatter(preds_transformation_back_pre_h_poly3[0, :].cpu().detach().numpy(),
        #         preds_transformation_back_pre_h_poly3[1, :].cpu().detach().numpy(),
        #         s=8, label=label_pre_h_poly3, color=type_to_color_dict['pre_h_poly3'], alpha=0.5) 
        # axes[0].set_title('Image with Lanes Drawing')
        # axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)


        # # Second subplot: Top down scatter
        # axes[1].scatter(pts_projects[0, :].cpu().detach().numpy(),
        #     pts_projects[1, :].cpu().detach().numpy(),
        #     s=20, label='hnet_proj', color="black", marker='s')
        # axes[1].scatter(pts_projects_pre_h[0, :].cpu().detach().numpy(),
        #     pts_projects_pre_h[1, :].cpu().detach().numpy(),
        #     s=20, label='preh_proj', color="grey", marker='s')
        # axes[1].scatter(hnet_poly2_preds[0, :].cpu().detach().numpy(), 
        #         hnet_poly2_preds[1, :].cpu().detach().numpy(), 
        #         s=5, label=label_hnet_poly2, color=type_to_color_dict['hnet_poly2'])
        # axes[1].scatter(hnet_poly3_preds[0, :].cpu().detach().numpy(),
        #         hnet_poly3_preds[1, :].cpu().detach().numpy(),
        #         s=5, label=label_hnet_poly3, color=type_to_color_dict['hnet_poly3'])
        # axes[1].scatter(pre_h_poly2_preds[0, :].cpu().detach().numpy(), 
        #         pre_h_poly2_preds[1, :].cpu().detach().numpy(),
        #         s=5, label=label_pre_h_poly2, color=type_to_color_dict['pre_h_poly2'])
        # axes[1].scatter(pre_h_poly3_preds[0, :].cpu().detach().numpy(),
        #         pre_h_poly3_preds[1, :].cpu().detach().numpy(),
        #         s=5, label=label_pre_h_poly3, color=type_to_color_dict['pre_h_poly3'])
        # axes[1].legend()
        # axes[1].set_title('Top Down Scatter preds')
        # plt.tight_layout()
        # plt.show()

        ### debug ###


        if self.regularization_type == REG_TYPE.COEFFICIENTS_L1:
            single_frame_loss += torch.mean(torch.abs(torch.mul(
                COEFF_PENALTY_DEG_TO_WEIGHTS[poly_fit_order], poly_coeffs.transpose(0, 1))))
        elif self.regularization_type == REG_TYPE.COEFFICIENTS_L2:
            single_frame_loss += torch.mean(torch.mul(COEFF_PENALTY_DEG_TO_WEIGHTS[poly_fit_order],
                                                      torch.pow(poly_coeffs.transpose(0, 1), 2)))

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
