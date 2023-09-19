import torch


def hnet_transformation(input_pts, transformation_coefficient, poly_fit_order: int = 3, device: str = 'cuda'):
    """
    :param input_pts: list size N (batch size ) of [k, 3] (k is the number of points, k is different for each sample)
    :param transformation_coefficient: the 6 params from the HNet, shape: [N, 6]
    :param poly_fit_order: the order of the polynomial
    :param device: the device to use
    :return
    - x_pred_transformation_back: the predicted and back-projected points of the lane, shape: [N, k, 3]
    - the transformation matrix (H), shape: [N, 3, 3]
    - the projected and normalized points of the lane, shape: [N, k, 3]
    """
    # 1. create H matrix from transformation_coefficient
    N = transformation_coefficient.shape[0]
    H = torch.zeros(N, 3, 3, device=device)

    # assign the h_prediction to the H matrix
    H[:, 0, 0] = transformation_coefficient[:, 0]  # a
    H[:, 0, 1] = transformation_coefficient[:, 1]  # b
    H[:, 0, 2] = transformation_coefficient[:, 2]  # c
    H[:, 1, 1] = transformation_coefficient[:, 3]  # d
    H[:, 1, 2] = transformation_coefficient[:, 4]  # e
    H[:, 2, 1] = transformation_coefficient[:, 5]  # f
    H[:, -1, -1] = 1
    H = H.type(torch.FloatTensor).to(device)

    # Initialize lists to store results
    all_x_preds = []
    all_preds = []
    all_preds_transformed_back = []
    all_pts_normalized = []

    for i in range(N):

        # 2. transform input_pts using H matrix
        # Reshape gt_pts to have shape (3, k) for broadcasting
        pts_reshaped = input_pts[i].permute(1, 0)

        # todo: getting a runtime error here
        pts_reshaped = pts_reshaped.type(torch.FloatTensor).to(device)
        pts_projects = torch.matmul(H[i], pts_reshaped)  # (3, k)

        # Extra returns for use
        pts_projects_normalized = pts_projects / pts_projects[2, :]
        all_pts_normalized.append(pts_projects_normalized)

        # filter points that has 0 in their z coordinate
        valid_points_indices = torch.where(pts_projects[2, :] != 0)[0]
        pts_projects_filtered = pts_projects[:, valid_points_indices]

        # X = pts_projects[i, 0, :] / pts_projects[i, 2, :]
        # Y = pts_projects[i, 1, :] / pts_projects[i, 2, :]

        X = pts_projects_filtered[0, :] / pts_projects_filtered[2, :]
        Y = pts_projects_filtered[1, :] / pts_projects_filtered[2, :]

        Y_One = torch.ones_like(Y)

        if poly_fit_order == 2:
            Y_stack = torch.stack([torch.pow(Y, 2), Y, Y_One], dim=1)
        elif poly_fit_order == 3:
            Y_stack = torch.stack([torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One], dim=1)
        else:
            raise ValueError('Unknown order', poly_fit_order)

        w = torch.matmul(torch.matmul(torch.inverse(torch.matmul(Y_stack.transpose(0, 1), Y_stack)),
                                      Y_stack.transpose(0, 1)), X.unsqueeze(1))

        x_preds = torch.matmul(Y_stack, w)
        # preds = torch.transpose(torch.stack([torch.squeeze(x_preds, -1) * pts_projects[i, 2, :],
        #                                      Y * pts_projects[i, 2, :], pts_projects[i, 2, :]], dim=1), 0, 1)
        preds = torch.transpose(torch.stack([torch.squeeze(x_preds, -1) * pts_projects_filtered[2, :],
                                             Y * pts_projects_filtered[2, :],
                                             pts_projects_filtered[2, :]], dim=1), 0, 1)

        # all_x_preds.append(x_preds)
        # all_preds.append(preds)

        pred_transformation_back_per_iteration = torch.matmul(torch.inverse(H[i]), preds)
        all_preds_transformed_back.append(pred_transformation_back_per_iteration)

    # Stack the results from all iterations
    # x_preds_stack = torch.stack(all_x_preds, dim=0)
    # preds_projects_stack = torch.stack(all_preds, dim=0)

    # Transform polynomial fit back using H matrix
    # pred_transformation_back = torch.matmul(torch.inverse(H), preds_projects_stack)

    return all_preds_transformed_back, H, all_pts_normalized

# todo: either fix this function to supoprt batch all at once
# use the function above which loops over the batchs

# def hnet_transformation(input_pts, transformation_coefficient, poly_fit_order: int = 3, device: str = 'cuda'):
#     """
#     :param input_pts: the points of the lane, shape: [N, k, 3] (k is the number of points, N is the batch size)
#     :param transformation_coefficient: the 6 params from the HNet, shape: [N, 6]
#     :param poly_fit_order: the order of the polynomial
#     :param device: the device to use
#     :return
#     - x_pred_transformation_back: the predicted and back-projected points of the lane, shape: [N, k, 3]
#     - the transformation matrix (H), shape: [N, 3, 3]
#     - the projected and normalized points of the lane, shape: [N, k, 3]
#     """
#     # 1. create H matrix from transformation_coefficient
#     N = transformation_coefficient.shape[0]
#     H = torch.zeros(N, 3, 3, device=device)

#     # assign the h_prediction to the H matrix
#     H[:, 0, 0] = transformation_coefficient[:, 0]  # a
#     H[:, 0, 1] = transformation_coefficient[:, 1]  # b
#     H[:, 0, 2] = transformation_coefficient[:, 2]  # c
#     H[:, 1, 1] = transformation_coefficient[:, 3]  # d
#     H[:, 1, 2] = transformation_coefficient[:, 4]  # e
#     H[:, 2, 1] = transformation_coefficient[:, 5]  # f
#     H[:, -1, -1] = 1
#     H = H.type(torch.FloatTensor).to(device)

#     # 2. transform input_pts using H matrix
#     # Reshape gt_pts to have shape (N, 3, k) for broadcasting
#     pts_reshaped = input_pts.permute(0, 2, 1)

#     # todo: getting a runtime error here
#     pts_reshaped = pts_reshaped.type(torch.FloatTensor).to(device)
#     pts_projects = torch.matmul(H, pts_reshaped)

#     # 3. compute polynomial fit of transformed input_pts
#     X = torch.transpose(pts_projects[:, 0, :] / pts_projects[:, 2, :], 0, 1)
#     Y = torch.transpose(pts_projects[:, 1, :] / pts_projects[:, 2, :], 0, 1)
#     Y_One = torch.ones_like(Y)
#     if poly_fit_order == 2:
#         Y_stack = torch.stack([torch.pow(Y, 2), Y, Y_One], dim=1)
#     elif poly_fit_order == 3:
#         Y_stack = torch.stack([torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One], dim=1)
#     else:
#         raise ValueError('Unknown order', poly_fit_order)

#     w = torch.matmul(torch.matmul(torch.pinverse(torch.matmul(Y_stack.transpose(0, 1), Y_stack)),
#                                   Y_stack.transpose(0, 1)), X.unsqueeze(1))

#     x_preds = torch.matmul(Y_stack, w)
#     # todo: check if the de-normalization is correct
#     preds = torch.transpose(torch.stack([torch.squeeze(x_preds, -1) * pts_projects[:, 2, :],
#                                          Y * pts_projects[2, :], pts_projects[2, :]], dim=1), 0, 1)

#     # 4. transform polynomial fit back using H matrix
#     x_pred_transformation_back = torch.matmul(torch.matrix_inverse(H), preds)

#     # extra returns for use to use
#     pts_projects_normalized = pts_projects / pts_projects[2, :]

#     return x_pred_transformation_back, H, pts_projects_normalized
