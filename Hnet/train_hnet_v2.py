import os
import time
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# from torch.nn.utils import clip_grad_norm_

from hnet_model import HNet, Resnet_HNet
from hnet_data_processor import TusimpleForHnetDataSet
from hnet_loss_v2 import PreTrainHnetLoss, HetLoss, REG_TYPE
from hnet_utils import save_loss_to_pickle, draw_images, plot_loss, PRE_H

torch.autograd.set_detect_anomaly(True)

PRE_TRAIN_LEARNING_RATE = 1e-4
TRAIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0002
PRINT_EVERY_N_BATCHES = 10  # 100

PRE_TRAIN_DIR = 'pre_train'
TRAIN_DIR = 'train'
IMAGES_DIR = 'images'
PLOT_DIR = 'plots'
WEIGHTS_DIR = 'weights'

# Use GPU if available, else use CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set_dir', type=str, help='The origin path of unzipped tusimple train dataset',
                        default='~/Downloads/train_set')
    parser.add_argument('--batch_size', type=int, help='The batch size of the dataset',
                        default=10)
    parser.add_argument('--phase', type=str,
                        help='The phase is train, pretrain or full_train', default='pretrain')
    parser.add_argument('--hnet_weights', type=str,
                        help='The hnet model weights path', required=False)

    # pre train phase arguments
    parser.add_argument('--pre_train_epochs', type=int,
                        help='The pre train epochs', default=5)
    parser.add_argument('--pre_train_save_dir', type=str, help='The pre train save dir',
                        default=f"./{PRE_TRAIN_DIR}")

    # train phase arguments
    parser.add_argument('--train_epochs', type=int,
                        help='The train epochs', default=5)
    parser.add_argument('--train_save_dir', type=str, help='The train hnet save dir',
                        default=f"./{TRAIN_DIR}")
    return parser.parse_args()


def train(args):
    # Define the batch size
    batch_size = args.batch_size

    # Build train set
    train_set = TusimpleForHnetDataSet(set_directory=args.data_set_dir, flag='train')
    print("train_set length {}".format(len(train_set)))
    validation_set = TusimpleForHnetDataSet(set_directory=args.data_set_dir, flag='validation')
    print("validation_set length {}".format(len(validation_set)))

    # Define DataLoaders
    data_loader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    data_loader_validation = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Define the model
    hnet_model = HNet()
    # hnet_model = Resnet_HNet()
    hnet_model.to(device)

    assert args.phase in ['pretrain', 'train',
                          'full_train'], "phase must be pretrain, train or full_train"

    if args.phase == 'pretrain':
        pre_train_hnet(args, data_loader_train, hnet_model)
    elif args.phase == 'train':
        train_hnet(args, data_loader_train, data_loader_validation, hnet_model)
    else:
        pre_train_hnet(args, data_loader_train, hnet_model)
        train_hnet(args, data_loader_train, data_loader_validation, hnet_model)


def train_hnet(args, data_loader_train, data_loader_validation, hnet_model):
    # Define the optimizer
    params = [p for p in hnet_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=TRAIN_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    num_epochs = args.train_epochs

    if args.hnet_weights is not None:
        hnet_model.load_state_dict(torch.load(args.hnet_weights))
        print("Load train hnet weights success")
    else:
        print("No train hnet weights")

    hnet_loss_function = HetLoss(REG_TYPE.NONE)
    # hnet_loss_function = HetLoss(regularization_type=REG_TYPE.COEFFICIENTS_L1)
    # hnet_loss_function = HetLoss(regularization_type=REG_TYPE.COEFFICIENTS_L2)

    # create weights directory
    weights_dir_path = os.path.join(args.train_save_dir, WEIGHTS_DIR)
    os.makedirs(weights_dir_path, exist_ok=True)
    images_dir_path = os.path.join(args.train_save_dir, IMAGES_DIR)
    os.makedirs(images_dir_path, exist_ok=True)
    plot_dir_path = os.path.join(args.train_save_dir, PLOT_DIR)
    os.makedirs(plot_dir_path, exist_ok=True)

    epochs_loss = []
    epochs_loss_validation, epochs_loss_validation_fixed = [], []
    for epoch in range(1, num_epochs + 1):
        # train one epoch
        mean_epoch_train_loss = train_one_epoch(data_loader_train, epoch, num_epochs,
                                                hnet_model, optimizer, hnet_loss_function,
                                                enable_draw_images=True,
                                                images_output_path=images_dir_path)
        epochs_loss.append(mean_epoch_train_loss)

        # evaluate one epoch
        mean_epoch_eval_hnet_loss = eval_one_epoch(data_loader_validation, epoch, num_epochs, hnet_loss_function,
                                                   eval_fixed_hnet_matrix=False,
                                                   hnet_model=hnet_model,
                                                   fixed_hnet_matrix=None)
        epochs_loss_validation.append(mean_epoch_eval_hnet_loss)

        mean_epoch_eval_fixed_h_loss = eval_one_epoch(data_loader_validation, epoch, num_epochs, hnet_loss_function,
                                                      eval_fixed_hnet_matrix=True,
                                                      fixed_hnet_matrix=PRE_H)
        epochs_loss_validation_fixed.append(mean_epoch_eval_fixed_h_loss)

        # save weights every 1 epoch
        if epoch % 1 == 0:
            file_path = os.path.join(
                weights_dir_path, f'{args.phase}_hnet_epoch_{epoch}.pth')
            torch.save(hnet_model.state_dict(), file_path)

            # plot loss over epochs and save
            plot_loss(epochs_loss, title='train HNet Loss', output_path=plot_dir_path)
            plot_loss(epochs_loss_validation, title='validation HNet Loss', output_path=plot_dir_path)
            plot_loss(epochs_loss_validation_fixed, title='validation HNet Loss with fixed H',
                      output_path=plot_dir_path)

    # save loss list to a pickle file
    save_loss_to_pickle(epochs_loss, pickle_file_path=os.path.join(plot_dir_path, 'train_hnet_loss.pkl'))
    save_loss_to_pickle(epochs_loss_validation,
                        pickle_file_path=os.path.join(plot_dir_path, '/train_hnet_validation_loss.pkl'))
    save_loss_to_pickle(epochs_loss_validation_fixed,
                        pickle_file_path=os.path.join(plot_dir_path, '/train_hnet_validation_loss_fixed.pkl'))


def train_one_epoch(data_loader_train, epoch, epochs, hnet_model, optimizer, hnet_loss_function,
                    enable_draw_images=False, images_output_path=None):
    start_time = time.time()
    curr_epoch_loss_list = []
    hnet_model.train()
    is_valid_batch_loss = False
    for i, (gt_images, gt_lane_points) in enumerate(data_loader_train):
        gt_images = Variable(gt_images).to(device).type(torch.float32)
        gt_lane_points = Variable(gt_lane_points).to(device)

        # todo: filter bad input points:
        # something like this: points = lane_points[lane_points[:, 2] > 0]
        # the issue is we can't do it for all the batch at once since the number of  valid filter
        # points is different for every batch
        # possible solutions:
        # 1. maybe to take only lanes were we have at least 40 valid points and let that be the fix number of points for every batch
        # 2. run loss batch by batch and not all at once, than average the loss over the batches

        optimizer.zero_grad()
        transformation_coefficient = hnet_model(gt_images)
        loss = hnet_loss_function(gt_lane_points, transformation_coefficient)

        # loss.backward()

        if loss.item() < 1000:
            is_valid_batch_loss = True
            loss.backward()
            optimizer.step()
            curr_epoch_loss_list.append(loss.item())

        # clip_grad_norm_(hnet_model.parameters(), max_norm=1.0)

        # for param in hnet_model.parameters():
        #     if torch.isnan(param.grad).any():
        #         print("NaN gradients detected!")

        # optimizer.step()

        # curr_epoch_loss_list.append(loss.item())

        # draw_images(gt_lane_points[i], gt_images[i],
        #             transformation_coefficient[0], 'train', i,
        #             output_path=images_output_path)

        if (i + 1) % PRINT_EVERY_N_BATCHES == 0 and is_valid_batch_loss:
            print('Train: Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, Time: {:.4f}s'
                  .format(epoch, epochs, i + 1, len(data_loader_train), loss.item(),
                          time.time() - start_time))
            start_time = time.time()  # todo time now takes the time of the validation phase as well so it's not accurate

    mean_epoch_loss = np.mean(curr_epoch_loss_list)

    # plot loss over batches
    if enable_draw_images:
        draw_images(gt_lane_points[0], gt_images[0],
                    transformation_coefficient[0], 'train', epoch,
                    output_path=images_output_path)

    return mean_epoch_loss


def eval_one_epoch(data_loader_eval, epoch, epochs, hnet_loss_function, eval_fixed_hnet_matrix=False,
                   hnet_model=None, fixed_hnet_matrix=None):
    start_time = time.time()
    curr_epoch_loss_list = []

    if eval_fixed_hnet_matrix:
        assert fixed_hnet_matrix is not None, "fixed_hnet_matrix is None but eval_fixed_hnet_matrix is True"
        mode_str = 'fixed H'
    else:
        assert hnet_model is not None, "hnet_model is None but eval_fixed_hnet_matrix is False"
        hnet_model.eval()
        hnet_model.to(device)
        mode_str = 'Hnet'

    with torch.no_grad():
        for i, (gt_images, gt_lane_points) in enumerate(data_loader_eval):
            gt_images = Variable(gt_images).to(device).type(torch.float32)
            gt_lane_points = Variable(gt_lane_points).to(device)

            # todo: filter bad input points:
            # something like this: points = lane_points[lane_points[:, 2] > 0]
            # the issue is we can't do it for all the batch at once since the number of  valid filter
            # points is different for every batch
            # possible solutions:
            # 1. maybe to take only lanes were we have at least 40 valid points and let that be the fix number of points for every batch
            # 2. run loss batch by batch and not all at once, than average the loss over the batches

            if eval_fixed_hnet_matrix:
                transformation_coefficient = fixed_hnet_matrix.repeat(gt_images.shape[0], 1)
            else:
                transformation_coefficient = hnet_model(gt_images)

            loss = hnet_loss_function(gt_lane_points, transformation_coefficient)

            # todo: handle cases of nan Hnet output
            if loss == -1:
                continue

            curr_epoch_loss_list.append(loss.item())

            if (i + 1) % PRINT_EVERY_N_BATCHES == 0:
                print('Eval {}: Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, Time: {:.4f}s'
                      .format(mode_str, epoch, epochs, i + 1, len(data_loader_eval), loss.item(),
                              time.time() - start_time))
                start_time = time.time()  # todo time now takes the time of the validation phase as well so it's not accurate

        # draw_images(gt_lane_points[0], gt_images[0], transformation_coefficient[0], i)
    mean_epoch_loss = np.mean(curr_epoch_loss_list)

    return mean_epoch_loss


def pre_train_hnet(args, data_loader_train, hnet_model):
    # Define the optimizer
    params = [p for p in hnet_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=PRE_TRAIN_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epochs = args.pre_train_epochs

    if args.hnet_weights is not None:
        hnet_model.load_state_dict(torch.load(args.hnet_weights))
        print("Load pretrain hnet weights success")
    else:
        print("No pretrain hnet weights")
    pre_train_loss = PreTrainHnetLoss()

    # create weights directory
    weights_dir_path = os.path.join(args.pre_train_save_dir, WEIGHTS_DIR)
    os.makedirs(weights_dir_path, exist_ok=True)
    images_dir_path = os.path.join(args.pre_train_save_dir, IMAGES_DIR)
    os.makedirs(images_dir_path, exist_ok=True)
    plot_dir_path = os.path.join(args.pre_train_save_dir, PLOT_DIR)
    os.makedirs(plot_dir_path, exist_ok=True)

    epochs_loss = []
    gt_lane_points = None
    gt_images = None
    transformation_coefficient = None
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        curr_epoch_loss_list = []
        for i, (gt_images, gt_lane_points) in enumerate(data_loader_train):
            gt_images = Variable(gt_images).to(device).type(torch.float32)
            gt_lane_points = Variable(gt_lane_points).to(device)

            optimizer.zero_grad()
            transformation_coefficient = hnet_model(gt_images)
            loss = pre_train_loss(transformation_coefficient)
            loss.backward()
            optimizer.step()

            curr_epoch_loss_list.append(loss.item())

            if (i + 1) % PRINT_EVERY_N_BATCHES == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}s'
                      .format(epoch, epochs, i + 1, len(data_loader_train), loss.item(),
                              time.time() - start_time))
                start_time = time.time()

        epochs_loss.append(np.mean(curr_epoch_loss_list))

        if epoch % 1 == 0:
            # 1. save the model
            # 2. draw the images
            # 3. plot the loss
            file_path = os.path.join(weights_dir_path, f'{args.phase}_hnet_epoch_{epoch}.pth')
            torch.save(hnet_model.state_dict(), file_path)
            draw_images(gt_lane_points[0], gt_images[0],
                        transformation_coefficient[0], f'{args.phase}', epoch, images_dir_path)
            # plot loss over epochs and save
            plot_loss(epochs_loss, title='Pretrain HNet Loss', output_path=plot_dir_path)
    # save loss list to a pickle file
    save_loss_to_pickle(epochs_loss, pickle_file_path=os.path.join(plot_dir_path, '/pre_train_hnet_loss.pkl'))


if __name__ == '__main__':
    # plot_loss()
    init_seeds()
    args = parse_args()
    train(args)
