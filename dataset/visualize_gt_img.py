import os
import cv2
import json
import argparse
import numpy as np

h_samples = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
             430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]

TEST_IMAGE_NUMBER_IN_SET = 20


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', type=str,
                        help='for example 1492627471490503361_0', required=True)
    parser.add_argument('--test_set_dir', type=str, help='the test set dir path',
                        required=False, default="TUSIMPLE/test_set/")
    parser.add_argument('--output_path', type=str,
                        help='The output dir to save the predict result', required=True)
    return parser.parse_args()


def visualize_gt_lanes(image_name, test_set_dir, otuput_path):
    test_json_path = os.path.join(test_set_dir, "test_label.json")

    gt_lanes = None
    # read json and look for image_name in one of the json lines
    with open(test_json_path, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            if image_name in json_line["raw_file"]:
                image_path_in_test_set = json_line["raw_file"]
                # found the line
                gt_lanes = json_line["lanes"]
                gt_h_samples = json_line["h_samples"]
                break
    if gt_lanes is None:
        raise Exception("image_name not found in json")

    # read image
    image_path = os.path.join(test_set_dir, image_path_in_test_set)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # draw lanes on image in green
    for gt_lane in gt_lanes:
        relevant_gt_lane_indices = np.where(np.array(gt_lane) > 0)[0]
        for relavent_idx_in_lane in relevant_gt_lane_indices:
            cv2.circle(image, (gt_lane[relavent_idx_in_lane],
                       h_samples[relavent_idx_in_lane]), 8, (0, 255, 0), -1)
    # resize to 512 * 256
    image = cv2.resize(image, dsize=(512, 256),
                       interpolation=cv2.INTER_LINEAR)
            
    # save image
    output_image_name = f"gt_{image_name}.png"
    output_image_path = os.path.join(otuput_path, output_image_name)
    cv2.imwrite(output_image_path, image)


if __name__ == '__main__':
    args = parse_args()
    image_name = args.image_name
    test_set_dir = args.test_set_dir
    output_path = args.output_path
    visualize_gt_lanes(image_name, test_set_dir, output_path)
