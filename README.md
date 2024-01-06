# Deep Learning Project - Lane Detection H-Net Improvement

This reference code is an implemtation of our project in Deep Learning course in Tel Aviv University.
This repository is **forked** from [billpsomas-
lane_detection](https://github.com/billpsomas/lane_detection) and contains code written in PyTorch to implement the LaneNet introduced in [Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/pdf/1802.05591.pdf) paper and **our imeplentation for improvement and innovation of the H-Net as documented in our [project report.](https://docs.google.com/document/d/1HAnisFuNLcUtdAft1VDcHXThCYekvkZ3olOHriSuKh8/edit?usp=sharing).**


please see (and download) nesseary files in the project's [drive directory](https://drive.google.com/drive/folders/1AmjGIgCvKWoAIeoWqXxtn3kLTXudImXu?usp=sharing).

## cloning the repository
clone or download the repository to your local machine from [here](https://github.com/tal-grossman/Lane_Detection_PyTorch_project)


## Enviorment Setup


1. Install requirements
    ```bash
    pip install -r requirements.txt
    ```

2. For running training and testing, you need to download the [TuSimple dataset](https://drive.google.com/drive/folders/1CYRruRAuypHGc6o3V1_3wo_jr-PyxiJD?usp=sharing). 
**Recommended to extract the dataset in the same folder as this repository.**
3. Prepare TUSIMPLE dataset by running the following command:
    ```bash
    python3 utils/process_train_set.py --src_dir /path/to/your/extracted/train_set
    ```
    Now TUSIMPLE folder should look like this:
    
    ```
    Lane_Detection_PyTorch
    |---TUSIMPLE
        |---Lanenet_output
        |   |--lanenet_epoch_29_batch_size_8.model
        |
        |---training
        |   |--lgt_binary_image
        |   |--gt_image
        |   |--gt_instance_image
        |
        |---txt_for_local
        |   |--test.txt
        |   |--train.txt
        |   |--val.txt
        |
        |---test_set
        |   |--clips
        |   |--test_tasks_0627.json
        |   |--test_label.json
        |   |--readme.md
        |
        |---test_clips
    ```

4. Download the the LaneNet and H-Net models trained by us and presented in our proejct report from [here](https://drive.google.com/drive/folders/1AmjGIgCvKWoAIeoWqXxtn3kLTXudImXu?usp=sharing)

## How To Run
Recommneded to run the following commands from the project's root directory.

### Training

#### Pre-Train the H-Net
as elaborated in the report, we pre-train the H-Net to get better initial weights for the H-Net by minimizing the distance to a fixed homograpy matrix coefficients.


```
python3 train_hnet_v2.py --data_set_dir ./TUSIMPLE/train_set --phase pretrain --pre_train_epochs <desired_num_of_epochs> --pre_train_save_dir <desried_output_path>
```

example:
```
python3 train_hnet_v2.py --data_set_dir /home/tal/git/University/Lane_Detection_PyTorch_project/TUSIMPLE/train_set --phase pretrain --pre_train_epochs 20 --pre_train_save_dir /home/tal/git/University/Lane_Detection_PyTorch_project/trained_models/hnet/pre_train
```

**Additional flags:** 

`--hnet_weights` - path to existing weights for the H-Net to load and continue training from.
`--batch_size` - batch size for training (default=10)

#### Train the H-Net
After pre-training the H-Net, we train the H-Net with the full loss, with L1/L2 regularization function as elaborated in the report. poly_order flag is used to determine the order of the polyfit function used for fitting the lane points in world domain.

```
python3 train_hnet_v2.py --data_set_dir ./TUSIMPLE/train_set --phase train --hnet_weights <path_to_pre_trained_weights> --train_epochs <desired_num_of_epochs> --poly_order <2/3> --regularization_type <1/2> --train_save_dir <desried_output_path>
```

example:
```
python3 train_hnet_v2.py --data_set_dir /home/tal/git/University/Lane_Detection_PyTorch_project/TUSIMPLE/train_set --phase train --hnet_weights /home/tal/git/University/Lane_Detection_PyTorch_project/trained_models/H-Net/pre_train_with_info/pretrain_hnet_epoch_20.pth --train_epochs 20 --poly_order 3 --regularization_type 2 --train_save_dir /home/tal/git/University/Lane_Detection_PyTorch_project/trained_models/hnet/train/poly_3_reg_2
```

**Additional flags:** 

`--batch_size` - batch size for training (default=10)

**Note:**: You can train the model without `--hnet_weights` flag, in this case the model will be trained from scratch. (as in the original paper).


### Evaluating/Testing (and get score) 

``` 
python3 evaluate.py --lanenet_model_path <path_to_lanenet_model> --hnet_model_path <path_to_hnet_model> --polyfit_with_ransac --evaluation_output <path_to_output_dir> --run_name <name_for_output_dir>
```

example:
```
python3 evaluate.py --lanenet_model_path /home/tal/git/University/Lane_Detection_PyTorch_project/trained_models/LaneNet/lanenet_epoch_29_batch_size_8.model --hnet_model_path /home/tal/git/University/Lane_Detection_PyTorch_project/trained_models/H-Net/train_hnet_poly_order_3_reg2/train_hnet_poly_order_3_epoch_10.pth --polyfit_with_ransac --evaluation_output /home/tal/git/University/Lane_Detection_PyTorch_project/evaluation_for_report --run_name reg2_poly_3_w_ransac
```

**Note**:

to evaluate inference pipline without RANSAC, remove the `--polyfit_with_ransac` flag.

**Additional flags:** 

`--use_pre_H` - use pre-trained H-Net model instead of running H-Net inference. (default=False)

`--poly_order` - order of the polyfit function used for fitting the lane points in world domain (default=2). if not saved already by the H-Net model. (should not happen).

#### For getting results for already evaluated models

```
python3 evaluate.py --run_results_only True --pred_lanenet_file_path <path_to_lanenet_pred_file> --pred_hnet_file_path <path_to_hnet_pred_file>

```
example:
```
python3 evaluate.py --run_results_only True --pred_lanenet_file_path /home/tal/git/University/Lane_Detection_PyTorch_project/evaluated_results/evalualte_poly3_with_reg2/pred_lanenet.json --pred_hnet_file_path /home/tal/git/University/Lane_Detection_PyTorch_project/evaluated_results/evalualte_poly3_with_reg2/pred_hnet.json
```


### Run Pipeline on any image

We can run inference on any image from, example for image in the dataset:

```
python3 lanenet_hnet_predict.py --image_path <path_to_image> --lanenet_weights <path_to_lanenet_model> --hnet_weights <path_to_hnet_model> --polyfit_with_ransac --output_path <path_to_output_dir>
```
example:
```
python3 lanenet_hnet_predict.py --image_path TUSIMPLE/test_clips/clip_1/1.jpg --lanenet_weights trained_models/LaneNet/lanenet_epoch_29_batch_size_8.model --hnet_weights trained_models/H-Net/train_hnet_poly_order_3_reg2/train_hnet_poly_order_3_epoch_10.pth --polyfit_with_ransac --output_path ./predict_example
```

**Note**:
to run inference pipline without RANSAC, remove the `--polyfit_with_ransac` flag.
