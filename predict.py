import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import logging
from PIL import Image


from evaluate import dice_coeff, mIoU
from utils.data_loading import BasicDataset
from unet import UNet
import pdb

@torch.inference_mode()
def predict(net, dataloader, device, dir_output, mask_values, out_threshold=0.5):
    dice_score = 0
    total_cross_entropy = 0.0
    total_samples = 0

    # TODO 实现对训练好的模型测试，包括数据载入，输入网络进行预测
    net.eval()
    Path(dir_output).mkdir(parents=True, exist_ok=True)

    for batch in tqdm(dataloader, desc='Predicting', unit='batch'):
        images = batch['image']
        true_masks = batch['mask']
        filenames = batch['filename']
        extensions = batch['extension']  # 新增字段

        images = images.to(device,dtype=torch.float32)
        true_masks = true_masks.to(device,dtype=torch.long)

        batch_size = images.size(0)
        total_samples += batch_size

        # 前向传播
        outputs = net(images)

        # 计算交叉熵
        cross_entropy = F.cross_entropy(outputs, true_masks)  # 调用交叉熵函数
        total_cross_entropy += cross_entropy.item() * images.size(0)

        # 计算DICE
        true_masks_onehot = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
        mask_pred = outputs.argmax(dim=1)
        mask_pred_onehot = F.one_hot(mask_pred, net.n_classes).permute(0, 3, 1, 2).float()

        # TODO 调用dice_coeff()函数计算DICE值，调用F.cross_entropy()函数计算交叉熵cross-entropy值
        dice = dice_coeff(mask_pred_onehot[:, 1:], true_masks_onehot[:, 1:])
        dice_score += dice.item() * batch_size(0)

       # TODO 调用mask_to_image()，并保存预测mask图像至dir_output, 命名与数据原始名称相同，如：27.tif
        # 保存预测结果
        for i in range(len(filenames)):
            filename = filenames[i]
            ext = extensions[i]
            pred_mask = mask_pred[i].cpu().numpy()  # 直接使用 mask_pred
            img = mask_to_image(pred_mask, mask_values)
            output_path = Path(dir_output) / f"{filename}{ext}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format='TIFF')

    # 计算平均值
    avg_cross_entropy = total_cross_entropy / total_samples
    avg_dice = dice_score / total_samples

    return avg_dice,avg_cross_entropy

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--dir_img', default='/kaggle/input/exp4444/实验四/Pytorch-UNet-master/data/test/img/', help='path of input')
    parser.add_argument('--dir_mask', default='/kaggle/input/exp4444/实验四/Pytorch-UNet-master/data/test/mask/', help='path of mask')
    parser.add_argument('--model', '-m', default='/kaggle/input/check-exp444/checkpoint_epoch4.pth',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--output', '-o', default='/kaggle/working/exp4444/data/pred/', help='Filenames of output images')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--out_threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    dataset = BasicDataset(Path(args.dir_img), Path(args.dir_mask), args.scale)
    num = len(dataset)
    print(num)
    loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=True, **loader_args)
    dice_score, cross_entropy_score = predict(net=net, dataloader=dataloader, dir_output=args.output, out_threshold=args.out_threshold,
                   device=device, mask_values=mask_values)
    print(f'Average DICE score on test dataset: {dice_score:.4f}')
    print(f'Average Cross-Entropy on test dataset: {cross_entropy_score:.4f}')






