# convert_to_onxx.py
# Dan Popp
# 12/2/21
#
# This file will take a pytorch model and convert it to an ONXX model, so it can be imported into Keras
import argparse
import os

import torch
from torch.autograd import Variable

from timm import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, metavar='PATH', required=True,
                        help='The checkpoint to load the model weights from')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory to save the onxx file')

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                        help='Name of model to train (default: "countception"')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--bn-tf', action='store_true', default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')
    parser.add_argument('--new-layers', type=int, default=1, metavar='N',
                        help='number of layers to reinitialize when fine-tuning (default: 0)')
    args = parser.parse_args()

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.checkpoint,
        new_layers=args.new_layers)

    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    output_name = os.path.join(args.output_dir, '%s.onxx' % checkpoint_name)
    dummy = Variable(torch.randn((32, 3, 299, 299)))
    torch.onnx.export(model, dummy, output_name)


if __name__ == '__main__':
    main()
