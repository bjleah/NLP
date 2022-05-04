
import argparse
from pathlib import Path
import os

from_pretrained = os.getcwd() + "\\model-ernie-gram-zh.1"
data_dir = os.getcwd() + "\\data"
save_dir = os.getcwd()

def get_param():
    parser = argparse.ArgumentParser('classify model with ERNIE')
    parser.add_argument('--from_pretrained', type=Path, default=from_pretrained, help='pretrained model directory or tag')
    parser.add_argument('--max_seqlen', type=int, default=128, help='max sentence length, should not greater than 512')
    parser.add_argument('--bsz', type=int, default=128, help='global batch size for each optimizer step')
    parser.add_argument('--micro_bsz', type=int, default=32, help='batch size for each device. if `--bsz` > `--micro_bsz` * num_device, will do grad accumulate')
    parser.add_argument('--epoch', type=int, default=3, help='epoch')
    parser.add_argument('--data_dir', type=str, default=data_dir, help='data directory includes train / develop data')
    parser.add_argument('--use_lr_decay', action='store_true', help='if set, learning rate will decay to zero at `max_steps`')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='if use_lr_decay is set, ' 'learning rate will raise to `lr` at `warmup_proportion` * `max_steps` and decay to 0. at `max_steps`')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--inference_model_dir', type=Path, default=None, help='inference model output directory')
    parser.add_argument('--save_dir', type=Path, default=save_dir, help='model output directory')
    parser.add_argument('--max_steps', type=int, default=None, help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay, aka L2 regularizer')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='checkpoint to warm start from')
    parser.add_argument('--use_amp', action='store_true', help='only activate AMP(auto mixed precision accelatoin) on TensorCore compatible devices')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_param()
    print(args.data_dir)

