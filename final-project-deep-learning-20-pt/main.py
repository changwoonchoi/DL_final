import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os, nltk
import numpy as np
import argparse

from miscc.config import cfg, cfg_from_file
import pprint
import datetime
import dateutil.tz

from torch.utils.tensorboard import SummaryWriter
from utils.log import create_logger
from utils.data_utils import CUBDataset
from utils.trainer import condGANTrainer as trainer


# Set a config file as 'train_birds.yml' in training, as 'eval_birds.yml' for evaluation
parser = argparse.ArgumentParser(description='xxx')
parser.add_argument('--config', type=str, default='cfg/train_birds.yml')
args = parser.parse_args()
cfg_from_file(args.config)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

# Set directories and logger
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
output_dir = 'sample/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
log_dir = os.path.join(cfg.LOG_DIR, '%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp))
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, 'train.log')
log, logclose = create_logger(log_filename=log_filename)
writer = SummaryWriter(log_dir=log_dir)

with open(log_filename, 'w+') as logFile:
        pprint.pprint(cfg, logFile)

log('')
log('============================================================================')
log('')
log('Dataset:')
image_transform = transforms.Compose([
    transforms.Resize((128, 128))
])

train_dataset = CUBDataset(cfg.DATA_DIR, transform=image_transform, split='train')
test_dataset = CUBDataset(cfg.DATA_DIR, transform=image_transform, split='test')

log(f'\ttrain data directory:\n{train_dataset.split_dir}')
log(f'\ttest data directory:\n{test_dataset.split_dir}\n')

log(f'\t# of train filenames:{train_dataset.filenames.shape}')
log(f'\t# of test filenames:{test_dataset.filenames.shape}\n')

log(f'\texample of filename of train image:{train_dataset.filenames[0]}')
log(f'\texample of filename of test image:{test_dataset.filenames[0]}\n')

log(f'\texample of caption and its ids:\n{train_dataset.captions[0]}\n{train_dataset.captions_ids[0]}\n')
log(f'\texample of caption and its ids:\n{test_dataset.captions[0]}\n{test_dataset.captions_ids[0]}\n')

log(f'\t# of train captions:{np.asarray(train_dataset.captions).shape}')
log(f'\t# of test captions:{np.asarray(test_dataset.captions).shape}\n')

log(f'\t# of train caption ids:{np.asarray(train_dataset.captions_ids).shape}')
log(f'\t# of test caption ids:{np.asarray(test_dataset.captions_ids).shape}\n')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
        drop_last=False, shuffle=False, num_workers=int(cfg.WORKERS))

algo = trainer(output_dir, train_dataloader, test_dataloader, train_dataset.n_words, log=log, writer=writer)

if cfg.TRAIN.FLAG:
    algo.train()
else:
    algo.generate_data_for_eval()


logclose()
