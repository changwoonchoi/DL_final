import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from miscc.config import cfg
from PIL import Image

import numpy as np
import os
import time

#################################################
# DO NOT CHANGE 
from utils.model import RNN_ENCODER, CNN_ENCODER, GENERATOR, DISCRIMINATOR
#################################################

                
class condGANTrainer(object):
    def __init__(self, output_dir, train_dataloader, test_dataloader, n_words, dataloader_for_wrong_samples=None,
                 log=None, writer=None):
        self.output_dir = output_dir
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.dataloader_for_wrong_samples = dataloader_for_wrong_samples
        
        self.batch_size = cfg.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        
        self.n_words = n_words # size of the dictionary

        self.log = log if log is not None else print
        self.writer = writer

    
    def prepare_data(self, data):
        """
        Prepares data given by dataloader
        e.g., x = Variable(x).cuda()
        """
        #################################################
        # TODO
        # this part can be different, depending on which algorithm is used
        #################################################

        imgs = data['img']
        captions = data['caps']
        captions_lens = data['cap_len']
        class_ids = data['cls_id']
        keys = data['key']
        sentence_idx = data['sent_ix']

        # sort data by the length in a decreasing order
        # the reason of sorting data can be found in https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html 
        sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            if cfg.CUDA:
                real_imgs.append(Variable(imgs[i]).cuda())
            else:
                real_imgs.append(Variable(imgs[i]))


        captions = captions[sorted_cap_indices].squeeze()
        class_ids = class_ids[sorted_cap_indices].numpy()
        keys = [keys[i] for i in sorted_cap_indices.numpy()]
        sentence_idx = sentence_idx[sorted_cap_indices].numpy()

        if cfg.CUDA:
            captions = Variable(captions).cuda()
            sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        else:
            captions = Variable(captions)
            sorted_cap_lens = Variable(sorted_cap_lens)

        return [real_imgs, captions, sorted_cap_lens, class_ids, keys, sentence_idx]
    
    def train(self):
        """
        e.g., for epoch in range(cfg.TRAIN.MAX_EPOCH):
                  for step, data in enumerate(self.train_dataloader, 0):
                      x = self.prepare_data()
                      .....
        """
        #################################################
        # TODO: Implement text guided image manipulation
        # TODO: You should remain log something during training (e.g., loss, performance, ...) with both log and writer
        # Ex. log('Loss at epoch {}: {}'.format(epoch, loss)')
        # Ex. writer.add_scalar('Loss/train', loss, epoch)

        
        #################################################

    def generate_data_for_eval(self):
        # load the text encoder model to generate images for evaluation
        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER),
                                map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.RNN_ENCODER)
        self.text_encoder.eval()

        #################################################
        # TODO
        # this part can be different, depending on which algorithm is used
        #################################################

        # load the generator model to generate images for evaluation
        self.netG = GENERATOR()
        state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR),
                                map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        for p in self.netG.parameters():
            p.requires_grad = False
        print('Load generator from:', cfg.TRAIN.GENERATOR)
        self.netG.eval()

        noise = Variable(torch.FloatTensor(self.batch_size, cfg.GAN.Z_DIM))

        if cfg.CUDA:
            self.text_encoder = self.text_encoder.cuda()
            self.netG = self.netG.cuda()
            noise = noise.cuda()

        for step, data in enumerate(self.test_dataloader, 0):
            imgs, captions, cap_lens, class_ids, keys, sent_idx = self.prepare_data(data)

            #################################################
            # TODO
            # word embedding might be returned as well
            # hidden = self.text_encoder.init_hidden(self.batch_size)
            # sent_emb = self.text_encoder(captions, cap_lens, hidden)
            # sent_emb = sent_emb.detach()
            #################################################

            noise.data.normal_(0, 1)

            #################################################
            # TODO
            # this part can be different, depending on which algorithm is used
            # the main purpose is generating synthetic images using caption embedding and latent vector (noise)
            # fake_img = self.netG(noise, sent_emb, img_emb, ...)
            fake_imgs = None
            #################################################

            # Save original img
            for j in range(self.batch_size):
                if not os.path.exists(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0])):
                    os.mkdir(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j].split('/')[0]))
                if not os.path.exists(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j].split('/')[0])):
                    os.mkdir(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j].split('/')[0]))
                if not os.path.exists(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j] + '.png')):
                    im = imgs[j].data.cpu().numpy()
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    print(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j] + '.png'))
                    im.save(os.path.join(cfg.TEST.ORIG_TEST_IMAGES, keys[j] + '.png'))

                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                print(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))
                im.save(os.path.join(cfg.TEST.GENERATED_TEST_IMAGES, keys[j] + '_{}.png'.format(sent_idx[j])))

        
    def save_model(self):
        """
        Saves models
        """
        torch.save(self.netG.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR))
        torch.save(self.text_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER))
        torch.save(self.image_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.CNN_ENCODER))
