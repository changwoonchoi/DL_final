import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from miscc.config import cfg
from utils.misc import weights_init, load_params, copy_G_params, mkdir_p, build_super_images
from PIL import Image
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.loss import discriminator_loss, generator_loss, KL_loss, words_loss

import numpy as np
import os
import time

#################################################
# DO NOT CHANGE 
from utils.model import RNN_ENCODER, CNN_ENCODER, GENERATOR, DISCRIMINATOR
#################################################


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['8']  # relu2_2
        model = models.vgg16()
        url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.resquires_grad = False
        print('Load pretrained model from ', url)
        self.vgg = model.features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


class condGANTrainer(object):
    def __init__(self, output_dir, train_dataloader, test_dataloader, n_words, ixtoword, dataloader_for_wrong_samples=None,
                 log=None, writer=None):
        self.output_dir = output_dir
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.dataloader_for_wrong_samples = dataloader_for_wrong_samples

        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(int(cfg.GPU_ID))
        cudnn.benchmark = True

        self.batch_size = cfg.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.num_batches = len(self.train_dataloader)
        
        self.n_words = n_words  # size of the dictionary
        self.ixtoword = ixtoword
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
        wrong_caps = data['wrong_caps']
        wrong_caps_len = data['wrong_cap_len']
        wrong_cls_id = data['wrong_cls_id']
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

        w_sorted_cap_lens, w_sorted_cap_indices = torch.sort(wrong_caps_len, 0, True)

        wrong_caps = wrong_caps[w_sorted_cap_indices].squeeze()
        wrong_cls_id = wrong_cls_id[w_sorted_cap_indices].numpy()

        if cfg.CUDA:
            wrong_caps = Variable(wrong_caps).cuda()
            w_sorted_cap_lens = Variable(w_sorted_cap_lens).cuda()
        else:
            wrong_caps = Variable(wrong_caps)
            w_sorted_cap_lens = Variable(w_sorted_cap_lens)

        return [real_imgs, captions, sorted_cap_lens, class_ids, keys, wrong_caps, w_sorted_cap_lens, wrong_cls_id, sentence_idx]

    def build_model(self):
        if cfg.TRAIN.NET_E == '':
            print('pretrained encoder not loaded')
            return
        # build VGGNet
        VGG = VGGNet()
        for p in VGG.parameters():
            p.requires_grad = False
        print("Load VGG")
        VGG.eval()
        # build text encoder, image encoder
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.require_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()
        # build generator and discriminator
        netsD = []
        netG = GENERATOR()
        if cfg.TREE.BRANCH_NUM > 0:
            netsD.append(DISCRIMINATOR(res=64))
        if cfg.TREE.BRANCH_NUM > 1:
            netsD.append(DISCRIMINATOR(res=128))
        if cfg.TREE.BRANCH_NUM > 2:
            netsD.append(DISCRIMINATOR(res=256))

        netG.apply(weights_init)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)

        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            VGG = VGG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch, VGG]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))
        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()
        return real_labels, fake_labels, match_labels

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
        text_encoder, image_encoder, netG, netsD, start_epoch, VGG = self.build_model()
        # breakpoint()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()
        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        gen_iterations = 0
        # breakpoint()
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            data_iter = iter(self.train_dataloader)

            step = 0
            # breakpoint()
            # for step, data in enumerate(self.train_dataloader):
            while step < self.num_batches:
                # 1. Prepare training data and compute text embeddings

                data = data_iter.next()
                # data = self.train_dataloader[step]
                imgs, captions, cap_lens, class_ids, keys, wrong_caps, wrong_caps_len, wrong_cls_id, _ = \
                    self.prepare_data(data)
                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sen_emb: batch_size x nef

                # matched text embeddings
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                # mismatched text embeddings
                w_words_embs, w_sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
                w_words_embs, w_setn_emb = w_words_embs.detach(), w_sent_emb.detach()

                # image features (regional, global)

                region_features, cnn_code = image_encoder(imgs[len(netsD) - 1])

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                # 2. Modify real images
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar, _, _ = netG(noise, sent_emb, words_embs, mask, cnn_code, region_features)

                # 3. update discriminator
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i], sent_emb, real_labels, fake_labels,
                                              words_embs, cap_lens, image_encoder, class_ids, w_words_embs,
                                              wrong_caps_len, wrong_cls_id)
                    errD.backward(retain_graph=True)
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD)

                # 4. updage generator
                step += 1
                gen_iterations += 1
                netG.zero_grad()
                errG_total, G_logs = generator_loss(netsD, image_encoder, fake_imgs, real_labels, words_embs, sent_emb,
                                                    match_labels, cap_lens, class_ids, VGG, imgs)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)

                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb, words_embs, mask, image_encoder, captions,
                                          cap_lens, epoch, cnn_code, region_features, imgs, name='average')
                    load_params(netG, backup_para)

                end_t = time.time()
                print('[%d/%d][%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs'
                      % (epoch, self.max_epoch, self.num_batches, errD_total, errG_total, end_t - start_t))

                if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    self.save_model(netG, avg_param_G, netsD, epoch)
            self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def generate_data_for_eval(self):
        # load the text encoder model to generate images for evaluation
        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        # state_dict = torch.load(os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER), map_location=lambda storage, loc: storage)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        self.text_encoder = self.text_encoder.cuda()
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        # print('Load text encoder from:', cfg.TRAIN.RNN_ENCODER)
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        self.text_encoder.eval()

        self.image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        self.image_encoder.load_state_dict(state_dict)
        print('Load image encoder from:', img_encoder_path)
        self.image_encoder = self.image_encoder.cuda()
        self.image_encoder.eval()

        VGG = VGGNet()
        VGG.cuda()
        VGG.eval()

        #################################################
        # TODO
        # this part can be different, depending on which algorithm is used
        #################################################

        # load the generator model to generate images for evaluation
        self.netG = GENERATOR()
        self.netG.apply(weights_init)
        self.netG.cuda()
        self.netG.eval()
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

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask, image_encoder, captions, cap_lens,
                         gen_iterations, cnn_code, region_features, real_imgs, name='current'):
        fake_imgs, attention_maps, _, _, _, _ = netG(noise, sent_emb, words_embs, mask,
                                                     cnn_code, region_features)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png' \
                           % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png' \
                       % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

        '''
        # save the real images 
        for k in range(8):
            im = real_imgs[-1][k].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            fullpath = '%s/R_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, k)
            im.save(fullpath)
        '''

    def save_model(self, netG, avg_param_G, netsD, epoch):
        """
        Saves models
        """
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(), '%s/netD%d.pth' % (self.model_dir, i))
        print('save G/Ds models')
        '''
        torch.save(self.netG.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.GENERATOR))
        torch.save(self.text_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.RNN_ENCODER))
        torch.save(self.image_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.CNN_ENCODER))
        '''
