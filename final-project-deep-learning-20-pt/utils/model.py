import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg
from utils.attention import SpatialAttentionGeneral as SPATIAL_ATT
from utils.attention import ChannelAttention as CHANNEL_ATT
import pdb

#######################################################################################################
# DO NOT CHANGE
class RNN_ENCODER(nn.Module):
#######################################################################################################
    def __init__(self, ntoken, ninput=300, drop_prob=0.5, nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken        # size of the dictionary
        self.ninput = ninput        # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers      # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN.TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def define_module(self):
        '''
        e.g., nn.Embedding, nn.Dropout, nn.LSTM
        '''
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden, self.nlayers, batch_first=True, dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions, batch_size, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions, batch_size, self.nhidden).zero_()))
        else:
            raise NotImplementedError

    def forward(self, captions, cap_lens, hidden, mask=None):
        '''
        1. caption -> embedding
        2. pack_padded_sequence (embedding, cap_lens)
        3. for rnn, hidden is used
        4. sentence embedding should be returned
        '''
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        words_emb = output.transpose(1, 2)
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            raise NotImplementedError
        sent_emb = sent_emb.view((-1, self.nhidden * self.num_directions))
        return words_emb, sent_emb

#######################################################################################################
# DO NOT CHANGE
class CNN_ENCODER(nn.Module):
#######################################################################################################
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        # model = models.inception_v3(init_weights=False)
        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # this is the image size
        # x.shape: 10 3 256 256

        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)

        # feature.shape: 10 256 17 17
        # cnn_code.shape: 10 256
        return features, cnn_code


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf + cfg.TEXT.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU()
        )
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code, cnn_code):
        c_z_code = torch.cat((c_code, z_code), 1)
        if not cfg.TRAIN.FLAG and not cfg.B_VALIDATION:
            cnn_code = cnn_code.repeat(c_z_code.size(0), 1)

        c_z_cnn_code = torch.cat((c_z_code, cnn_code), 1)
        out_code = self.fc(c_z_cnn_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code32 = self.upsample3(out_code)
        out_code64 = self.upsample4(out_code32)

        return out_code64

class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = SPATIAL_ATT(ngf, self.ef_dim)
        self.channel_att = CHANNEL_ATT(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 3)
        self.upsample = upBlock(ngf * 3, ngf)
        self.SAIN = ACM(ngf * 3)

    def forward(self, h_code, c_code, word_embs, mask, img):
        # pdb.set_trace()
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        c_code_channel, att_channel = self.channel_att(c_code, word_embs, h_code.size(2), h_code.size(3))
        c_code = c_code.view(word_embs.size(0), -1, h_code.size(2), h_code.size(3))

        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        h_c_c_img_code = self.SAIN(h_c_c_code, img)

        out_code = self.residual(h_c_c_img_code)
        out_code = self.upsample(out_code)

        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

#######################################################################################################
# DO NOT CHANGE
class GENERATOR(nn.Module):
#######################################################################################################
    def __init__(self):
        super(GENERATOR, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
            self.img_net1 = GET_IMAGE_G(ngf)
            self.SAIN1 = ACM(ngf)
            self.imgUpSample = imgUpBlock(nef, ngf)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net2 = GET_IMAGE_G(ngf)
            self.SAIN2 = ACM(ngf)
            self.imgUpSample2 = upBlock(ngf, ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net3 = GET_IMAGE_G(ngf)
            self.SAIN3 = ACM(ngf)
            self.imgUpSample3 = upBlock(ngf, ngf)

    def forward(self, z_code, sent_emb, word_embs, mask, cnn_code, region_features):
        """
        z_code: batch x cfg.GAN.Z_DIM
        sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
        imgs / img_emb: batch x ...
        return: generated image
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code, cnn_code)
            img_code64 = self.imgUpSample(region_features)
            h_code_img1 = self.SAIN1(h_code1, img_code64)
            fake_img1 = self.img_net1(h_code_img1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = \
                self.h_net2(h_code1, c_code, word_embs, mask, img_code64)
            img_code128 = self.imgUpSample2(img_code64)
            h_code_img2 = self.SAIN2(h_code2, img_code128)
            fake_img2 = self.img_net2(h_code_img2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = \
                self.h_net3(h_code2, c_code, word_embs, mask, img_code128)
            img_code256 = self.imgUpSample3(img_code128)
            h_code_img3 = self.SAIN3(h_code3, img_code256)
            fake_img3 = self.img_net3(h_code_img3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)
        return fake_imgs, att_maps, mu, logvar, h_code2, c_code


#######################################################################################################
# DO NOT CHANGE
class DISCRIMINATOR(nn.Module):
#######################################################################################################
    def __init__(self, res=64, b_jcu=True):
        super(DISCRIMINATOR, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.res = res
        self.img_code_s16 = encode_image_by_16times(ndf)
        if self.res == 64:
            if b_jcu:
                self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
            else:
                self.UNCOND_DNET = None
            self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)
        elif self.res == 128:
            self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
            self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
            if b_jcu:
                self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
            else:
                self.UNCOND_DNET = None
            self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)
        elif self.res == 256:
            self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
            self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
            self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
            self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
            if b_jcu:
                self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
            else:
                self.UNCOND_DNET = None
            self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)
        else:
            raise ValueError

    def forward(self, x_var):
        if self.res == 64:
            x_code4 = self.img_code_s16(x_var)
        elif self.res == 128:
            x_code8 = self.img_code_s16(x_var)
            x_code4 = self.img_code_s32(x_code8)
            x_code4 = self.img_code_s32_1(x_code4)
        elif self.res == 256:
            x_code16 = self.img_code_s16(x_var)
            x_code8 = self.img_code_s32(x_code16)
            x_code4 = self.img_code_s64(x_code8)
            x_code4 = self.img_code_s64_1(x_code4)
            x_code4 = self.img_code_s64_2(x_code4)
        else:
            raise ValueError
        return x_code4


# Conditional Augmentation Network from H.Zhang et al., Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks
class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


# Affine Combination Module from ManiGAN
class ACM(nn.Module):
    def __init__(self, channel_num):
        super(ACM, self).__init__()
        self.conv = conv3x3(cfg.GAN.GF_DIM, 128)
        self.conv_weight = conv3x3(128, channel_num)
        self.conv_bias = conv3x3(128, channel_num)

    def forward(self, x, img):
        out_code = self.conv(img)
        out_code_weight = self.conv_weight(out_code)
        out_code_bias = self.conv_bias(out_code)
        return x * out_code_weight + out_code_bias


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels not even num'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.InstanceNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.InstanceNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

def conv1x1(in_planes, out_planes, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.InstanceNorm2d(out_planes * 2),
        GLU()
    )
    return block

def imgUpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=3.8, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.InstanceNorm2d(out_planes * 2),
        GLU()
    )
    return block

def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.InstanceNorm2d(out_planes * 2),
        GLU()
    )
    return block

def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)
