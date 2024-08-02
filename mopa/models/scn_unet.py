import torch
import torch.nn as nn

import sparseconvnet as scn

DIMENSION = 3


class UNetSCN(nn.Module):
    def __init__(self,
                 in_channels,
                 m=16,  # number of unet features (multiplied in each layer)
                 block_reps=1,  # depth
                 residual_blocks=False,  # ResNet style basic blocks
                 full_scale=4096,
                 num_planes=7,
                 pretrained=False
                 ):
        super(UNetSCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = m
        n_planes = [(n + 1) * m for n in range(num_planes)]

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(DIMENSION, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)).add(
            scn.UNet(DIMENSION, block_reps, n_planes, residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(DIMENSION))

    def forward(self, x):
        x = self.sparseModel(x)
        return x


# non-recurrent version of SCN_UNet
class UNetSCN_ED(nn.Module):
    leakiness = 0
    downsample = [2, 2]

    def __init__(self,
                 in_channels,
                 m=16,
                 block_reps=1,
                 residual_blocks=False,
                 full_scale=4096,
                 dimension=3
                 ):
        nn.Module.__init__(self)
        self.dimension = DIMENSION
        self.input = scn.InputLayer(DIMENSION, full_scale, mode=4)
        self.down_in = scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)
        # self.main_block1 = self.block(m, m, 2, 1, residual_blocks)
        self.main_block1 = scn.Sequential() \
                               .add(scn.BatchNormLeakyReLU(m, leakiness=0)) \
                               .add(scn.SubmanifoldConvolution(self.dimension, m, m, 3, False))
        self.main_block2 = self.block(m, 2 * m, 1, 2, residual_blocks)
        self.main_block3 = self.block(2 * m, 3 * m, 1, 2, residual_blocks)
        self.main_block4 = self.block(3 * m, 4 * m, 1, 2, residual_blocks)
        self.main_block5 = self.block(4 * m, 5 * m, 1, 2, residual_blocks)
        self.main_block6 = self.block(5 * m, 6 * m, 1, 2, residual_blocks)
        self.main_block7 = self.block(6 * m, 7 * m, 1, 2, residual_blocks)

        self.deconv7 = scn.Sequential() \
                           .add(scn.BatchNormLeakyReLU(7 * m, leakiness=0)) \
                           .add(scn.Deconvolution(self.dimension, 7 * m, 6 * m, 2, 2, False))
        self.join7 = scn.JoinTable()
        
        self.deconv6 = self.decoder(12 * m, 5 * m)
        self.join6 = scn.JoinTable()

        self.deconv5 = self.decoder(10 * m, 4 * m)
        self.join5 = scn.JoinTable()

        self.deconv4 = self.decoder(8 * m, 3 * m)
        self.join4 = scn.JoinTable()

        self.deconv3 = self.decoder(6 * m, 2 * m)
        self.join3 = scn.JoinTable()

        self.deconv2 = self.decoder(4 * m, 1 * m)
        self.join2 = scn.JoinTable()

        self.deconv1 = scn.Sequential() \
                           .add(scn.BatchNormLeakyReLU(2 * m, leakiness=0)) \
                           .add(scn.SubmanifoldConvolution(self.dimension, 2 * m, 1 * m, 3, False))
        # self.deconv1 = self.decoder(2 * m, m)

        self.output = scn.Sequential() \
                          .add(scn.BatchNormReLU(1 * m)) \
                          .add(scn.OutputLayer(DIMENSION))


    def forward(self, x, output_feat=False):
        x = self.input(x)
        x = self.down_in(x)
        feature_1 = self.main_block1(x)                 # 16 -> 16
        feature_2 = self.main_block2(feature_1)         # 16 -> 32
        feature_3 = self.main_block3(feature_2)         # 32 -> 48
        feature_4 = self.main_block4(feature_3)         # 48 -> 64
        feature_5 = self.main_block5(feature_4)         # 64 -> 80
        feature_6 = self.main_block6(feature_5)         # 80 -> 96
        feature_7 = self.main_block7(feature_6)         # 96 -> 112

        if output_feat:
            feats = feature_7.clone()
        decoder_6 = self.deconv7(feature_7)
        decoder_6 = self.join7([feature_6, decoder_6])

        decoder_5 = self.deconv6(decoder_6)
        decoder_5 = self.join6([feature_5, decoder_5])

        decoder_4 = self.deconv5(decoder_5)
        decoder_4 = self.join5([feature_4, decoder_4])

        decoder_3 = self.deconv4(decoder_4)
        decoder_3 = self.join4([feature_3, decoder_3])

        decoder_2 = self.deconv3(decoder_3)
        decoder_2 = self.join4([feature_2, decoder_2])

        decoder_1 = self.deconv2(decoder_2)
        decoder_1 = self.join2([feature_1, decoder_1])

        decoder_1 = self.deconv1(decoder_1)
        # decoder_1 = self.deconv1(decoder_2)
        out = self.output(decoder_1)

        if output_feat:
            return out, feats
        else:
            return out


    def decoder(self, a, b):
        midplanes = int(a/2)
        return (
            scn.Sequential()
                .add(scn.BatchNormLeakyReLU(a, leakiness=0))
                .add(scn.SubmanifoldConvolution(self.dimension, a, midplanes, 3, False))
                .add(scn.BatchNormLeakyReLU(midplanes, leakiness=0))
                .add(scn.Deconvolution(self.dimension, midplanes, b, 2, 2, False))
        )

    def residual(self, nIn, nOut, stride):
        if stride > 1:
            return scn.Convolution(self.dimension, nIn, nOut, 2, stride, False)
        elif nIn != nOut:
            return scn.NetworkInNetwork(nIn, nOut, False)
        else:
            return scn.Identity()

    def block(self, nPlanes, n, reps, stride, residual_blocks):
        m = scn.Sequential()
        if residual_blocks:
            for rep in range(reps):
                if rep == 0:
                    m.add(scn.BatchNormReLU(nPlanes))
                    m.add(
                        scn.ConcatTable()
                            .add(self.residual(nPlanes, n, stride))
                            .add(
                            scn.Sequential()
                                .add(
                                scn.SubmanifoldConvolution(self.dimension, nPlanes, n, 3, False)
                                if stride == 1
                                else scn.Convolution(
                                    self.dimension, nPlanes, n, 2, stride, False
                                )
                            )
                                .add(scn.BatchNormReLU(n))
                                .add(scn.SubmanifoldConvolution(self.dimension, n, n, 3, False))
                        )
                    )
                else:
                    m.add(
                        scn.ConcatTable()
                            .add(
                            scn.Sequential()
                                .add(scn.BatchNormReLU(nPlanes))
                                .add(
                                scn.SubmanifoldConvolution(self.dimension, nPlanes, n, 3, False)
                            )
                                .add(scn.BatchNormReLU(n))
                                .add(scn.SubmanifoldConvolution(self.dimension, n, n, 3, False))
                        )
                            .add(scn.Identity())
                    )
                m.add(scn.AddTable())
                nPlanes = n
        else:
            for rep in range(reps):
                if rep == 0:
                    m.add(scn.BatchNormLeakyReLU(nPlanes, leakiness=0))
                    m.add(
                        scn.Sequential()
                            .add(
                            scn.SubmanifoldConvolution(self.dimension, nPlanes, n, 3, False)
                            if stride == 1
                            else scn.Convolution(
                                self.dimension, nPlanes, n, 2, stride, False
                            )
                        )
                            .add(scn.BatchNormLeakyReLU(n, leakiness=0))
                            .add(scn.SubmanifoldConvolution(self.dimension, n, n, 3, False))
                    )
                else:
                    m.add(
                            scn.Sequential()
                                .add(scn.BatchNormLeakyReLU(nPlanes, leakiness=0))
                                .add(
                                scn.SubmanifoldConvolution(self.dimension, nPlanes, n, 3, False)
                            )
                                .add(scn.BatchNormLeakyReLU(n, leakiness=0))
                                .add(scn.SubmanifoldConvolution(self.dimension, n, n, 3, False))
                    )
                nPlanes = n
        return m


def test():
    b, n = 2, 100
    coords = torch.randint(4096, [b, n, DIMENSION])
    batch_idxs = torch.arange(b).reshape(b, 1, 1).repeat(1, n, 1)
    coords = torch.cat([coords, batch_idxs], 2).reshape(-1, DIMENSION + 1)

    in_channels = 3
    feats = torch.rand(b * n, in_channels)
    print(coords.shape)

    x = [coords, feats.cuda()]

    net = UNetSCN_ED(in_channels).cuda()
    output = net(x)
    # out_feats = net(x)

    print('output', output.shape)


if __name__ == '__main__':
    test()
