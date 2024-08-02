import torch
import torch.nn as nn

from mopa.models.resnet34_unet import UNetResNet34
from mopa.models.scn_unet import UNetSCN



def batch_segment(xm_feats, batch_masks):
    # segment xm_feats in to list based on the batch_masks: [TENSOR_B1,...,TENSOR_Bm]
    qkv_embed_list = []
    start_flag = 0
    max_len = 0
    for mask in batch_masks:
        embed = xm_feats[start_flag : (start_flag+mask)]
        qkv_embed_list.append(embed)
        max_len = embed.shape[0] if embed.shape[0] > max_len else max_len
        # update start_flag
        start_flag += mask
    return qkv_embed_list, max_len

class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs,
                 output_all=False
                 ):
        super(Net2DSeg, self).__init__()

        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        # segmentation head
        self.linear = nn.Linear(feat_channels, num_classes)

        self.output_all = output_all

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(feat_channels, num_classes)

    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        img_indices = data_batch['img_indices']

        # 2D network
        x = self.net_2d(img)

        # 2D-3D feature lifting
        if self.output_all:
            x_all = x.clone().permute(0, 2, 3, 1)
            pred_all = self.linear(x_all)

        img_feats = []
        for i in range(x.shape[0]):
            img_feats.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        img_feats = torch.cat(img_feats, 0)

        # linear
        preds = {'feats': img_feats}

        if self.output_all:
            preds['seg_logit_all'] = pred_all

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(img_feats)

        x = self.linear(img_feats)
        preds['seg_logit'] = x

        return preds


class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 da_method=None,
                 pretrained=False
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        self.backbone_3d = backbone_3d
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        if "Base" not in backbone_3d:
            # segmentation head
            self.linear = nn.Linear(self.net_3d.out_channels, num_classes)

            # 2nd segmentation head
            self.dual_head = dual_head
            if dual_head:
                self.linear2 = nn.Linear(self.net_3d.out_channels, num_classes)

            # da method
            self.da_method = da_method
            if da_method == "MCD":
                self.linear3 = nn.Linear(self.net_3d.out_channels, num_classes)

    def forward(self, data_batch):
        feats = self.net_3d(data_batch['x'])
        x = self.linear(feats)

        preds = {
            'feats': feats,
            'seg_logit': x,
        }

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(feats)

        return preds


def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)

    return out_dict, img_indices


def test_Net3DSeg(backbone='SCN'):
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))

    if backbone == 'SCN':
        feats = torch.rand(num_coords, in_channels)
        feats = feats.cuda()

        net_3d = Net3DSeg(num_seg_classes,
                          dual_head=True,
                          backbone_3d='SCN',
                          backbone_3d_kwargs={'in_channels': in_channels})

        net_3d.cuda()
        out_dict = net_3d({
            'x': [coords, feats],
        })
        for k, v in out_dict.items():
            print('Net3DSeg:', k, v.shape)

    elif "SPVCNN" in backbone:
        from torchsparse import SparseTensor
        feats = torch.cat((coords, torch.rand(num_coords, in_channels)), dim=1)
        lidar = SparseTensor(feats, coords).cuda()
        net_3d = Net3DSeg(num_seg_classes,
                          dual_head=True,
                          backbone_3d='SPVCNN_Base',
                          backbone_3d_kwargs={'in_channels': in_channels}).cuda()
        # Load pretrained test
        # import os
        # print(os.getcwd())
        # state_dict = torch.load("init")["model"]
        # net_3d = load_state(net_3d, state_dict)
        out_dict = net_3d({"lidar":lidar})
        for k, v in out_dict.items():
            print('Net3DSeg:', k, v.shape)

    elif backbone == "SalsaNext":
        range_img = torch.rand(1, 5, 64, 2048).cuda()
        net_3d = Net3DSeg(num_seg_classes,
                          dual_head=True,
                          backbone_3d='SalsaNext',
                          backbone_3d_kwargs={'in_channels': in_channels}).cuda()
        out_dict = net_3d({"proj_in": range_img})
        for k, v in out_dict.items():
            print('Net3DSeg:', k, v.shape)
    
    return out_dict


def load_state(net, state_dict, strict=True):
	if strict:
		net.load_state_dict(state_dict=state_dict)
	else:
		# customized partially load function
		net_state_keys = list(net.state_dict().keys())
		for name, param in state_dict.items():
			name_m = name if "module." not in name else name[7:]
			if name_m in net.state_dict().keys():
				dst_param_shape = net.state_dict()[name_m].shape
				if param.shape == dst_param_shape:
					net.state_dict()[name_m].copy_(param.view(dst_param_shape))
					net_state_keys.remove(name_m)
		# indicating missed keys
		if net_state_keys:
			print(">> Failed to load: {}".format(net_state_keys))
			return net
	return net


if __name__ == '__main__':
    # Test lines for xmuda
    # img_dict, img_indices = test_Net2DSeg()
    pc_dict = test_Net3DSeg("SalsaNext")




