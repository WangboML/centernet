# 骨干网络的设计
import torch.nn as nn
from hourglass_utils import convolution, residual
from hourglass_utils import make_layer, make_layer_revr
from hourglass_utils import make_pool_layer,make_unpool_layer,make_merge_layer,make_hg_layers



class hourglass(nn.Module):
    # 首先初始化一些用到的变量和函数
    def __init__(
        self,
        n,   # 表示有几个下采样或上采样模块，这里是5个
        dims, # 表示几个下采样层输入的通道数，最后一个是过渡层输入的通道数
        modules, # 表示几个下采样层和最后过度层中res残差模块的个数；
        layer=residual, # 表示使用的基本单元是残差模块
        make_up_layer=make_layer, #就是基本单元的堆叠
        make_low_layer=make_layer,
        make_hg_layer=make_hg_layers,  # 原始的hourglass这一层不做尺寸变化，这里尺寸为一半
        make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer,
        make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer,
        **kwargs
    ):
        super(hourglass, self).__init__()

        self.n   = n

        curr_mod = modules[0]  # 2
        next_mod = modules[1]  # 2

        curr_dim = dims[0]  # 256
        next_dim = dims[1]  # 256


        # 经过curr_mod个残差网络，由于stride默认为1，dim不改变所以没有任何改变
        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        # 这里的max没有任何操作
        self.max1 = make_pool_layer(curr_dim)
        #经过mod个残差，第一个改变尺寸和维度，其余的res不改变
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        #再次迭代调用自身，此时 n - 1, dims[1:], modules[1:]发生改变，前进一位
        self.low2 = hourglass(
            n - 1, dims[1:], modules[1:], layer=layer,
            make_up_layer=make_up_layer,
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )

        # 经过几个残差回复维度
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        # 进行上采样回复尺寸
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)    #经过modules个残差单元，不改变维度和尺寸；modules分别为[2, 2, 2, 2, 2];
                              # 最后一个modeules值为4；用于最后一个map进行4次残差不改变任何make_low_layer
        max1 = self.max1(x)  # 不做任何处理

        low1 = self.low1(max1) #modules个残差单元，第一个改变维度，并且尺寸降低为一半
                               # [256到256]；[256到384]；[384到384]；[384到384]，[384到512]
        low2 = self.low2(low1) #调用自身，返回的是该次的merge(up1, up2)
        low3 = self.low3(low2) # 回复维度
        up2  = self.up2(low3)   # 恢复尺寸
        return self.merge(up1, up2) # 加操作进行返回



backbonenet=hourglass(
            n       = 5,
            dims    = [256, 256, 384, 384, 384, 512],
            modules = [2, 2, 2, 2, 2, 4])

# inter= torch.randn(6, 256,128,128)
#
# output=backbonenet(inter)
#
# nout=output