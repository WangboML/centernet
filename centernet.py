
import torch.nn as nn
from .centernet_utils import convolution, residual
from .centernet_utils import make_br_layer,make_ct_layer,make_tl_layer,make_cnv_layer,make_kp_layer,make_inter_layer
from .centernet_utils import _tranpose_and_gather_feat,_decode

from .hourglass import backbonenet



class centernet(nn.Module):
    # 如果是center52 则nstack=1；
    # 如果是center102 则nstack=2；
    def __init__(
            self, args, nstack=1, out_dim=80, pre=None, curr_dim =256,cnv_dim=256,
            make_heat_layer=make_kp_layer,
            make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer
    ):
        super(centernet, self).__init__()

        self.nstack = nstack  # 有几个stacked结构 centernet52:1; centernet102:2
        self._decode = _decode
        #self._db = db         # 网络一些参数的设置

        # self.K = self._db.configs["top_k"]
        # self.ae_threshold = self._db.configs["ae_threshold"]
        # self.kernel = self._db.configs["nms_kernel"]
        # self.input_size = self._db.configs["input_size"][0]
        # self.output_size = self._db.configs["output_sizes"][0][0]

        self.K = args.top_k
        self.ae_threshold = args.ae_threshold
        self.kernel = args.nms_kernel
        self.input_size = args.input_size[0]
        self.output_size = args.output_size[0][0]

        #curr_dim = dims[0]   # 骨干网络的输入维度，是pre操作之后的维度；也就是骨干网络的输出维度

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre            # 首先定义一个改变维度和大小的预卷积操作

        self.backbone = nn.ModuleList([backbonenet for _ in range(nstack)])   # 然后经历一个骨干网络

        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)   # 骨干网络之后的一个卷积操作
        ])

        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)   #三个分支的左上分支
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)  # 三个分支的右下分支
        ])

        self.ct_cnvs = nn.ModuleList([
            make_ct_layer(cnv_dim) for _ in range(nstack)  # 三个分支的中间分支
        ])

        ## keypoint heatmaps（三个分支经过卷积生成热度图）
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.ct_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        ## tags（左上和右下两个分支经过卷积生成embeding图）
        self.tl_tags = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.br_tags = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])


        ### 回归分支
        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.ct_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])



        for tl_heat, br_heat, ct_heat in zip(self.tl_heats, self.br_heats, self.ct_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)
            ct_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image = xs[0]
        tl_inds = xs[1]
        br_inds = xs[2]
        ct_inds = xs[3]

        inter = self.pre(image)
        outs = []

        layers = zip(
            self.backbone, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.ct_cnvs, self.tl_heats,
            self.br_heats, self.ct_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs,
            self.ct_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_ = layer[0:2]
            tl_cnv_, br_cnv_ = layer[2:4]
            ct_cnv_, tl_heat_ = layer[4:6]
            br_heat_, ct_heat_ = layer[6:8]
            tl_tag_, br_tag_ = layer[8:10]
            tl_regr_, br_regr_ = layer[10:12]
            ct_regr_ = layer[12]

            kp = kp_(inter)
            cnv = cnv_(kp)

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)
            ct_cnv = ct_cnv_(cnv)

            tl_heat, br_heat, ct_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), ct_heat_(ct_cnv)
            tl_tag, br_tag = tl_tag_(tl_cnv), br_tag_(br_cnv)
            tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)

            tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)

            outs += [tl_heat, br_heat, ct_heat, tl_tag, br_tag, tl_regr, br_regr,
                     ct_regr]  # 这个加操作没有影响，因为每个batch调用的时候都归【】


            # 如果使用两个形同结构叠加结果的方式的话
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)

        outs = []

        layers = zip(
            self.backbone, self.cnvs,
            self.tl_cnvs, self.br_cnvs,
            self.ct_cnvs, self.tl_heats,
            self.br_heats, self.ct_heats,
            self.tl_tags, self.br_tags,
            self.tl_regrs, self.br_regrs,
            self.ct_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_ = layer[0:2]
            tl_cnv_, br_cnv_ = layer[2:4]
            ct_cnv_, tl_heat_ = layer[4:6]
            br_heat_, ct_heat_ = layer[6:8]
            tl_tag_, br_tag_ = layer[8:10]
            tl_regr_, br_regr_ = layer[10:12]
            ct_regr_ = layer[12]

            kp = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                ct_cnv = ct_cnv_(cnv)

                tl_heat, br_heat, ct_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), ct_heat_(ct_cnv)
                tl_tag, br_tag = tl_tag_(tl_cnv), br_tag_(br_cnv)
                tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,
                         ct_heat, ct_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-8:], **kwargs)

    def forward(self, *xs, **kwargs):
        # 如果输入的tensor大于1的话
        if len(xs) > 1:
            return self._train(*xs, **kwargs)  #调用train
        return self._test(*xs, **kwargs)       # 否则调用test



# if __name__=="__main__":




