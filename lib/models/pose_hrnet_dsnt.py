import dsntnn

from lib.models.pose_hrnet import PoseHighResolutionNet


class PoseHighResolutionNetDSNT(PoseHighResolutionNet):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # 1x1 convolution is usually used to change number of channels / heatmaps: we want n_locations
        # however, the HRNet already ends with FINAL_CONV_KERNEL=1 (by default) and n_locations out-channels
        # n_locations = cfg['MODEL']['NUM_JOINTS']
        # self.hm_conv = nn.Conv2d(n_locations, n_locations, kernel_size=1, bias=False)

    def forward(self, x):
        # heatmap output of the HRNet
        fcn_out = super().forward(x)

        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        # however, as documented in __init__(), HRNet already ends with 1x1 conv and n_locations out channels
        # so we can just pass through:
        # unnormalized_heatmaps = self.hm_conv(fcn_out)
        unnormalized_heatmaps = fcn_out

        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)

        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNetDSNT(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
