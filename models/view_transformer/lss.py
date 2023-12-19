import torch
import torch.nn as nn

from models.view_transformer.lss_utils import QuickCumsum, cumsum_trick, gen_dx_bx


class LiftSplatShootTransformer(nn.Module):

    def __init__(self, grid_conf, dep_sup=False, out_dep=False, device='cuda:0'):
        super().__init__()

        self.grid_conf = grid_conf
        self.device = device

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                self.grid_conf['ybound'],
                                self.grid_conf['zbound'],
                                )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.downsample = 32
        self.in_channels = 768
        self.out_dep = out_dep
        self.dep_sup = dep_sup

        # ogfH = 600
        # ogfW = 800
        ogfH = 150
        ogfW = 200

        # ([49, 18, 25, 3])
        self.frustum = self.create_frustum(ogfH, ogfW)
        # D = 49
        self.D, _, _, _ = self.frustum.shape
        self.C = 64 # bev_feature_channels

        self.in_channels = 768
        self.depthnet = nn.Conv2d(self.in_channels, self.D + self.C, kernel_size=1, padding=0)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    # create frustrum of camera view
    # shape returned when (600, 800) are inputs: ([49, 18, 25, 3])
    def create_frustum(self, ogfH, ogfW):
        # make grid in image plane
        # final dimensions after downsampling for computation
        fH, fW = ogfH // self.downsample, ogfW // self.downsample   # 18, 25
        # create depth samples within the dbounds defined
        # depth samples: ([49, 18, 25])
        depth_samples = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = depth_samples.shape   # D = 49
        # xs, ys: ([49, 18, 25])
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x C
        frustum = torch.stack((xs, ys, depth_samples), -1)
        # frustum shape: ([49, 18, 25, 3])
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        D, H, W, C = self.frustum.shape
        post_trans = torch.zeros(B, N, 1, 1, 1, 3).to(self.device)
        points = self.frustum - post_trans
        points = points.unsqueeze(-1)

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)
    
    def get_cam_feats(self, x, h, w):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.depthnet(x)        # x shape: ([6, 113, 29, 50])
        depth = self.get_depth_dist(x[:, :self.D])
        # depth shape: ([6, 49, 29, 50])
        depth_logit = x[:, :self.D]
        # depth_logit shape: ([6, 49, 29, 50])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        x = new_x.view(B, N, self.C, self.D, h, w)  # feature.shape: 1,6,64,49,18,25
        # x = x.view(B, N, self.C, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, depth, depth_logit

    def voxel_pooling(self, geom_feats, x):
        # torch.Size([1, 6, 50, 4, 6, 64])
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # except, we might have to convert nx to torch.long
        nx = self.nx.to(torch.long)

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()   # ([1, 6, 49, 18, 25, 3])
        geom_feats = geom_feats.view(Nprime, 3)   # ([132300, 3])
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # ([132300, 4])

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]                         # torch.Size([52291, 64])       # torch.Size([2775, 64])
        geom_feats = geom_feats[kept]       # torch.Size([52291, 4])        # torch.Size([2775, 4])

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B)\
            + geom_feats[:, 1] * (nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            # torch.Size([486, 64]), torch.Size([486, 4])
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)   # torch.Size([1812, 64]), torch.Size([1812, 4])
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[0], nx[1]), device=x.device)                     # torch.Size([1, 64, 1, 200, 200])
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # torch.Size([1, 64, 1, 200, 200])

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # ([1, 64, 98, 100])

        return final

    def get_voxels(self, x, rots, trans, intrins):
        # size of ([1, 6, 50, 4, 6, 3])
        geom = self.get_geometry(rots, trans, intrins)
        b, _, _, h, w, _  = geom.shape
        # the next 2 lines are skeptical
        if self.downsample==32:
            x = x[:, :,:,:h,:w]

        x, depth, depth_logit = self.get_cam_feats(x, h, w) # shape: ([1, 6, 50, 4, 6, 64])
        
        if self.out_dep:
            x = self.voxel_pooling(geom, torch.cat((x, depth[:,None,...,None]), dim=-1))
        else:
            x = self.voxel_pooling(geom, x) # shape: ([1, 64, 100, 100])

        # if self.dep_sup:
        #     return x, depth_logit

        return x, depth_logit

    def forward(self, x, rots, trans, intrins):
        # input shape: [B x 6 x 768 x 8 x 13]
        x, logit = self.get_voxels(x, rots, trans, intrins) # shape: ([B, 64, 98, 100])
        return x, logit
