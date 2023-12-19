import torch
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torch.utils.data import DataLoader

from data_loader.nuscenes import NuScenesDataset
from models.DiffDiTBEVModel import DiffDiTBEV
from models.view_transformer.lss_utils import gen_dx_bx
from utils.visualization_utils import NormalizeInverse, add_ego, get_nusc_maps, plot_nusc_map

mpl.use('Agg')

denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))

normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))

def viz_model_preds(version,
                    modelf,
                    dataroot='data/nuscenes',
                    map_folder='data/nuscenes',
                    gpuid=1,
                    viz_train=False,

                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,

                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],
                    # xbound=[-25, 25, 0.5],
                    # zbound=[-10, 10, 20],
                    # ybound=[0, 50, 0.5],
                    # dbound=[0, 50, 1],

                    bsz=4,
                    nworkers=10,
                    ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                }

    dataset = NuScenesDataset('v1.0-mini', dataroot)

    trainloader = DataLoader(
        dataset, batch_size=bsz, shuffle=True, 
        num_workers=nworkers,
    )

    valloader = DataLoader(
        dataset, batch_size=bsz, shuffle=True, 
        num_workers=nworkers,
    )

    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu')

    # need to add config
    model = DiffDiTBEV(config=None, device='cpu')

    print('loading', modelf)
    model.load_state_dict(torch.load(modelf, map_location=device))
    model.to('cpu')

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nuscenes.scene:
        log = loader.dataset.nuscenes.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']


    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, binimgs) in enumerate(loader):
            
            # combining the batch with 6 images from each camera sensor
            res_images = imgs.view(-1, imgs.size(2), imgs.size(3), imgs.size(4))

            out = model(res_images.to('cpu'),
                    rots.to('cpu'),
                    trans.to('cpu'),
                    intrins.to('cpu'),
                    binimgs,
                    )

            for si in range(imgs.shape[0]):
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.86))
                plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')

                # plot static map (improves visualization)
                rec = loader.dataset.samples[counter]
                plot_nusc_map(rec, nusc_maps, loader.dataset.nuscenes, scene2map, dx, bx)
                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                imname = f'viz_out/diffdit/eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                counter += 1
            
            break


if __name__ == '__main__':
    # viz_model_preds('v1.0-mini', 'trained_models/diffbev_50.pth')
    viz_model_preds('v1.0-mini', 'trained_models/diffditbev_model_5.pt')
