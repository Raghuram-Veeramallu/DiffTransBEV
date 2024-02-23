import torch

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use CUDA stream only if a GPU is available
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream()
        # NOTE: With Amp, it isn't necessary to manually convert data to half. Check the referenced codebase
        self.preload()

    def preload(self):
        try:
            self.next_imgs, self.next_rots, self.next_trans, self.next_intrins, self.next_ann_img = next(self.loader)
        except StopIteration:
            self.next_imgs = None
            self.next_rots = None
            self.next_trans = None
            self.next_intrins = None
            self.next_ann_img = None
            return

        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                self.next_imgs = self.next_imgs.cuda(non_blocking=True)
                self.next_rots = self.next_rots.cuda(non_blocking=True)
                self.next_trans = self.next_trans.cuda(non_blocking=True)
                self.next_intrins = self.next_intrins.cuda(non_blocking=True)
                self.next_ann_img = self.next_ann_img.cuda(non_blocking=True)
        else:
            self.next_imgs = self.next_imgs.to(self.device, non_blocking=True)
            self.next_rots = self.next_rots.to(self.device, non_blocking=True)
            self.next_trans = self.next_trans.to(self.device, non_blocking=True)
            self.next_intrins = self.next_intrins.to(self.device, non_blocking=True)
            self.next_ann_img = self.next_ann_img.to(self.device, non_blocking=True)


    def next(self):
        imgs = self.next_imgs
        rots = self.next_rots
        trans = self.next_trans
        intrins = self.next_intrins
        ann_img = self.next_ann_img

        if self.device.type == 'cuda':
            torch.cuda.current_stream().wait_stream(self.stream)
            if imgs is not None:
                imgs.record_stream(torch.cuda.current_stream())
            if rots is not None:
                rots.record_stream(torch.cuda.current_stream())
            if trans is not None:
                trans.record_stream(torch.cuda.current_stream())
            if intrins is not None:
                intrins.record_stream(torch.cuda.current_stream())
            if ann_img is not None:
                ann_img.record_stream(torch.cuda.current_stream())

        self.preload()

        return imgs, rots, trans, intrins, ann_img
