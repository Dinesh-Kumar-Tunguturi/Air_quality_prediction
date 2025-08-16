# import torch

# def grids_inverse(self):
#     preds = torch.zeros(self.original_size)
#     b, c, h, w = self.original_size

#     count_mt = torch.zeros((b, 1, h, w))

#     crop_size_h_ratio = self.opt['val'].get('crop_size_h_ratio', 1.0)
#     crop_size_h = int(crop_size_h_ratio * h)

#     crop_size_w_ratio = self.opt['val'].get('crop_size_w_ratio', 1.0)
#     crop_size_w = int(crop_size_w_ratio * w)

#     crop_size_h = crop_size_h // self.scale * self.scale
#     crop_size_w = crop_size_w // self.scale * self.scale

#     for cnt, each_idx in enumerate(self.idxes):
#         i = each_idx['i']
#         j = each_idx['j']
#         preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
#         count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

#     self.output = (preds / count_mt).to(self.device)
#     self.lq = self.origin_lq