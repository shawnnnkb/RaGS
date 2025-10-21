import torch
from gsplat import rasterization

def main():
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   means = torch.randn((100, 3), device=device)
   quats = torch.randn((100, 4), device=device)
   scales = torch.rand((100, 3), device=device) * 0.1
   colors = torch.rand((100, 128), device=device)
   opacities = torch.rand((100,), device=device)
   # define cameras
   # width, height = 300, 200
   # viewmats = torch.eye(4, device=device)[None, :, :]
   # Ks = torch.tensor([
   #    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
   width, height = 160, 160
   viewmats, Ks = get_bev_setting(width, height)
   # render
   colors, alphas, meta = rasterization(
      means, quats, scales, opacities, colors, viewmats, Ks, width, height
   )
   print(colors.shape, alphas.shape)
   # torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
   print(meta.keys())
   # dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
   # 'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
   # 'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])

def get_bev_setting(width, height, device='cuda'):
   eye = torch.tensor([0., 0., 5.], device=device)     # 相机在 z=5 的位置
   target = torch.tensor([0., 0., 0.], device=device)  # 看向 z=0 的地面
   up = torch.tensor([0., 1., 0.], device=device)      # Y 向上（屏幕竖直向上）

   def look_at(eye, target, up):
        z = (eye - target)
        z = z / z.norm()
        x = torch.cross(up, z)
        x = x / x.norm()
        y = torch.cross(z, x)

        view = torch.eye(4, device=device)
        view[0, :3] = x
        view[1, :3] = y
        view[2, :3] = z
        view[:3, 3] = -eye @ torch.stack([x, y, z])
        return view

   Ks = torch.tensor([
      [10000., 0., width / 2],     # fx, cx
      [0., 10000., height / 2],    # fy, cy
      [0., 0., 1.]
      ], device=device)[None, :, :]
   viewmats = look_at(eye, target, up)[None, :, :]  # shape: [1, 4, 4]
   
   return viewmats, Ks


if __name__ == '__main__':
   main()