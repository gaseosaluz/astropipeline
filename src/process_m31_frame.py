import sys
import torch
import numpy as np
import tifffile
from tqdm import tqdm
from spandrel import ModelLoader

def load_tiff16(filepath: str, device: str = "cuda") -> torch.Tensor:
    img_np = tifffile.imread(filepath)
    if img_np.dtype != np.uint16:
        raise ValueError(f"Expected uint16 TIFF, got {img_np.dtype}")
    
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4):
        img_np = np.transpose(img_np, (1, 2, 0))
    
    if img_np.shape[-1] > 3:
        img_np = img_np[:, :, :3]
    elif img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)
        
    img_float = img_np.astype(np.float32) / 65535.0
    return torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)

def save_tiff16(tensor: torch.Tensor, out_path: str):
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0)
    img_np = (tensor.numpy() * 65535.0).round().astype(np.uint16)
    tifffile.imwrite(out_path, img_np, compression=None)
    print(f"Saved: {out_path} ({img_np.shape[1]}x{img_np.shape[0]}, {img_np.shape[2]} channels, 16-bit)")

def run_tiled_upscale(model_wrapper, tensor: torch.Tensor, tile_size: int = 512, overlap: int = 64) -> torch.Tensor:
    b, c, h, w = tensor.shape
    
    # Dynamically determine exact model scale from a 64x64 probe tile
    probe = torch.zeros((1, c, 64, 64), device=tensor.device)
    with torch.no_grad():
        probe_out = model_wrapper(probe)
    scale = probe_out.shape[-1] // 64
    print(f"Detected true model upscale factor: {scale}x")
    
    stride = tile_size - overlap
    out_h = h * scale
    out_w = w * scale
    
    output = torch.zeros((b, c, out_h, out_w), device=tensor.device, dtype=torch.float32)
    weights = torch.zeros((1, 1, out_h, out_w), device=tensor.device, dtype=torch.float32)
    
    y_starts = list(range(0, h, stride))
    x_starts = list(range(0, w, stride))
    
    with torch.no_grad():
        for y in tqdm(y_starts, desc="Tiled Inference"):
            y_start = min(y, max(0, h - tile_size))
            y_end = min(y_start + tile_size, h)
            
            for x in x_starts:
                x_start = min(x, max(0, w - tile_size))
                x_end = min(x_start + tile_size, w)
                
                tile = tensor[:, :, y_start:y_end, x_start:x_end]
                tile_out = model_wrapper(tile)
                _, _, th_out, tw_out = tile_out.shape
                
                # 2D Bartlett blending window
                win_y = torch.bartlett_window(th_out, periodic=False, device=tensor.device)
                win_x = torch.bartlett_window(tw_out, periodic=False, device=tensor.device)
                win2d = (win_y.unsqueeze(1) * win_x.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                
                oy_start = y_start * scale
                oy_end = oy_start + th_out
                ox_start = x_start * scale
                ox_end = ox_start + tw_out
                
                output[:, :, oy_start:oy_end, ox_start:ox_end] += tile_out * win2d
                weights[:, :, oy_start:oy_end, ox_start:ox_end] += win2d

    output /= torch.clamp(weights, min=1e-6)
    return output

def main():
    if len(sys.argv) < 4:
        print("Usage: python process_m31_frame.py <input.tif> <model_weights.pth> <output.tif>")
        sys.exit(1)
        
    in_file = sys.argv[1]
    weights_path = sys.argv[2]
    out_file = sys.argv[3]
    
    print(f"Loading weights: {weights_path}")
    loader = ModelLoader()
    model = loader.load_from_file(weights_path).to("cuda").eval()
    
    print(f"Loading input: {in_file}")
    img_tensor = load_tiff16(in_file, device="cuda")
    print(f"Input tensor shape: {img_tensor.shape} (BCHW)")
    
    res = run_tiled_upscale(model, img_tensor, tile_size=512, overlap=64)
    save_tiff16(res, out_file)
    print("Inference finished successfully.")

if __name__ == "__main__":
    main()

