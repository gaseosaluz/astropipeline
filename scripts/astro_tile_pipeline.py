import os
import torch
import numpy as np
import tifffile
from tqdm import tqdm

def load_tiff16_to_tensor(filepath: str, device: str = "cuda") -> torch.Tensor:
    """Reads a 16-bit TIFF and converts to a normalized (0.0 to 1.0) float32 CUDA tensor (1, C, H, W)."""
    img_np = tifffile.imread(filepath)
    if img_np.dtype != np.uint16:
        raise ValueError(f"Expected uint16 image, got {img_np.dtype}")
    
    # Handle HWC vs CHW (typically HWC from stacking software)
    if img_np.ndim == 2:  # Monochrome
        img_np = np.expand_dims(img_np, axis=-1)
    
    # Normalize uint16 to [0.0, 1.0] float32
    img_float = img_np.astype(np.float32) / 65535.0
    
    # Transpose HWC -> CHW -> BCHW
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

def save_tensor_to_tiff16(tensor: torch.Tensor, out_path: str):
    """Converts a normalized (0.0 to 1.0) BCHW tensor back to 16-bit TIFF."""
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().clamp(0.0, 1.0)
    img_np = (tensor.numpy() * 65535.0).round().astype(np.uint16)
    tifffile.imwrite(out_path, img_np, compression=None)

def tiled_inference(model, tensor: torch.Tensor, tile_size: int = 512, overlap: int = 64) -> torch.Tensor:
    """
    Runs model inference using overlapping tiles with linear feathering 
    to prevent tile seams and boundary artifacts on high-res DSO images.
    """
    b, c, h, w = tensor.shape
    stride = tile_size - overlap
    
    # Output canvas and weight accumulator on GPU
    output = torch.zeros_like(tensor)
    weights = torch.zeros((1, 1, h, w), device=tensor.device)
    
    # Generate 2D linear blending window (Bartlett / triangular window)
    win1d = torch.bartlett_window(tile_size, periodic=False, device=tensor.device)
    win2d = (win1d.unsqueeze(1) * win1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    y_starts = list(range(0, h - overlap, stride))
    x_starts = list(range(0, w - overlap, stride))
    
    model.eval()
    with torch.no_grad():
        for y in tqdm(y_starts, desc="Processing Rows"):
            y_end = min(y + tile_size, h)
            y_start = max(0, y_end - tile_size)
            
            for x in x_starts:
                x_end = min(x + tile_size, w)
                x_start = max(0, x_end - tile_size)
                
                tile = tensor[:, :, y_start:y_end, x_start:x_end]
                
                # Model inference call (identity mock or actual PyTorch network)
                tile_out = model(tile)
                
                output[:, :, y_start:y_end, x_start:x_end] += tile_out * win2d
                weights[:, :, y_start:y_end, x_start:x_end] += win2d

    # Normalize blended overlapping regions
    output /= torch.clamp(weights, min=1e-6)
    return output

if __name__ == "__main__":
    print(f"CUDA status: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}")
    # Verify synthetic tensor pass
    dummy_input = torch.rand((1, 3, 2048, 2048), dtype=torch.float32, device="cuda")
    mock_model = torch.nn.Identity().to("cuda")
    
    res = tiled_inference(mock_model, dummy_input, tile_size=512, overlap=64)
    print("Tile pipeline benchmark complete. Output shape:", res.shape)
    print("Max VRAM allocated:", f"{torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
