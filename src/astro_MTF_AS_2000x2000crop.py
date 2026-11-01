import tifffile, numpy as np, cv2

print('Loading upscaled starless plate...')
img = tifffile.imread('output/m31_starless_upscaled.tif').astype(np.float32) / 65535.0

# Extract a 2000x2000 crop centered on the galactic core
h, w, _ = img.shape
cy, cx = h // 2, w // 2
crop = img[max(0, cy-1000):min(h, cy+1000), max(0, cx-1000):min(w, cx+1000)]

# Standard Astro MTF AutoStretch (matches PixInsight / Siril AutoStretch)
med = np.median(crop)
c0 = max(0.0, float(med - 1.5 * np.std(crop)))
c1 = 1.0
m = 0.12  # Target midtone balance
x = np.clip((crop - c0) / (c1 - c0 + 1e-6), 0.0, 1.0)
stretched = np.where(x <= 0, 0.0, (x * (m - 1.0)) / ((2.0 * m - 1.0) * x - m + 1e-6))

out_jpg = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)
cv2.imwrite('output/m31_starless_core_preview.jpg', cv2.cvtColor(out_jpg, cv2.COLOR_RGB2BGR))
print('Stretched preview successfully written to output/m31_starless_core_preview.jpg')
