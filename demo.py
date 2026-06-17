from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from mmseg.structures import SegDataSample
from RSGPNet import RSGPNetSegmentation
from PIL import Image
import numpy as np

img_path = 'resources/80.jpg'

# name_list = ['vegetation', 'building', 'road', 'vehicle', 'background']
name_list = ['road', 'building', 'grass', 'tree', 'car','clutter']

with open('./configs/cls_potsdam.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()


img = Image.open(img_path)
img_tensor = transforms.Compose([
    transforms.ToTensor(),
])(img).unsqueeze(0).to('cuda') # This variable is only a placeholder; the actual data is read within the model. (To be optimized)

data_sample = SegDataSample()
img_meta = {
    'img_path': img_path,
    'ori_shape': img.size[::-1] # H, W
}
data_sample.set_metainfo(img_meta)


model = RSGPNetSegmentation(
    type='RSGPNetSegmentation',
    model_type='SAM3',
    classname_path='./configs/cls_potsdam.txt',
    prob_thd=0.1,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
)

seg_pred = model.predict(img_tensor, data_samples=[data_sample])
seg_pred = seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)
# 类别颜色映射
# color_map = np.array([
#     [107,142,35],    # vegetation
#     [102,102,156],   # building
#     [128,64,128],    # road
#     [0,0,142],       # vehicle
#     [0,0,0]          # background
# ], dtype=np.uint8)
color_map = np.array([
    [255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]         # background
], dtype=np.uint8)

seg_rgb = color_map[seg_pred]

import cv2

def create_overlay(original_img, seg_rgb, alpha=0.5):
    original_array = np.array(original_img.convert("RGB"))

    if original_array.shape[:2] != seg_rgb.shape[:2]:
        original_img = original_img.resize(
            (seg_rgb.shape[1], seg_rgb.shape[0]),
            Image.Resampling.LANCZOS
        )
        original_array = np.array(original_img.convert("RGB"))

    overlay = (
        (1 - alpha) * original_array.astype(np.float32)
        + alpha * seg_rgb.astype(np.float32)
    )

    return overlay.astype(np.uint8)


def add_contours(overlay_img, seg_pred, contour_color=(255, 255, 255), thickness=2):
    overlay_with_contours = overlay_img.copy()

    for class_id in np.unique(seg_pred):
        class_mask = (seg_pred == class_id).astype(np.uint8)

        contours, _ = cv2.findContours(
            class_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay_with_contours,
            contours,
            -1,
            contour_color,
            thickness
        )

    return overlay_with_contours



overlay_img = create_overlay(img, seg_rgb, alpha=0.5)


overlay_with_contours = add_contours(
    overlay_img,
    seg_pred,
    contour_color=(255, 255, 255),
    thickness=2
)

Image.fromarray(seg_rgb).save("seg_pred_color.png")

Image.fromarray(overlay_img).save("overlay_no_contours.png")

Image.fromarray(overlay_with_contours).save("overlay_with_contours.png")

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(img)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(seg_rgb)
ax[1].set_title("Segmentation")
ax[1].axis("off")

ax[2].imshow(overlay_with_contours)
ax[2].set_title("Overlay with Contours")
ax[2].axis("off")

plt.tight_layout()
plt.savefig("seg_pred_comparison.png", bbox_inches="tight", dpi=300)
plt.close()

print("结果已保存：")
print("seg_pred_color.png")
print("overlay_no_contours.png")
print("overlay_with_contours.png")
print("seg_pred_comparison.png")
