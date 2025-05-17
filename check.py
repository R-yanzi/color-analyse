import numpy as np

data = np.load("data/dataset_rgb_region.npz")
images = data["images"]
labels = data["labels"]

print("图像数量:", len(images))
print("图像形状:", images.shape)
print("标签形状:", labels.shape)
print("第一条样本标签:", labels[0])

# import numpy as np
#
# data = np.load("data/dataset_rgb_region.npz")  # 注意路径
# print("images shape:", data["images"].shape)
# print("labels shape:", data["labels"].shape)

