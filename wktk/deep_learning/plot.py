# @date: 2018/11/22 11:52
# @author: wangke
# @concat: wangkebb@163.com
# =========================
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def imshow_batch(img, labels=None, classes=None, unnormalize_fn=None):
    batch_size = img.size()[0]  # batch size for print grid label

    # plot grid of images
    img = torchvision.utils.make_grid(img)
    if unnormalize_fn:
        img = unnormalize_fn(img)
    npimg = img.numpy()  # to numpy
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)

    # print grid labels
    labels = np.array([classes[i] for i in labels])
    n_row, n_col = (int(np.ceil(batch_size / 8)), 8) \
        if batch_size > 8 else (1, batch_size)

    width, height, _ = npimg.shape
    labels = np.resize(labels, (n_row, n_col))
    for i in range(n_row):
        for j in range(n_col):
            plt.text((width / n_row) * j, (height / n_col) * i, labels[i, j],
                     ha="left", va="top", bbox=dict(facecolor='w', alpha=0.3))

    plt.show()
