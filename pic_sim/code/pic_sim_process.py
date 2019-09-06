from tqdm import tqdm
import json
import requests
# 安装 scikit-image-0.15.0 和 opencv-python-4.1.0.25
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import cv2
import time

"""
SSIM试图模拟图像结构信息中感知到的变化，而MSE实际上是估计感知到的误差。 
SSIM比较两个窗口（即小的子样本），而不是像MSE那样比较整个图像。 
SSIM能够解释图像结构的变化，而不仅仅是感知到的变化。
"""

def compare_images(imageA, imageB, title):
    # 计算输入图片的SSIM指标值的大小
    # SSMI指标，该指标能够更好的反应出两张图片的相似度，该指标的范围是[-1, 1]，
    # 当SSIM = -1时表示两张图片完全不相似，当SSIM = 1时表示两张图片非常相似。即该值越接近1说明两张图片越相似
    s = ssim(imageA, imageB)

    # # # 画图展示
    # # 创建figure
    # fig = plt.figure(title)
    # plt.suptitle("SSIM: {0:.2f}".format(s))
    #
    # # 显示第一张图片
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(imageA, cmap=plt.cm.gray)
    # plt.axis("off")
    #
    # # 显示第二张图片
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(imageB, cmap=plt.cm.gray)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig('pic__sim.png')
    # plt.show()

    return s

def run_main(input_file):
    with open(input_file,encoding='utf-8',mode='r') as fp:
        linenum = 0
        for line in tqdm(fp.readlines()):
            if linenum >= 1:
                break
            item = json.loads(line)
            with open('temp_img_new.png','wb') as fw:
                fw.write(requests.get(item['img_new']).content)
            with open('temp_img_old.png', 'wb') as fw:
                fw.write(requests.get(item['img_old']).content)

            # 读取图片
            original1 = cv2.imread('temp_img_new.png')
            contrast1 = cv2.imread('temp_img_old.png')

            # 将彩色图转换为灰度图
            original = cv2.cvtColor(original1, cv2.COLOR_BGR2GRAY)
            contrast = cv2.cvtColor(contrast1, cv2.COLOR_BGR2GRAY)
            (H, W) = original.shape
            contrast = cv2.resize(contrast, (H, W))

            # 比较图片
            sim_score = compare_images(original, contrast, "Original vs Contrast")
            return sim_score

if __name__ == "__main__":
    t1 = time.time()
    sim_score = run_main('../data/url.txt')
    t2 = time.time()
    print("两张图片的SSMI值为:{0:.2f}".format(sim_score))
    print("耗时：{0}".format(t2-t1))