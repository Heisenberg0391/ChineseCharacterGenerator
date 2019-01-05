# coding=utf-8
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from config import cfg
import os
import glob
import cv2
import mahotas
import progressbar
import codecs

def build_dict():
    """ 打开字典，加载全部字符到list
        每行是一个字
    """
    dict = []
    with codecs.open(cfg.dict_path, mode='r', encoding='utf-8') as f:
        # 按行读取语料
        for line in f:
            # 当前行单词去除结尾，为了正常读取空格，第一行两个空格
            word = line.strip('\r\n')
            # 只要没超出上限就继续添加单词
            dict.append(word)
    return dict

def augmentation(img, mode, size):
    ''' 不能直接在原始image上改动
        添加随机模糊和噪声
    '''
    image = img.copy()
    # 高斯模糊
    if mode == 0:
        image = cv2.GaussianBlur(image,(5, 5), np.random.randint(1, 10))

    # 模糊后二值化，虚化边缘
    if mode == 1:
        image = cv2.GaussianBlur(image, (5, 5), np.random.randint(1, 6))
        T = mahotas.thresholding.otsu(image)
        thresh = image.copy()
        thresh[thresh > T] = 255
        thresh[thresh < 255] = 0
        image = thresh

    # 横线干扰
    if mode == 2:
        for i in range(0, image.shape[0], 2):
            cv2.line(image, (0, i), (size[0], i), 0, 1)

    # 竖线
    if mode == 3:
        for i in range(0, image.shape[1], 2):
            cv2.line(image, (i, 0), (i, size[0]), 0, 1)

    # 十字线
    if mode == 4:
        for i in range(0,image.shape[0], 2):
            cv2.line(image, (0, i), (size[0], i), 0, 1)
        for i in range(0, image.shape[0], 2):
            cv2.line(image, (i, 0), (i, size[0]), 0, 1)

    # 左右运动模糊
    if mode == 5:
        kernel_size = 7
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        image = cv2.filter2D(image, -1, kernel_motion_blur)

    # 上下运动模糊
    if mode == 6:
        kernel_size = 9
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        image = cv2.filter2D(image, -1, kernel_motion_blur)

    # 高斯噪声
    if mode == 7:
        row, col = image.shape
        mean = 0
        sigma = 2
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        image = noisy.astype(np.uint8)

    return image

# 根据字体输出图像
def draw_txt(n, charset, fonts, size):
    img_w, img_h = (size[0], size[1])
    factor = 1  # 初始字体大小
    # 初始化进度条
    widgets = ["数据集创建中: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=n,
                                   widgets=widgets).start()
    # 遍历所有字
    for i in range(n):
        char = charset[i]  # 当前字
        # 遍历字体
        for j, each in zip(range(len(fonts)), fonts):
            # 数据增强
            for mode in range(0, 8):
                # 创建画布
                canvas = np.zeros(shape=(img_w, img_h), dtype=np.uint8)
                canvas[0:] = 255
                # 从ndarray转成image进行渲染
                ndimg = Image.fromarray(canvas).convert('RGBA')
                draw = ImageDraw.Draw(ndimg)

                font = ImageFont.truetype(each, int(img_h * factor), 0)
                text_size = font.getsize(char)  # 获取当前字体下的文本区域大小

                # 自动调整字体大小避免超出边界, 至少留白水平10%
                margin = [img_w - int(0.2 * img_w), img_h - int(0.2 * img_h)]
                while (text_size[0] > margin[0]) or (text_size[1] > margin[1]):
                    factor -= 0.01  # 控制字体大小
                    font = ImageFont.truetype(each, int(img_h * factor), 0)  # 加载字体
                    text_size = font.getsize(char)

                # 随机平移
                horizontal_space = int(img_w - text_size[0])
                vertical_space = int(img_h - text_size[1])
                start_x = np.random.randint(1, horizontal_space - 1)
                start_y = np.random.randint(1, vertical_space - 1)

                # 绘制当前文本行
                draw.text((start_x, start_y), char, font=font, fill=(0, 0, 0, 255))
                img_array = np.array(ndimg)
                # ndimg.show()
                # 转灰度图
                img = img_array[:, :, 0]  # [32, 256, 4]
                # 生成保存路径

                save_dir = os.path.join(cfg.IMAGE_PATH, char)
                img_name = str(i) + str(j) + str(mode) + '.jpg'  # 第i个字第j个字体第mode种增强
                save_path = os.path.join(save_dir, img_name)
                # 数据增强
                aug = augmentation(img, mode, size)
                out = Image.fromarray(aug)

                # 检查路径是否存在，如果存在则直接保存图像
                # 否则需先创建路径
                if os.path.isdir(save_dir):
                    out.save(save_path )
                else:
                    os.makedirs(save_dir)
                    out.save(save_path )

        pbar.update(i)
    pbar.finish()

# 自动加载字体文件
def load_fonts():
    fnts = []

    # 字体路径
    font_path = os.path.join(cfg.FONT_PATH, "*.ttf")
    # 获取全部字体路径，存成list
    fonts = list(glob.glob(font_path))

    # 遍历字体文件
    for each in fonts:
        fnts.append(each)

    return fnts


if __name__ == '__main__':
    # 定义一些数组和参数
    label = []
    training_data = []
    # 批大小
    batchSize = 1024
    # 图像尺寸
    size = (64, 64)  # w, h
    # 字体list，每一个字符遍历所有字体，依次输出
    fonts = load_fonts()
    # 字符集，将其中的字符保存成图像
    charset = build_dict()
    # 生成n个字
    # n = 10
    n = len(charset)
    draw_txt(n, charset, fonts, size)
