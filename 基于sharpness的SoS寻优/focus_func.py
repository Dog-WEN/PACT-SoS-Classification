import numpy as np
from scipy import ndimage


def calculate_brenner_gradient(images):
    """
        计算多张图像的Brenner梯度

        参数:
        images (numpy.ndarray): 的三维数组，表示 n 张 x*y 像素的图像

        返回:
        numpy.ndarray: 形状为(n,)的一维数组，包含每张图像的Brenner梯度值
    """
    # 确保输入数据为numpy数组
    images = np.array(images)

    # 初始化结果数组
    results = np.zeros(images.shape[0])

    # 对每张图像计算Brenner梯度
    for i in range(images.shape[0]):
        img = images[i]

        # 计算水平方向的Brenner梯度
        horizontal_diff = img[:, 2:] - img[:, :-2]
        horizontal_gradient = np.sum(horizontal_diff ** 2)

        # 计算垂直方向的Brenner梯度
        vertical_diff = img[2:, :] - img[:-2, :]
        vertical_gradient = np.sum(vertical_diff ** 2)

        # 总Brenner梯度为水平和垂直方向的梯度之和
        results[i] = horizontal_gradient + vertical_gradient

    return results


def calculate_tenenbaum_gradient(images):
    """
    计算多张图像的Tenenbaum梯度

    参数:
        images (numpy.ndarray): 的三维数组，表示 n 张 x*y 像素的图像

    返回:
    numpy.ndarray: 形状为(n,)的一维数组，包含每张图像的Tenenbaum梯度值
    """
    # 确保输入数据为numpy数组
    images = np.array(images)

    # 初始化Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 初始化结果数组
    results = np.zeros(images.shape[0])

    # 对每张图像计算Tenenbaum梯度
    for i in range(images.shape[0]):
        img = images[i]

        # 应用Sobel算子计算水平和垂直梯度
        grad_x = ndimage.convolve(img, sobel_x)
        grad_y = ndimage.convolve(img, sobel_y)

        # 计算梯度平方和
        tenenbaum_gradient = np.sum(grad_x ** 2 + grad_y ** 2)

        results[i] = tenenbaum_gradient

    return results

def calculate_normalized_variance(images):
    """
    计算多张图像的归一化方差

    参数:
        images (numpy.ndarray): 的三维数组，表示 n 张 x*y 像素的图像

    返回:
    numpy.ndarray: 形状为(n,)的一维数组，包含每张图像的归一化方差值
    """
    # 确保输入数据为numpy数组
    images = np.array(images)

    # 初始化结果数组
    results = np.zeros(images.shape[0])

    # 对每张图像计算归一化方差
    for i in range(images.shape[0]):
        img = images[i]

        # 计算图像的均值
        mean = np.mean(img)

        # 计算图像的方差
        variance = np.var(img)

        # 计算归一化方差（方差除以均值）
        if mean != 0:  # 避免除以零
            normalized_variance = variance / mean
        else:
            normalized_variance = 0

        results[i] = normalized_variance

    return results