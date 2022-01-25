import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
def conv2(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    for r in np.uint16(np.arange(filter_size / 2.0, img.shape[0] - filter_size / 2.0 + 1)):
        for c in np.uint16(np.arange(filter_size / 2.0, img.shape[1] - filter_size / 2.0 + 1)):
            curr_region = img[r - np.uint16(np.floor(filter_size / 2.0)):r + np.uint16(np.ceil(filter_size / 2.0)),
                          c - np.uint16(np.floor(filter_size / 2.0)):c + np.uint16(np.ceil(filter_size / 2.0))]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)
            result[r, c] = conv_sum

    final_result = result[np.uint16(filter_size / 2.0):result.shape[0] - np.uint16(filter_size / 2.0),
                   np.uint16(filter_size / 2.0):result.shape[1] - np.uint16(filter_size / 2.0)]
    final_result[final_result > 255] = 255
    final_result[final_result < 0] = 0
    return final_result


def fft_conv(img, conv_filter):
    M, N = img.shape
    m, n = conv_filter.shape
    flag = False
    conv_filter = np.fliplr(conv_filter)
    if M % 2 == 0:                      #Padding and Align the filter with the image center
        img = np.pad(img, ((1, 0), (1, 0)), 'constant', constant_values=(0, 0))
        M += 1
        N += 1
        flag = True
    pad_value = int(M / 2) - int(m / 2)
    conv_filter = np.pad(conv_filter, ((pad_value, pad_value), (pad_value, pad_value)), 'constant',
                         constant_values=(0, 0))
    img = np.fft.fft2(img)
    conv_filter = np.fft.fft2(conv_filter)
    res = np.multiply(conv_filter, img)
    res = np.fft.ifft2(res)
    res = res.astype(int)
    res = np.fft.fftshift(res)  # Centralize
    res[res > 255] = 255
    res[res < 0] = 0
    if flag:
        res = res[1:, 1:]
    return res
if __name__ == '__main__':
    img = Image.open('test.jpeg')
    img = img.resize((1023, 1023))
    img = img.convert('L')
    myfiler = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    start = time.time()
    img_conv = conv2(np.asarray(img), myfiler)
    end = time.time()
    print(end - start)

    start = time.time()
    img_fftconv = fft_conv(np.asarray(img), myfiler)
    end = time.time()
    print(end - start)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(img_conv)
    plt.subplot(1, 3, 3)
    plt.imshow(img_fftconv)
    plt.show()
