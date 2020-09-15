import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from skimage import io


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_image', help='Path to input image')
    parser.add_argument('kernel',
                        help='type Gaussian or Sobel. If Gaussian, three filters of different sizes (5x5, 7x7, 11x11) will be applied to the images, generating three results. If Sobel, two filters (3x3) will be applied from the Sobel edge detector (horizontal detector and vertical detector) and the pixel magnitude will be calculated.')
    parser.add_argument('--out_path', help='Path to save the results.')

    args = parser.parse_args()
    return args


def create_gaussian_kernel(sigma):
    """
    Create a normalized Gaussian kernel with size NxN, where N = (3 * sigma + 1). if the size is even, then the size is increased by 1, to be the next largest odd number.

    :param sigma: Standard deviation. Also used to calculate the size of the filter, it represents the importance that larger pixels will have over the central pixel.
    :return: NxN normalized Gaussian kernel.
    """

    size = 3 * sigma + 1
    if size % 2 == 0:
        size += 1

    kernel = np.zeros(shape=(size, size))

    indx = 0
    for x in range(-(size // 2), (size // 2) + 1):
        indy = 0
        for y in range(-(size // 2), (size // 2) + 1):
            kernel[indx, indy] = (1 / (2 * np.pi * sigma ** 2)) * (np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
            indy += 1
        indx += 1

    sum_ = np.sum(kernel)
    kernel /= sum_  # normalize
    return kernel


def convolution(image, kernel):
    """
    Applies a given K kernel to an image I.

    :param image: Gray scale image.
    :param kernel: A pre-defined kernel.
    :return: Image after applying the kernel.
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    center = kernel_height // 2

    padded_image = np.pad(image, (center, center))
    convoluted_image = np.zeros(shape=(image_height, image_width))

    indx = 0
    for i in range(center, padded_image.shape[0] - center):
        indy = 0
        for j in range(center, padded_image.shape[1] - center):
            convoluted_image[indx, indy] = np.sum(
                padded_image[i - center:i + kernel_height - center, j - center:j + kernel_width - center] * kernel)
            indy += 1
        indx += 1
    return convoluted_image


def save(filter_name, fig, name, out_path):
    """
    Saves an image.

    :param filter_name: Gaussian or Sobel.
    :param fig: Figure object.
    :param name: Image name.
    :param out_path: Path to save the image.
    """

    if out_path and not os.path.exists(out_path):
        os.makedirs(out_path)

    image_name = filter_name + '_' + name
    image_path = os.path.join(out_path, image_name)
    fig.savefig(image_path)


def gaussian(image, out_path, name):
    """
    Apply three Gaussian filters (5x5, 7x7 and 11x11) to a grayscale image and save the results.

    :param image: Grayscale image.
    :param out_path: Path to save the results.
    :param name: Image name.
    """

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle('Gaussian Kernels')

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('image')
    ax[0, 0].axis('off')

    kernel = create_gaussian_kernel(1)  # sigma = 1
    convoluted_image = convolution(image.copy(), kernel)
    convoluted_image_normalized = normalize(convoluted_image)
    ax[0, 1].imshow(convoluted_image_normalized, cmap='gray')
    ax[0, 1].set_title('sigma = 1')
    ax[0, 1].axis('off')

    kernel = create_gaussian_kernel(2)  # sigma = 2
    convoluted_image = convolution(image.copy(), kernel)
    convoluted_image_normalized = normalize(convoluted_image)
    ax[1, 0].imshow(convoluted_image_normalized, cmap='gray')
    ax[1, 0].set_title('sigma = 2')
    ax[1, 0].axis('off')

    kernel = create_gaussian_kernel(3)  # sigma = 3
    convoluted_image = convolution(image.copy(), kernel)
    convoluted_image_normalized = normalize(convoluted_image)
    ax[1, 1].imshow(convoluted_image_normalized, cmap='gray')
    ax[1, 1].set_title('sigma = 3')
    ax[1, 1].axis('off')

    if out_path:
        save('gaussian', fig, name, out_path)
    plt.show()


def normalize(data, range_=(0, 255)):
    """
    Normalizes the image to a specified range of values.

    :param data: Image data.
    :param range_: Tuple with the minimum and maximum value to normalize the image. Default (0, 255).
    :return: Normalized image.
    """

    min_ = np.min(data)
    max_ = np.max(data)

    x = (data - min_) / (max_ - min_)
    x_scaled = x * (range_[1] - range_[0]) + range_[0]
    return np.array(x_scaled, dtype=np.uint8)


def sobel(image, out_path, name):
    """
    Apply the Sobel filter to an image and save the result.

    :param image: Grayscale image.
    :param out_path: Path to save the results.
    :param name: Image name.
    """

    horizontal = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
    vertical = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle('Sobel Kernels')

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('image')
    ax[0, 0].axis('off')

    # Normalized image for viewing only
    horizontal_image = convolution(image.copy(), horizontal)
    ax[0, 1].imshow(normalize(horizontal_image.copy()), cmap='gray')
    ax[0, 1].set_title('horizontal normalized')
    ax[0, 1].axis('off')

    # Normalized image for viewing only
    vertical_image = convolution(image.copy(), vertical)
    ax[1, 0].imshow(normalize(vertical_image.copy()), cmap='gray')
    ax[1, 0].set_title('vertical normalized')
    ax[1, 0].axis('off')

    magnitude = np.sqrt(horizontal_image ** 2 + vertical_image ** 2)
    magnitude_normalized = normalize(magnitude)
    ax[1, 1].imshow(magnitude_normalized, cmap='gray')
    ax[1, 1].set_title('magnitude normalized')
    ax[1, 1].axis('off')

    if out_path:
        save('sobel', fig, name, out_path)
    plt.show()


def main():
    args = arg_parse()

    image = io.imread(args.input_image)
    image_name = args.input_image.split('/')[-1]
    if args.kernel == 'Gaussian':
        gaussian(image, args.out_path, image_name)
    elif args.kernel == 'Sobel':
        sobel(image, args.out_path, image_name)


if __name__ == '__main__':
    main()
