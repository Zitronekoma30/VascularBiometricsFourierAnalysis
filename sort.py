import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.signal import gaussian
from skimage import io, color

PATH_REAL = './Images/Real'
PATH_FAKE = './Images/Fake'
PATH_SORT = './Images/Unsorted'

def mean_squared_error(img1_fft, img2_fft):
    magnitude_difference = np.abs(img1_fft) - np.abs(img2_fft)
    return np.mean(np.square(magnitude_difference))

def bandpass_filter(shape, low_cutoff, high_cutoff):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2

    # Create a mask with ones in the band and zeros elsewhere
    mask = np.zeros((rows, cols))
    mask[int(crow)-high_cutoff:int(crow)+high_cutoff, int(ccol)-high_cutoff:int(ccol)+high_cutoff] = 1
    mask[int(crow)-low_cutoff:int(crow)+low_cutoff, int(ccol)-low_cutoff:int(ccol)+low_cutoff] = 0

    return mask

def apply_bandpass_filter(img_fft, low_cutoff, high_cutoff):
    rows, cols = img_fft.shape
    mask = bandpass_filter((rows, cols), low_cutoff, high_cutoff)
    img_fft_filtered = img_fft * mask
    return img_fft_filtered

def get_usable_fft(path):
    img = io.imread(path) # load image
    img = color.rgb2gray(img) # make grayscale
    img_fft = fft2(img) # get fourier transform of image
    img_fft_centered = np.abs(fftshift(img_fft)) # shift 0 frequency to center
    return img_fft_centered

def convert_all_img_dir(path):
    out = []
    for filename in os.listdir(path):
        out.append(get_usable_fft(os.path.join(path, filename)))

def populate_initial(path_real, path_fake, path_sort):
    # populate lists
    real = convert_all_img_dir(path_real)
    fake = convert_all_img_dir(path_fake)
    sort = convert_all_img_dir(path_sort)
    return real, fake, sort

def main(real, fake, sort):
    # TODO implement sorting into either real or fake from sort
    # depending on whether a real image or a fake image is most similar.

    # TODO implement reversing the process, using the now sorted "sort images"
    # for reference and instead sorting the real and fake images.
    # how many real / fake images end up in the correct category will determine
    # how good / consistent the algorithm is. 
    pass

#if __name__ == "__main__":
#    main(*populate_initial(PATH_REAL, PATH_FAKE, PATH_SORT))

img1 = get_usable_fft("./Images/Image1.png")
img1 = apply_bandpass_filter(img1, 50, 100)
# visualize
vis = img1
plt.imshow(np.log(1 + np.abs(vis)), cmap='gray')
plt.colorbar()
plt.title("Magnitude Difference")
plt.show()