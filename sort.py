import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.signal import gaussian
from skimage import io, color

PATH_REAL = './Images/genuine'
PATH_FAKE = './Images/spoofed'
PATH_SORT = './Images/unsorted'

def mean_squared_error(img1_fft, img2_fft):
    magnitude_difference = np.abs(img1_fft) - np.abs(img2_fft)
    return np.mean(np.square(magnitude_difference))

def apply_bandpass_filter(image, low_cutoff, high_cutoff):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2

    # Create a rectangular bandpass filter
    bandpass_filter = np.zeros((rows, cols))
    for x in range(rows):
        for y in range(cols):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if low_cutoff <= distance <= high_cutoff:
                bandpass_filter[x, y] = 1

    # Apply the bandpass filter to the image
    filtered_image = image * bandpass_filter

    return filtered_image

def get_usable_fft(path):
    img = io.imread(path) # load image
    #img = color.rgb2gray(img) # make grayscale
    img_fft = fft2(img) # get fourier transform of image
    img_fft_centered = np.abs(fftshift(img_fft)) # shift 0 frequency to center
    return img_fft_centered

def convert_all_img_dir(path):
    out = []
    for filename in os.listdir(path):
        img = get_usable_fft(os.path.join(path, filename))
        out.append(img)
    return out    

def populate_initial(path_real, path_fake, path_sort):
    # populate lists
    real = convert_all_img_dir(path_real)
    fake = convert_all_img_dir(path_fake)
    sort = convert_all_img_dir(path_sort)
    return real, fake, sort

def main(path_real, path_fake, path_sort, k):
    real, fake, sort = populate_initial(path_real, path_fake, path_sort)
    print(knn_sort(real, fake, sort, k))

    # TODO implement reversing the process, using the now sorted "sort images"
    # for reference and instead sorting the real and fake images.
    # how many real / fake images end up in the correct category will determine
    # how good / consistent the algorithm is. 


def knn_sort(real, fake, sort, k=5) -> bool:
    '''Sorts images into real or fake based on k nearest neighbors, returns true if real false if fake'''
    bands = 5
    #generate "bands" amount of intervals from 0 to 300
    interval_size = 300 // bands
    intervals = [(i * interval_size, (i+1) * interval_size) for i in range(bands)]
    
    real_mse = []
    fake_mse = []

    sort_bands = []


    #loop over all images to be sorted
    for unsorted_img in sort:
        real_i = 0
        fake_i = 0

        for interval in intervals:
            sort_band = apply_bandpass_filter(unsorted_img, interval[0], interval[1])
            sort_bands.append(sort_band)

        #loop over all real images
        for real_img in real:
            print(f"{real_i}/{len(real)}")
            real_i += 1
            #calculate mean squared error for every band
            for interval in intervals:
                real_band = apply_bandpass_filter(real_img, interval[0], interval[1])
                sort_band_index = intervals.index(interval)
                sort_band = sort_bands[sort_band_index]
                mse = mean_squared_error(real_band, sort_band)
                real_mse.append(mse)

        #loop over all fake images
        for fake_img in fake:
            print(f"{fake_i}/{len(fake)}")
            fake_i += 1
            #calculate mean squared error for every band
            for interval in intervals:
                fake_band = apply_bandpass_filter(fake_img, interval[0], interval[1])
                sort_band_index = intervals.index(interval)
                sort_band = sort_bands[sort_band_index]
                mse = mean_squared_error(fake_band, sort_band)
                fake_mse.append(mse)
            
        #sort mse lists
        real_mse.sort()
        fake_mse.sort()
        
        #get k lowest mse values
        real_k = real_mse[:k]
        fake_k = fake_mse[:k]

        #calculate average mse for real images
        real_average_mse = sum(real_k) / len(real_k)
        #calculate average mse for fake images
        fake_average_mse = sum(fake_k) / len(fake_k)

        # compare average mse for real and fake images
        if real_average_mse < fake_average_mse:
            #append to real
            real.append(unsorted_img)
            return true

        if fake_average_mse < real_average_mse:
            #append to fake
            fake.append(unsorted_img)
            return false
        # TODO handle case where real_average_mse == fake_average_mse, ask Prof for a good way to handle this.
        
        

print("start")
main(PATH_REAL, PATH_FAKE, PATH_SORT, 15)



#imgs_real = convert_all_img_dir("./Images/genuine")
#imgs_fake = convert_all_img_dir("./Images/spoofed")
#img_sort_real = get_usable_fft("./Images/unsorted/017-PLUS-FV3-Laser_PALMAR_001_01_09_02.png")

#real_mse = []
#fake_mse = []
#
#for img in imgs_real:
#    real_mse.append(mean_squared_error(img_sort_real, img))
#
#for img in imgs_fake:
#    fake_mse.append(mean_squared_error(img_sort_real, img))
#
#real_average_mse = sum(real_mse) / len(real_mse)
#fake_average_mse = sum(fake_mse) / len(fake_mse)
#
#print(f"real: {real_average_mse}")
#print(f"fake: {fake_average_mse}")
#
#print(min(real_mse))
#print(min(fake_mse))

#  visualize
#vis = apply_bandpass_filter(img_sort_real, 150, 200)
#plt.imshow(np.log(1 + np.abs(vis)), cmap='gray')
#plt.colorbar()
#plt.title("Magnitude Difference")
#plt.show()