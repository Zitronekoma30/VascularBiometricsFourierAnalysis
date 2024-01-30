import time
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.signal import gaussian
from skimage import io, color
from multiprocessing import Pool
import cv2
from collections import Counter
from scipy.spatial import distance
import sys

PATH_REAL = './Images/Data_prepared/PLUS/genuine'
PATH_FAKE = './Images/PLUSsynth' # './Images/PLUSsynth'
PATH_SORT = './Images/Data_prepared/PLUS/spoofed'

class ImageFFT:
    subject_id = ""
    filename = ""
    fft = None
    bands = []
    band_energies = []

    def __init__(self, id, filename, fft):
        self.filename = filename
        self.fft = fft
        self.subject_id = id
        self.bands = []
        self.band_energies = []

    def calculate_bands(self, band_count):
        interval_size = 300 // band_count
        intervals = [(i * interval_size, (i+1) * interval_size) for i in range(band_count)]

        for interval in intervals:
            band = apply_bandpass_filter(self.fft, interval[0], interval[1])
            self.bands.append(band)
            energy = band_energy(band)
            self.band_energies.append(energy)
        #print(self.band_energies)
    
    def calculate_average_energy(self):
        return sum(self.band_energies) / len(self.band_energies)


def get_subject_id(path, filename):
    ### HOW TO FIND THE SUBJECT ID ###
    # SCUT FVD: ID_finger_session_shot_light.bmp / ID identifies the subject
    # IDIAP Vera FV: <size>/<source>/<subject-id>-<gender>/<subject-id>_<side>_<trial>
    # PLUSVein-FV3: [scanner name]_[DORSAL/PALMAR]_[session ID]_[user ID]_[finger ID]_[image ID].png
    subject_id = filename

    if "SCUT" in path or "SCUT" in filename:
        subject_id = filename.split("_")[0]
        # remove anything before a "-" since sometimes there is a leading 001, 002, etc. follwed by a "-" before the subject id
        subject_id = subject_id.split("-")
        subject_id = subject_id[len(subject_id) - 1]

    elif "IDIAP" in path or "IDIAP" in filename:
        # TODO this one is different in some folders, need to write function to detect which pattern is used
        subject_id = filename.split("_")[2]
    elif "PLUS" in path or "PLUS" in filename:
        subject_id = filename.split("_")[3]


    return filename # TODO find subject based on path and which dataset is being used.


def band_energy(img_fft):
    return np.sum(np.square(img_fft))

def apply_bandpass_filter(image, low_cutoff, high_cutoff):
    rows, cols = image.shape
    center_x, center_y = rows // 2, cols // 2

    # Create a grid of coordinates
    x, y = np.ogrid[:rows, :cols]

    # Calculate the distance from the center for each coordinate
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Create the bandpass filter
    bandpass_filter = (low_cutoff <= distance_from_center) & (distance_from_center <= high_cutoff)

    # Apply the bandpass filter to the image
    filtered_image = image * bandpass_filter

    return filtered_image

def get_usable_fft(path):
    img = io.imread(path) # load image
    # check if image is square
    if img.shape[0] != img.shape[1]:
        # resize
        img = resize_image(img, 300, 300)
    
    if len(img.shape) == 3:
        img = color.rgb2gray(img) # make grayscale
    
    img_fft = fft2(img) # get fourier transform of image
    img_fft_centered = np.abs(fftshift(img_fft)) # shift 0 frequency to center
    return img_fft_centered

def convert_all_img_dir(path):
    out = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename != ".DS_Store":
                id = get_subject_id(root, filename)
                img = get_usable_fft(os.path.join(root, filename))
                out.append(ImageFFT(id, filename, img))
    return out

def populate_initial(path_real, path_fake, path_sort):
    # populate lists
    real = convert_all_img_dir(path_real)
    fake = convert_all_img_dir(path_fake)
    sort = convert_all_img_dir(path_sort)
    return real, fake, sort

def calculate_energy_for_image(args):
    i = 0
    images, intervals, unsorted_img, sort_bands = args
    energy_list = []
    for img in images:
        bands = []
        for interval in intervals:
            band = apply_bandpass_filter(img, interval[0], interval[1])
            energy = band_energy(band)
            bands.append(energy)
            ### printing
            i += 1
            total_bands = len(images) * len(intervals)
            percentage = (i / total_bands) * 100
            #print(f'\rProcessing {percentage:.2f}%...', end='', flush=True)
            ###
        energy_list.append(bands)
    print("\n")
    return energy_list


def knn_sort(real, fake, sort, k=5, loso = False) -> bool:
    '''Sorts images into real or fake based on k nearest neighbors, returns true if real false if fake'''
    # TODO change implementation to nearest neighbour instead of average
    band_count = 30

    # calculate bands and their energy for each image
    print("calculating bands and band energy")
    for img in real:
        img.calculate_bands(band_count)
        # progress bar
        print(f'\rProcessing {real.index(img) / len(real) * 100:.2f}%...', end='', flush=True)

    
    print("real done")
    
    for img in fake:
        img.calculate_bands(band_count)
        # progress bar
        print(f'\rProcessing {fake.index(img) / len(fake) * 100:.2f}%...', end='', flush=True)

    print("fake done")

    for img in sort:
        img.calculate_bands(band_count)
        # progress bar
        print(f'\rProcessing {sort.index(img) / len(sort) * 100:.2f}%...', end='', flush=True)

    print("sort done")

    print("sorting images")
    
    # loop over all images to be sorted
    for unsorted_img in sort:
        unsorted_energy_total = unsorted_img.calculate_average_energy()

        real_sums = []
        fake_sums = []
        
        for realimg in real:
            if loso and realimg.subject_id == unsorted_img.subject_id:
                continue
            if not loso and realimg.filename == unsorted_img.filename:
                continue
            real_sums.append(realimg.calculate_average_energy())
        
        for fakeimg in fake:
            if loso and fakeimg.subject_id == unsorted_img.subject_id:
                continue
            if not loso and fakeimg.filename == unsorted_img.filename:
                continue
            fake_sums.append(fakeimg.calculate_average_energy())

        real_sums = sorted(real_sums, key=lambda x: abs(x - unsorted_energy_total))
        fake_sums = sorted(fake_sums, key=lambda x: abs(x - unsorted_energy_total))

        print(unsorted_img.band_energies[0])
        print(real[0].band_energies[0])
              
        real_energy_total = sum(real_sums[:k]) / k
        fake_energy_total = sum(fake_sums[:k]) / k
        print(f"real: {real_energy_total}")
        print(f"fake: {fake_energy_total}")
        print(f"sort: {unsorted_energy_total}")

        ## Classification
        diff_real = abs(unsorted_energy_total - real_energy_total)
        diff_fake = abs(unsorted_energy_total - fake_energy_total)
        if diff_real < diff_fake:
            real.append(unsorted_img)
            print("real")
        else:
            fake.append(unsorted_img)
            print("fake")

    return True
#    return most_common  


def visualize(fft):
    plt.imshow(np.log(1 + np.abs(fft)), cmap='gray')
    plt.colorbar()
    plt.title("Magnitude Difference")
    plt.show()
        
def resize_image(img, img_width, img_height):
    method = cv2.INTER_LANCZOS4
    cv2.resize(img, dsize=(img_width, img_height), interpolation=method)

### MAIN ###

def main(path_real, path_fake, path_sort, k, loso):
    print("converting images")
    real, fake, sort = populate_initial(path_real, path_fake, path_sort)
    original_fake_amt = len(fake)
    original_sort_amt = len(sort)
    original_real_amt = len(real)

    print(knn_sort(real, fake, sort, k, loso))
    
    new_fake_amt = len(fake)
    new_real_amt = len(real)
    fake_sorted = new_fake_amt - original_fake_amt
    real_sorted = new_real_amt - original_real_amt

    # write to file
    output = "TEST AT: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n"
    output += f"k: {k} \nloso: {loso}\n"
    output += f"orig fake amt:{original_fake_amt} \nnew fake:{new_fake_amt} \norig real amt:{original_real_amt} \nnew real amt:{new_real_amt} \noriginal sort:{original_sort_amt}\n"
    output += f"fake sorted:{fake_sorted} \nreal sorted:{real_sorted}\n------------------\n"

    with open("results.txt", "a") as f:
        f.write(output)

    # TODO implement reversing the process, using the now sorted "sort images"
    # for reference and instead sorting the real and fake images.
    # how many real / fake images end up in the correct category will determine
    # how good / consistent the algorithm is. 

if __name__ == "__main__":
    print("start")
    args = sys.argv
    if len(args) < 6:
        print("args: path_real, path_fake, path_sort, k, leave one subject out (y/n)")
        exit()
    if args[5] == "y":
        main(args[1], args[2], args[3], int(args[4]), True)
    else:
        main(args[1], args[2], args[3], int(args[4]), False)
    