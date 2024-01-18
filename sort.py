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

PATH_REAL = './Images/Data_prepared/PLUS/genuine'
PATH_FAKE = './Images/PLUSsynth' # './Images/PLUSsynth'
PATH_SORT = './Images/Data_prepared/PLUS/spoofed'

def mean_squared_error(img1_fft, img2_fft):
    magnitude_difference = np.abs(img1_fft) - np.abs(img2_fft)
    return np.mean(np.square(magnitude_difference))

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
    if img.shape[0] == img.shape[1]:
        # resize
        pass
    
    if len(img.shape) == 3:
        img = color.rgb2gray(img) # make grayscale
    
    img_fft = fft2(img) # get fourier transform of image
    img_fft_centered = np.abs(fftshift(img_fft)) # shift 0 frequency to center
    return img_fft_centered

def convert_all_img_dir(path):
    '''returns a dict: key = subject id, value = list of images'''
    out = {} # dictionary (subject id, [images...])

    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename != ".DS_Store":
                id = get_subject_id(root, filename)
                if out.get(id) == None:
                    out[id] = []
                img = get_usable_fft(os.path.join(root, filename))
                out.get(id).append(img)
    
    return out

def populate_initial(path_real, path_fake, path_sort):
    # populate lists
    real = convert_all_img_dir(path_real)
    fake = convert_all_img_dir(path_fake)
    sort = convert_all_img_dir(path_sort)
    return real, fake, sort

# def calculate_mse_for_image(args):
#     i = 0
#     images, intervals, unsorted_img, sort_bands = args
#     mse_list = []
#     for img in images:
#         for interval in intervals:
#             band = apply_bandpass_filter(img, interval[0], interval[1])
#             sort_band = sort_bands[intervals.index(interval)]
#             mse = mean_squared_error(band, sort_band)
#             print(f"{i}/{len(images)*len(intervals)}")
#             i += 1
#             mse_list.append(mse)
#     return mse_list

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

def calculate_energy_unsorted(image, intervals):
    i = 0
    energy_list = []
    for interval in intervals:
        band = apply_bandpass_filter(image, interval[0], interval[1])
        energy = band_energy(band)
        i += 1
        energy_list.append(energy)
    return energy_list

def knn_sort(real, fake, sort, k=5) -> bool:
    '''Sorts images into real or fake based on k nearest neighbors, returns true if real false if fake'''
    # TODO change implementation to nearest neighbour instead of average
    bands = 30
    #generate "bands" amount of intervals from 0 to 300
    interval_size = 300 // bands
    intervals = [(i * interval_size, (i+1) * interval_size) for i in range(bands)]

    # loop over all subjects in the unsorted images
    for subject_id in sort:
        real_images = []
        fake_images = []

        for subject in real:
            if subject != subject_id:
                real_images += real.get(subject)

        for subject in fake:
            if subject != subject_id:
                fake_images += fake.get(subject)

        # loop over all images to be sorted
        for unsorted_img in sort.get(subject_id):
            sort_bands = []
            # generate bandpass filters for each interval
            for interval in intervals:
                sort_band = apply_bandpass_filter(unsorted_img, interval[0], interval[1])
                sort_bands.append(sort_band)

            ### MEAN SQUARED ERROR
            # #loop over all images
            # real_mse = calculate_mse_for_image((real_images, intervals, unsorted_img, sort_bands))
            # fake_mse = calculate_mse_for_image((fake_images, intervals, unsorted_img, sort_bands))
            
            ## Unsorted Image Energy
            unsorted_energy = calculate_energy_unsorted(unsorted_img, intervals)
            unsorted_energy.sort()
            unsorted_energy_total = sum(unsorted_energy) / len(unsorted_energy)

            ### ENERGY
            # loop over images
            real_energy = calculate_energy_for_image((real_images, intervals, unsorted_img, sort_bands))
            fake_energy = calculate_energy_for_image((fake_images, intervals, unsorted_img, sort_bands))
            real_energy.sort()
            fake_energy.sort()

            ## Avg Energy
            real_sums = []
            fake_sums = []

            for bs in real_energy: 
                real_sums.append(sum(bs) / len(bs))
            
            for bs in fake_energy:
                fake_sums.append(sum(bs) / len(bs))

            real_sums = sorted(real_sums, key=lambda x: abs(x - unsorted_energy_total))
            fake_sums = sorted(fake_sums, key=lambda x: abs(x - unsorted_energy_total))
            
            real_energy_total = sum(real_sums[:k]) / k
            fake_energy_total = sum(fake_sums[:k]) / k

            print(f"real: {real_energy_total}")
            print(f"fake: {fake_energy_total}")
            print(f"sort: {unsorted_energy_total}")

            ## Classification
            diff_real = abs(unsorted_energy_total - real_energy_total)
            diff_fake = abs(unsorted_energy_total - fake_energy_total)

            if diff_real < diff_fake:
                if real.get(subject_id) == None:
                    real[subject_id] = []
                real.get(subject_id).append(unsorted_img)
                print("real")
            else:
                if fake.get(subject_id) == None:
                    fake[subject_id] = []
                fake.get(subject_id).append(unsorted_img)
                print("fake")
    return True
            
            #### True knn
            #label = knn(k, unsorted_energy, real_energy, fake_energy)
            #if label == 'real':
            #    if real.get(subject_id) == None:
            #        real[subject_id] = []
            #    real.get(subject_id).append(unsorted_img)
            #    return True
            #else:
            #    if fake.get(subject_id) == None:
            #        fake[subject_id] = []
            #    fake.get(subject_id).append(unsorted_img)
            #    return False

#def knn(k, unsorted_energy, real_energy, fake_energy):
#    # Calculate distances and label them
#    distances = [(distance.euclidean(unsorted_energy, real), 'real') for real in real_energy]
#    distances += [(distance.euclidean(unsorted_energy, fake), 'fake') for fake in fake_energy]
#
#    # Sort distances
#    distances.sort(key=lambda x: x[0])
#    print(distances)
#    # Get labels of top k elements
#    labels = [label for _, label in distances[:k]]
#    print(labels)
#    # Get most common label
#    most_common = Counter(labels).most_common(1)[0][0]
#
#    return most_common  

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

def visualize(fft):
    plt.imshow(np.log(1 + np.abs(fft)), cmap='gray')
    plt.colorbar()
    plt.title("Magnitude Difference")
    plt.show()
        
def resize_image(img, img_width, img_height):
    method = cv2.INTER_LANCZOS4
    cv2.resize(img, dsize=(img_width, img_height), interpolation=method)

### MAIN ###

def main(path_real, path_fake, path_sort, k):
    print("converting images")
    real, fake, sort = populate_initial(path_real, path_fake, path_sort)
    original_fake_amt = len(fake)
    original_sort_amt = len(sort)
    print("sorting images")
    print(knn_sort(real, fake, sort, k))
    new_fake_amt = len(fake)
    correctly_sorted = new_fake_amt - original_fake_amt
    # write to file
    with open("results.txt", "a") as f:
        f.write(f"{correctly_sorted} of {original_sort_amt} images were sorted correctly\n")
        f.write(f"original fake:{original_fake_amt} \n new fake:{new_fake_amt}\n original sort:{original_sort_amt}\n new fake: {new_fake_amt} \n k:{k} \n correctly sorted: {correctly_sorted}\n")

    # TODO implement reversing the process, using the now sorted "sort images"
    # for reference and instead sorting the real and fake images.
    # how many real / fake images end up in the correct category will determine
    # how good / consistent the algorithm is. 

if __name__ == "__main__":
    print("start")
    PATH_REAL = input("Enter path to real images: ")
    PATH_FAKE = input("Enter path to fake images: ")
    PATH_SORT = input("Enter path to sort images: ")
    main(PATH_REAL, PATH_FAKE, PATH_SORT, 10)