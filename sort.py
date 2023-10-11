import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

PATH_REAL = ''
PATH_FAKE = ''
PATH_SORT = ''

def convert_all_img_dir(path):
    out = []
    for filename in os.listdir(path):
        out.append(mpimg.imread(os.path.join(path, filename)))

def get_image_diff(img1, img2):
        return np.abs(img1) - np.abs(img2)

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

if __name__ == "__main__":
    main(*populate_initial(PATH_REAL, PATH_FAKE, PATH_SORT))