import os
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure
import glob
import csv
from matplotlib import colors
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from scipy.interpolate import RegularGridInterpolator
import SimpleITK as sitk
from scipy.ndimage import zoom
from scipy.stats import pearsonr
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import warnings

def whitening(image, eps=1e-8):
    image = image.astype(np.float32)
    ret = (image - np.mean(image)) / (np.std(image) + eps)
    return ret


def normalise_zero_one(image, eps=1e-8):
    image = image.astype(np.float32)
    ret = (image - np.min(image))
    ret /= (np.max(image) - np.min(image) + eps)

    return ret


def normalise_one_one(image):
    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.

    return ret

def reduce_hu_intensity_range(img, minv=100, maxv=1500):
    img = np.clip(img, minv, maxv)
    img = 255 * normalise_zero_one(img)

    return img


def gray2rgb(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        return np.dstack([img] * 3)
    else:
        return img


def to256(img):
    return 255 * (img - img.min()) / (img.max() - img.min() + 0.001)


def mat2gray(im):
    im = im.astype(np.float32)
    return (im - np.min(im)) / (np.max(im) - np.min(im) + 0.0001)


def blend2d(image, labelmap, alpha, label=1):
    image = np.stack((image,) * 3, axis=-1)
    labelmap = np.stack((labelmap,) * 3, axis=-1)
    labelmap[:, :, 1:2] = 0
    return alpha * labelmap + \
           np.multiply((1 - alpha) * mat2gray(image), mat2gray(labelmap == label)) \
           + np.multiply(mat2gray(image), 1 - mat2gray(labelmap == label))

def overlay_heatmap_on_image(img, heatmap):
    pred_map_color = cv2.applyColorMap((255 * (1 - heatmap)).astype(np.uint8), cv2.COLORMAP_JET)
    return (img * (1 - heatmap) + heatmap * pred_map_color).astype(np.uint8)


def local_normalisation(img):
    local_mean = gaussian_filter(img, 5)
    nI = img - local_mean
    sI = np.sqrt(0.5 + gaussian_filter(nI ** 2, 5))
    return nI / sI


# Make MIPs
#326
def extract_mip(image, d=10, s=40):
    image_c = image.copy()

    image_c[:, :s, ] = 0
    image_c[:, -s:, ] = 0
    image_c[:, :, :s] = 0
    image_c[:, :, -s:] = 0

    (_, _, Z) = np.meshgrid(range(image.shape[1]), range(image.shape[0]), range(image.shape[2]))
    M = Z * (image_c > 0)
    M = M.sum(axis=2) / (image_c > 0).sum(axis=2)
    M[np.isnan(M)] = 0
    mask = M > 0
    c = int(np.mean(M[mask]))

    image_frontal = np.max(image_c, axis=1)
    
    #image_frontal = np.max(image_c[:, c - 20:c + 20, :], axis=1)[::-1, :]
    image_sagittal = np.max(image_c[:, :, c - d:c + d], axis=2)[::1, :]

    return image_frontal, image_sagittal

def extract_random_example_array(image_list, example_size=[64, 64], n_examples=1, loc=[50, 50], anywhere=False,
                                 border_shift=10):
    """
        Randomly extract training examples from image (and corresponding label).
        Returns an image example array and the corresponding label array.

        Parameters
        ----------
        image_list: np.ndarray or list or tuple
            image(s) to extract random patches from
        example_size: list or tuple
            shape of the patches to extract
        n_examples: int
            number of patches to extract in total

        Returns
        -------
        examples
            random patches extracted from bigger images with same type as image_list with of shape
            [batch, example_size..., image_channels]
    """

    def get_range(img_idx):
        if anywhere:
            valid_loc_range = [image_list[img_idx].shape[i] - example_size[i] for i in range(rank)]

            rnd_loc = [np.random.randint(valid_loc_range[dim], size=n_examples)
                       if valid_loc_range[dim] > 0 else np.zeros(n_examples, dtype=int) for dim in range(rank)]
        else:
            low_valid_loc_range = [max(loc[i] - example_size[i] + border_shift, 0) for i in range(rank)]
            #             high_valid_loc_range = [min(loc[i] + example_size[i]//2,image_list[img_idx].shape[i])
            #                                     for i in range(rank)]
            high_valid_loc_range = \
                [min(loc[i] - border_shift, image_list[img_idx].shape[i] - example_size[i] - border_shift)
                 for i in range(rank)]
            rnd_loc = [np.random.randint(low_valid_loc_range[dim], high_valid_loc_range[dim], size=n_examples)
                       if high_valid_loc_range[dim] > low_valid_loc_range[dim] else np.zeros(n_examples, dtype=int)
                       for dim in range(rank)]
        for i in range(0, len(rnd_loc[1])):
            rnd_loc[1][i] = (image_list[img_idx].shape[1] - example_size[1]) // 2

        return rnd_loc

    assert n_examples > 0

    was_singular = False
    if isinstance(image_list, np.ndarray):
        image_list = [image_list]
        was_singular = True

    assert all([i_s >= e_s for i_s, e_s in zip(image_list[0].shape, example_size)]), \
        'Image must be bigger than example shape'
    assert (image_list[0].ndim - 1 == len(example_size) or image_list[0].ndim == len(example_size)), \
        'Example size doesnt fit image size'

    for i in image_list:
        if len(image_list) > 1:
            assert (
                i.ndim - 1 == image_list[0].ndim or i.ndim == image_list[0].ndim or i.ndim + 1 == image_list[0].ndim), \
                'Example size doesn''t fit image size'
            # assert all([i0_s == i_s for i0_s, i_s in zip(image_list[0].shape, i.shape)]), \
            #     'Image shapes must match'

    rank = len(example_size)

    # extract random examples from image and label

    examples = [[]] * len(image_list)
    y = [0] * n_examples

    for i in range(n_examples):
        rnd_loc = get_range(0)
        slicer = tuple([slice(rnd_loc[dim][i], rnd_loc[dim][i] + example_size[dim]) for dim in range(rank)])
        y[i] = loc[0] - rnd_loc[0][i]
        #         if y[i] >=100 or y[i] <=28:
        #             y[i] = 0
        #         else:
        #             y[i]= 1
        for j in range(len(image_list)):
            ex_img = image_list[j][slicer][np.newaxis]
            # concatenate and return the examples
            examples[j] = np.concatenate((examples[j], ex_img), axis=0) if (len(examples[j]) != 0) else ex_img

    if was_singular:
        return examples[0], y
    return examples, y


def pad_image_to_size(image, img_size=(64, 64, 64), loc=(2, 2, 2), **kwargs):
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # find image dimensionality
    rank = len(img_size)

    # create placeholders for new shape
    to_padding = [[0, 0] for _ in range(rank)]

    for i in range(rank):
        # for each dimensions find whether it is supposed to be cropped or padded
        if image.shape[i] < img_size[i]:
            if loc[i] == 0:
                to_padding[i][0] = (img_size[i] - image.shape[i])
                to_padding[i][1] = 0
            elif loc[i] == 1:
                to_padding[i][0] = 0
                to_padding[i][1] = (img_size[i] - image.shape[i])
            else:
                to_padding[i][0] = (img_size[i] - image.shape[i]) // 2 + (img_size[i] - image.shape[i]) % 2
                to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            to_padding[i][0] = 0
            to_padding[i][1] = 0

    return np.pad(image, to_padding, **kwargs)

def preprocess_mip_for_slice_detection(image, spacing, target_spacing, min_height=512,
                                       min_width=512):
    image = zoom(image, [spacing[2] / target_spacing, spacing[0] / target_spacing])
    image = reduce_hu_intensity_range(image)

    v = min_height if image.shape[0] <= min_height else 2 * min_height
    img_size = [v, min_width]
    padded_image = pad_image_to_size(image, img_size, loc=[1, -1], mode='constant')
    padded_image = padded_image[:v, :min_width] - 128
    return padded_image[np.newaxis, :, :, np.newaxis], image


def preprocess_sitk_image_for_slice_detection_frontal(sitk_image, target_spacing=1, mode='frontal', min_height=512,
                                              min_width=512):
    warnings.filterwarnings('ignore') 
    spacing = sitk_image.GetSpacing()
    direction = sitk_image.GetDirection()
    dx = int(direction[0])
    dy = int(direction[4])
    dz = int(direction[8])

    image = sitk.GetArrayFromImage(sitk_image)[::dx, ::dy, ::dz]

    image_frontal, image_sagittal = extract_mip(image)

    if mode == 'sagittal':
        image = image_sagittal
    else:
        image = image_frontal

    return preprocess_mip_for_slice_detection(image, spacing, target_spacing, min_height,
                                              min_width)
    
def preprocess_sitk_image_for_slice_detection_sagittal(sitk_image, target_spacing=1, mode='sagittal', min_height=512,
                                              min_width=512):
    warnings.filterwarnings('ignore') 
    spacing = sitk_image.GetSpacing()
    direction = sitk_image.GetDirection()
    dx = int(direction[0])
    dy = int(direction[4])
    dz = int(direction[8])

    image = sitk.GetArrayFromImage(sitk_image)[::dx, ::dy, ::dz]

    image_frontal, image_sagittal = extract_mip(image)

    if mode == 'sagittal':
        image = image_sagittal
    else:
        image = image_frontal

    return preprocess_mip_for_slice_detection(image, spacing, target_spacing, min_height,
                                              min_width)

def plot_mips(frontal_MIP, sagittal_MIP):
    plt.close()
    fig, axes = plt.subplots(1, 2)

    # Display images
    axes[0].imshow(np.flip(frontal_MIP), cmap='grey')
    axes[0].set_title('Frontal MIP')
    axes[1].imshow(np.flip(sagittal_MIP), cmap='grey')
    axes[1].set_title('Sagittal MIP')
    plt.tight_layout()
    plt.show()


# preprocess and save MIPs
def volume_preprocessing(root_dir, mode):
 
    paths = sorted(glob.glob(os.path.join(root_dir, "raw_data/volumes/*.nii.gz")))

    if mode == "sagittal":
        output_folder_results = os.path.join(root_dir, "raw_data/sagittal_MIPs") #Where to save highlighted image
        if not os.path.exists(output_folder_results):
            os.makedirs(output_folder_results)
    elif mode == "frontal":
        output_folder_results = os.path.join(root_dir, "raw_data/frontal_MIPs") #Where to save highlighted image
        if not os.path.exists(output_folder_results):
            os.makedirs(output_folder_results)
    os.chdir(output_folder_results)

    for i in range(len(paths)):
        exam_path = paths[i]
        sitk_image = sitk.ReadImage(exam_path)
        if mode == "sagittal":
            mip = preprocess_sitk_image_for_slice_detection(sitk_image, mode = "sagittal")[1]
            filename = os.path.splitext(exam_path)[0]
            filename =  os.path.splitext(filename)[0] 
            filename = filename.split("/")[-1]
            filename = filename + "sagittalMIP_LiTS.png"

            #plt.imshow(sag_img, cmap = 'gray')
            #plt.axis('off')
            plt.imsave(filename, np.flip(mip), cmap='gray')
            plt.clf()
            #print(f"Image {i} saved")
        elif mode == "frontal":
            mip = preprocess_sitk_image_for_slice_detection(sitk_image)[1]
            filename = os.path.splitext(exam_path)[0]
            filename =  os.path.splitext(filename)[0] 
            filename = filename.split("/")[-1]
            filename = filename + "frontalMIP_LiTS.png"

            #plt.imshow(sag_img, cmap = 'gray')
            #plt.axis('off')
            plt.imsave(filename, np.flip(mip), cmap='gray')
            plt.clf()
            #print(f"Image {i} saved")
