# CSC320 Winter 2019
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic0 packages
import numpy as np

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()
    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    row = source_patches.shape[0]
    col = source_patches.shape[1]
    best_D = np.empty([row, col])
    if(odd_iteration):
        start_x = 0
        end_x = row
        start_y = 0
        end_y = col
        loop = 1
    else:
        start_x = row - 1
        end_x = -1
        start_y = col - 1
        end_y = -1
        loop = -1

    for i in range(start_x, end_x, loop):
        for j in range(start_y, end_y, loop):
            if(propagation_enabled):
                nlist = [np.nan, np.nan, np.nan]
                dscore = [np.nan, np.nan, np.nan]
                if(odd_iteration):
                    #add self to list
                    nlist[0] = source_patches[i][j]
                    dscore[0] = calculate_D(source_patches[i][j], target_patches[i+f[i][j][0]][j+f[i][j][1]])
                    if(i-1 >= 0):
                        if(i+f[i-1][j][0] < row):
                            nlist[1] = source_patches[i-1][j]
                            dscore[1] = calculate_D(source_patches[i][j], target_patches[i+f[i-1][j][0]][j+f[i-1][j][1]])
                    if(j-1 >= 0):
                        if(j+f[i][j-1][1] < col):
                            nlist[2] = source_patches[i][j-1]
                            dscore[2] = calculate_D(source_patches[i][j], target_patches[i+f[i][j-1][0]][j+f[i][j-1][1]])  
                else:                    
                    nlist[0] = source_patches[i][j]
                    dscore[0] = calculate_D(source_patches[i][j], target_patches[i+f[i][j][0]][j+f[i][j][1]])
                    if(i+1 < row):
                        if(i+f[i+1][j][0] < row):
                            nlist[1] = source_patches[i+1][j]
                            dscore[1] = calculate_D(source_patches[i][j], target_patches[i+f[i+1][j][0]][j+f[i+1][j][1]])
                    if(j+1 < col):
                        if(j+f[i][j+1][1] < col):
                            nlist[2] = source_patches[i][j+1]
                            dscore[2] = calculate_D(source_patches[i][j], target_patches[i+f[i][j+1][0]][j+f[i][j+1][1]])
                n = np.nanargmin(dscore)
                #if no change
                best_D[i,j] = dscore[n]
                if(n == 0):
                    new_f[i,j] = f[i,j]
                    f[i,j] = f[i,j]
                elif(n == 1):
                    if(odd_iteration):
                        new_f[i,j] = f[i-1, j]
                        f[i,j] = f[i-1, j]
                    else:
                        new_f[i,j] = f[i+1, j]
                        f[i,j] = f[i+1, j]
                else:
                    if(odd_iteration):
                        new_f[i,j] = f[i, j-1]
                        f[i,j] = f[i, j-1]
                    else:
                        new_f[i,j] = f[i, j+1]
                        f[i,j] = f[i, j+1]
            if(random_enabled):
                u = []
                dscore_random = []
                c = 0
                u.append([i,j])
                dscore_random.append(calculate_D(source_patches[i][j], target_patches[i+f[i][j][0]][j+f[i][j][1]]))
                while((alpha**c)*w > 1):
                    R = [int(np.random.uniform(-1, 1)*(alpha**c)*w), int(np.random.uniform(-1,1)*(alpha**c)*w)]
                    x = i + f[i][j][0] + R[0]
                    if(x < 0):
                        x = 0
                    elif(x >= row):
                        x = row - 1
                    y = j + f[i][j][1] + R[1]
                    if(y < 0):
                        y = 0
                    elif(y >= col):
                        y = col -1
                    u.append([x,y])
                    dscore_random.append(calculate_D(source_patches[i][j], target_patches[x][y]))
                    c+=1
                index = np.argmin(dscore_random)
                if(dscore_random[index] < best_D[i][j]):
                    best_D[i][j] = dscore_random[index]
                    new_f[i][j] = [u[index][0] - i, u[index][1] -j]
                    f[i,j] = new_f[i][j]

    #############################################
    return new_f, best_D, global_vars

def calculate_D(source_patches, target_patches):
    p1 = source_patches.flatten()
    p2 = target_patches.flatten()
    n = p1-p2
    n = n[~np.isnan(n)]
    return np.linalg.norm(n)
# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    x = np.array([i for i in range(target.shape[1])])
    x = np.broadcast_to(x, (target.shape[0], target.shape[1]))
    y = np.array([i for i in range(target.shape[0])])
    y = np.broadcast_to(y, (target.shape[1], target.shape[0]))
    y = y.T
    rec_source = np.dstack((y,x))
    rec_source = rec_source + f

    #############################################

    return target[rec_source[:,:,0], rec_source[:,:,1]]


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))