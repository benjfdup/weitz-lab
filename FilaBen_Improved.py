import cv2
from skimage.draw import line
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Prereq. Functions ---
def get_total_gradient(image):
  """
  Parameters:
  -----------
  image : np.array
    a numpy array representing a grayscale OpenCV2 image

  Output:
  -------
  arr : tuple<np.array>
    a 3d numpy array of shape (2, y, x), the first representing the normalized y gradient
    (index = 0) & the second representing the normalized x gradient (index = 1)
  """
  grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=-1)
  grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=-1)

  magnitude = np.sqrt(grad_y**2 + grad_x**2)

  grad_y /= magnitude
  grad_x /= magnitude

  grad_y = np.nan_to_num(grad_y, nan=0)
  grad_x = np.nan_to_num(grad_x, nan=0)

  arr = np.asfarray((grad_y, grad_x))
  return arr

# --- Primary Function ---
def Filament_detect_with_stats(image_path, pixel_size, verbose = False): # pixel_size is in microns/pixel
    #absolute_march_length = 1 # in microns #original parameter value
    absolute_march_length = 0.7 # in microns
    march_length = round(absolute_march_length / pixel_size)
    circle_growth_length = round(0.1 / pixel_size)# in pixels, how far circles will grow to avoid speckling
    minimum_threshold_area = max(round(0.12 /  (pixel_size**2)), 3)# in pixels, will be multiplied by a constant (maybe). Tweak this

    # Load a single widefield image in .tif format
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if original_image is None:
      raise Exception("Error loading the image. Check the image path.")

    # Normalize the original image for enhanced contrast
    normalized_image = cv2.normalize(original_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    #defining the image shape
    image_shape = gray_image.shape

    # Apply Gaussian blur to reduce noise and enhance filament structures
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) # might want to remove this gaussian smoothing

    # ultimately redundant, but fast enough for now
    normalized_blurred_image_gradients = get_total_gradient(blurred_image)

    edges = cv2.Canny(blurred_image, threshold1=30, threshold2=150)

    # Skeleton of Edges (ensures 1 pixel thick -- may be able to remove, test without)
    skeleton_of_edges = cv2.ximgproc.thinning(edges)

    start_points_array = np.column_stack(np.where(skeleton_of_edges != 0)) #Sorted, first by y, then by x.

    # Gradient values at start points (confirmed correct)
    grad_values_at_points = np.transpose(normalized_blurred_image_gradients[:, start_points_array[:, 0], start_points_array[:, 1]])

    #endpoints of relevant march
    yMax = image_shape[0] - 1
    xMax = image_shape[1] - 1

    end_points_array = grad_values_at_points * march_length + start_points_array #unclipped
    end_points_array = np.rint(np.clip(end_points_array, 0, [yMax, xMax])).astype(int) #clipped

    #holds the found filament mid-points
    filament_spex_list = []

    for i in range(start_points_array.shape[0]):

      start = start_points_array[i, :]
      end = end_points_array[i, :]

      #forms a small box around the start point
      box_l_bound_y = max(0, start[0] - march_length) #lower bound y of small box
      box_u_bound_y = min(yMax, start[0] + march_length) #upper bound y of small box

      box_l_bound_x = max(0, start[1] - march_length) #lower bound x of small box
      box_u_bound_x = min(xMax, start[1] + march_length) #upper bound x of small box

      #you can futher optimize this by getting the general direction of the line marched, and only doing those points in that quadrant of the small square.
      #these are the only edges with which we could possibly intersect...
      #that also could just be slower in most cases then just doing this tbh
      edges_contained = start_points_array[(start_points_array[:, 0] >= box_l_bound_y) & (start_points_array[:, 0] <= box_u_bound_y) & (start_points_array[:, 1] >= box_l_bound_x) & (start_points_array[:, 1] <= box_u_bound_x)]

      #coordinates on the "march_length" line, in the direction of the gradient.
      coords_on_line = np.array(line(start[0], start[1], end[0], end[1])).T[1:, :] #test this!
      #print(coords_on_line)

      if verbose:
        im_holder = np.zeros(image_shape)
        im_holder[coords_on_line] = 255
        #Add in gray background image here...
        plt.imshow(im_holder)
        plt.show()

      #nearest neighbor search is handled automatically via line
      #from: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays

      m = (coords_on_line[:, None] == edges_contained).all(-1).any(1)
      intersects = coords_on_line[m]

      if np.any(intersects):
        filament_spex_list.append((intersects[0, :] + start) / 2)

    filament_spex_array = np.unique(np.rint(filament_spex_list).astype(int), axis = 0)

    display_holder = np.zeros(image_shape)
    for point in filament_spex_array:
      y, x = point
      cv2.circle(display_holder, (x, y), circle_growth_length, 255, -1)

    display_holder_uint8 = cv2.convertScaleAbs(display_holder)
    central_skeleton = cv2.ximgproc.thinning(display_holder_uint8)

    #-----try to optimize this. Still a bit slow-----
    #Find connected components in the binary image
    _, labels = cv2.connectedComponents(central_skeleton)
    label_sizes = np.bincount(labels.ravel())

    #Create a mask to keep only clusters larger than the specified threshold
    filter_mask = label_sizes[labels] >= minimum_threshold_area
    filtered_image = np.zeros_like(central_skeleton)
    filtered_image[filter_mask] = 255
    inverted_filter = cv2.bitwise_not(filtered_image)

    #removes blobs from skeleton
    skeleton_de_blobed = central_skeleton - inverted_filter

    #drawing final contours
    contours_holder = np.zeros_like(skeleton_de_blobed)
    final_fils, _ = cv2.findContours(skeleton_de_blobed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contours_holder, final_fils, -1, 255, thickness=cv2.FILLED)

    plt.imshow(contours_holder, cmap = 'gray')

    # sorta redundant, but should be fast enough
    result_image = skeleton_de_blobed

    height, width = result_image.shape
    display = np.zeros((height, width, 3), dtype=np.uint8)

    display[:, :, 1] = result_image
    display[:, :, 0] = result_image
    display[:, :, 2] = result_image

    final_display = cv2.addWeighted(display, 1, normalized_image, 0.7, 0)

    total_length = 0 # in pixels, NOT PHYSICAL UNITS...
    num_of_filaments = len(final_fils)
    contour_num = 0

    contour_lengths = [] # IN PHYSICAL UNITS... NOT PIXELS

    for contour in final_fils:
      # Calculate the length of each contour
      length = cv2.arcLength(contour, closed=False) / 2 #only an approximation
      total_length += length
      contour_lengths.append(length * pixel_size)

    # Display the original image with skeletonized contours and print skeleton lengths
    print('Total Length:', total_length * pixel_size, 'microns')
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final_display, cv2.COLOR_BGR2RGB))
    plt.title("Filament Detection, Total Length: " + str(total_length * pixel_size))
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    return total_length * pixel_size, num_of_filaments, contour_lengths
