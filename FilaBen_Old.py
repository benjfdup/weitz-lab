#imports
import cv2
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
  array_tuple : tuple<np.array>
    a tuple of numpy arrays, the first representing the normalized y gradient
    (index = 0) & the second representing the normalized x gradient (index = 1)
  """
  grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=-1)
  #print('Scharr Y Gradient:', grad_y)
  grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=-1) #test kernel sizes
  #print('Scharr X Gradient:', grad_x)

  magnitude = np.sqrt(grad_y**2 + grad_x**2)

  grad_y /= magnitude
  grad_x /= magnitude

  grad_y = np.nan_to_num(grad_y, nan=0)
  grad_x = np.nan_to_num(grad_x, nan=0)

  array_tuple = (grad_y, grad_x)
  return array_tuple

def get_grad_at_point(grad_tuple, point):
  """
  Parameters:
  -----------
  grad_tuple : tuple<np.array>
    a tuple of 2 numpy arrays, the first of which represents a normalized y
    gradient, the second of which a normalized x gradient.

  point : np.array
    an array of shape (2,) which represents the point at which you would like
    the gradient. Note that point[0] should represent the y coordinate, counting
    DOWNWARDS from the top of the image, and that point[1] should represent the
    x coodinate.

  Output:
  -------
  gradient : np.array
    an array of shape (2,) which represents the gradient at the given point.
    Note that gradient[0] encodes the y component (where positive is downward),
    & gradient[1] encodes the x component.
  """
  total_grad_y, total_grad_x = grad_tuple
  y, x = point
  local_grad_y = total_grad_y[y, x]
  local_grad_x = total_grad_x[y, x]

  gradient = np.array([float(local_grad_y), float(local_grad_x)])
  #print('Grad found by lookup:', gradient)
  return gradient

def get_endpoint(start, direction, length, image_shape):
  #print("Gradient Direction:", direction)
  """
  Parameters:
  -----------
  start : np.array
    an array of shape (2,) denoting the starting point. Note that this should be
    of the form y, x to match OpenCV2 and NumPy's conventions

  direction : np.array
    an array of shape (2, ) denoting the direction of the gradient at the start
    point. This should have, as its first element, the y value of the gradient,
    where positive denotes DOWNWARD. The second element should be the x value of
    the gradient

  length : float
    the length of the gradient march.

  image_shape : tuple
    the shape of the image on which you are working. Note that image_shape[0]
    denotes the amount of rows (or y-axis pixels) whereas image_shape[1] denotes
    the amount of columns

  Output:
  -------
  endpoint : np.array
    the point that denotes the end of the line. This is an array of integers,
    of shape (2,) where endpoint[0] denotes the y coordinate on which the
    gradient march ends and endpoint[1] denotes the x coordinate on which it
    ends
  """

  magnitude = np.sqrt(direction[0]**2 + direction[1]**2)

  if magnitude == 0: #to prevent divide by zero
    return start

  direction /= np.sqrt(direction[0]**2 + direction[1]**2) #normalize, incase
  # not done prior
  y_max = image_shape[0] - 1
  x_max = image_shape[1] - 1

  endpoint = np.add(start, direction*length)
  endpoint = np.rint(endpoint)

  endpoint = endpoint.astype(int)

  endpoint[0] = max(0, min(endpoint[0], y_max)) # this should work,
  endpoint[1] = max(0, min(endpoint[1], x_max)) # check rigorously, however

  return endpoint

def line_set(start, end, image_shape, background_image = np.array([])): #TODO: Speed this up

  #background_image is only for testing purposes
  """
  Parameters:
  -----------
  start : np.array<int>
    an array of shape (2,) encoding the starting position of our line. All
    elements must be integers, as it encodes a pixel on a screen. start[0]
    denotes the y coordinate of the point, where a greater y moves the point
    DOWNWARD. start[1] denotes the x coordinate of the point.

  end : np.array<int>
    an array of shape (2,) encoding the ending position of our line. All
    elements must be integers, as it encodes a pixel on a screen, end[0]
    denotes the y coordinate of the point, where a greater y moves the point
    DOWNWARD. end[1] denotes the x coordinate of the point.

  image_shape : tuple
    a tuple representing the shape of the image. First coordinate represents the
    image height. The second represents the image width

  background_image = np.array([]) : np.array([])
    Only for bugtesting purposes. If not its default value, represents the
    background image behind the line representing the detected contour's normal.
    must be of shape specified by image_shape

  Output:
  -------
  points : set<tuple<int>>
    a set of all pixels included in the line, where each point is an tuple of
    that contains only integers. Each point's first element is its y coordinate,
    where y = 0 is the TOP of the image, and increasing y moves the point down.
    Each points second element is its x coordinate.
  """
  line_holder = np.zeros(image_shape) #array on which line is to be drawn
  #print('Start Point:', start)
  #print('Calculated End Point:', end)
  y0, x0 = start
  y1, x1 = end

  cv2.line(line_holder, (x0, y0), (x1, y1), 255, 1)
  yMax = image_shape[0] - 1
  xMax = image_shape[1] - 1

  #line_diagonal_holder = np.zeros_like(line_holder)

  #fill in all diagonal locations adjacent to line pixels

  buffered_line_array = line_holder #+ line_diagonal_holder

  if background_image.size != 0:
    height, width = line_holder.shape
    #print('line holder shape:', line_holder.shape)
    line_holder_green = np.zeros((height, width, 3), dtype=np.uint8)
    line_holder_green[:, :, 1] = line_holder
    background_image_red = np.zeros((height, width, 3), dtype=np.uint8)
    background_image_red[:, :, 0] = background_image
    image_shown = cv2.addWeighted(line_holder_green, 1, background_image_red, 1, 0)
   #image_shown = cv2.addWeighted(background_img_color, 1, line_holder_red, 1, 0)
    plt.imshow(image_shown)  # Convert BGR to RGB for correct color display
    plt.axis('off')  # Turn off axis labels
    plt.show()

  nonzero_indices_tuple = np.nonzero(buffered_line_array)
  points = {tuple(start), tuple(end)}
  yCoords, xCoords = nonzero_indices_tuple

  for i in range(len(yCoords)):
    points.add((int(yCoords[i]), int(xCoords[i])))

  return points

def find_closest_coordinate(pixel_list, start): #TODO: speed up with NumPy
  #Remove all points in the contour that contains start
  """
  Parameters:
  -----------
  pixel_list : list<tuple<int>>
    a list of tuples representing the list of pixels you are sorting through,
    looking for the closest to start

  start : tuple<int>
    tuple of lenght 2 representing the point which is used as the benchmark for
    measuring distance

  Output:
  -------
  closest_coordinate : tuple<int>
    tuple of length 2 representing the closest point to start point specified

  """

  pixel_array = np.array(pixel_list)

  # Calculate the Euclidean distances between the start coordinate and all coords in pixel_array
  distances = np.linalg.norm(pixel_array - start, axis=1)

  # Find the index of the closest coordinate
  closest_index = np.argmin(distances)

  # Retrieve and return the closest coordinate
  closest_coordinate = pixel_list[closest_index]
  return closest_coordinate

# --- Primary Functions ---

def Filament_detect_with_stats(image_path, pixel_size, verbose = False): #pixel_size is in microns/pixel
    absolute_march_length = 0.7 #microns, hyperparameter, but not in reality #ORIGINAL PARAMETER VALUE
    #absolute_march_length = 1
    march_length = round(absolute_march_length / pixel_size)
    circle_growth_length = round(0.1 / pixel_size)#in pixels, how far circles will grow to avoid speckling
    minimum_threshold_area = max(round(0.12 /  (pixel_size**2)), 3)# in pixels, will be multiplied by a constant (maybe). Tweak this

    # Load a single widefield image in .tif format
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if original_image is None:
      raise Exception("Error loading the image. Check the image path.")

    # Normalize the original image for enhanced contrast
    normalized_image = cv2.normalize(original_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    normalized_gray_image_gradients = get_total_gradient(gray_image)

    image_shape = gray_image.shape

    # Apply Gaussian blur to reduce noise and enhance filament structures
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) #might want to remove this gaussian smoothing

    # Perform Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1=30, threshold2=150)

    # Find contours of detected filaments
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #likely dont need this step...

    display_contours = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8) #or this...

    cv2.drawContours(display_contours, contours, -1, (0, 255, 0), 1)  # Green color, 2-pixel thickness

    if verbose:
      plt.imshow((cv2.addWeighted(display_contours, 1, normalized_image, 0.7, 0)))
      plt.show()

    # Create an empty image for skeletonized contours
    skeleton_of_contours = np.zeros_like(gray_image)

    # Skeletonize the detected contours and add them to the skeleton image
    mask = np.zeros_like(edges, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=1)
    skeleton_of_contours = cv2.ximgproc.thinning(mask)

    #print('Skeletonized Contours')
    start_points_array = np.column_stack(np.where(skeleton_of_contours != 0))
    start_points = [tuple(row) for row in start_points_array]

    filament_pixels = []
    pt_index = 0

    #print('-- Beginning gradient march --')
    for point in start_points: # this for loop needs to be replaced, if possible
      #print('Fraction Completed:', pt_index/len(start_points))
      unit_grad = get_grad_at_point(normalized_gray_image_gradients, point)
      endpoint = get_endpoint(point, unit_grad, march_length, image_shape)

      if not verbose:
        points_on_normal = line_set(point, endpoint, image_shape)
      if verbose: #Maybe change this up
        points_on_normal = line_set(point, endpoint, image_shape, cv2.cvtColor(display_contours, cv2.COLOR_BGR2GRAY))

      points_to_check_against = set(start_points[:pt_index] + start_points[pt_index+1:])

      common_coordinates = list(points_on_normal.intersection(points_to_check_against))
      common_coordinates_array = np.array(common_coordinates)

      common_coords_is_not_empty = len(common_coordinates_array) != 0

      if common_coords_is_not_empty:
        closest_point = find_closest_coordinate(common_coordinates_array, point)
        avgPos = (np.array(point) + closest_point) / 2
        avgPos = np.rint(avgPos)
        avgPixel = avgPos.astype(int)
        filament_pixels.append(avgPixel)
      pt_index += 1

    filamented_image = np.zeros_like(skeleton_of_contours)
    for point in filament_pixels:
      y, x = point  # Extract x and y coordinates from the array
      cv2.circle(filamented_image, (int(round(x)), int(round(y))), circle_growth_length, 255, -1) #make radius a function of pixel size maybe

    # skeletonize the new filamented_image
    central_skeleton = cv2.ximgproc.thinning(filamented_image)

    #Find connected components in the binary image
    _, labels = cv2.connectedComponents(central_skeleton)

    #Calculate the size of each connected component (cluster) by counting the
    #number of pixels in each label:
    label_sizes = np.bincount(labels.ravel())

    #Create a mask to keep only clusters larger than the specified threshold
    filter_mask = label_sizes[labels] >= minimum_threshold_area
    filtered_image = np.zeros_like(central_skeleton)
    filtered_image[filter_mask] = 255
    inverted_filter = cv2.bitwise_not(filtered_image)

    #removes blobs from skeleton
    skeleton_de_blobed = central_skeleton - inverted_filter

    contours_holder = np.zeros_like(skeleton_de_blobed)

    final_fils, _ = cv2.findContours(skeleton_de_blobed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contours_holder, final_fils, -1, 255, thickness=cv2.FILLED)

    if verbose:
      plt.imshow(contours_holder, cmap = 'gray')

    result_image = skeleton_de_blobed

    height, width = result_image.shape

    display = np.zeros((height, width, 3), dtype=np.uint8)

    display[:, :, 1] = result_image
    display[:, :, 0] = result_image
    display[:, :, 2] = result_image

    final_display = cv2.addWeighted(display, 1, normalized_image, 0.7, 0)

    total_length = 0 #in pixels, NOT PHYSICAL UNITS...

    num_of_filaments = len(final_fils)

    contour_num = 0

    contour_lengths = []

    for contour in final_fils:
      # Calculate the length of each contour
      length = cv2.arcLength(contour, closed=False) / 2 #only an approximation
      total_length += length
      contour_lengths.append(length * pixel_size)

    # Display the original image with skeletonized contours and print skeleton lengths
    print('Total Length:', total_length * pixel_size, 'microns') #convert this to not be pixels
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
    return (total_length * pixel_size, num_of_filaments, contour_lengths)
