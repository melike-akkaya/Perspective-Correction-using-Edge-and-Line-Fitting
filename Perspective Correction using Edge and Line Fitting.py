# %% [markdown]
# # Perspective Correction using Edge and Line Fitting

# %% [markdown]
# Melike Akkaya
# 
# 2210356124

# %% [markdown]
# ## Introduction

# %% [markdown]
# In the age of digital documentation, capturing paper documents using mobile devices has become commonplace. However, these images often suffer from perspective distortions, folds, and other geometric inconsistencies that hinder automated processing and readability. This project focuses on addressing these issues by implementing a perspective correction system using edge detection, line fitting, and geometric transformations.
# 
# The goal of this assignment is to detect and correct distortions in document images by identifying their structural boundaries and transforming them into a front-facing, undistorted view. To achieve this, I used two core techniques: Hough Transform for detecting line structures in edge-detected images, and RANSAC for robust line fitting in the presence of noise and outliers. Once the document’s quadrilateral shape is estimated, I applied geometric transformations to correct its perspective.
# 
# The effectiveness of my method is evaluated using the Structural Similarity Index, which quantitatively compares my corrected outputs with their corresponding ground truth images.

# %%
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# %% [markdown]
# ## Processing Data

# %% [markdown]
# Before applying line detection and perspective correction, it is essential to preprocess the input images to enhance their features and suppress noise. These steps improve the visibility of document edges and make later stages more effective and reliable.
# 
# 
# - Resize : Ensures uniformity in processing and helps reduce computational cost.
# 
# 
# - Increased Contrast : Emphasizes lighter regions (like white paper) and suppresses darker regions (like shadows or background).
# 
# 
# - Grayscale : Simplifies the data.
# 
# 
# - Gaussian Blur : Reduces noise.
# 
# 
# - Dilate : Bridges gaps between broken edges.
# Note: These operation is especially helpful when dealing with folded documents because it provides edge connectivity.
# 
# 
# - Canny Edges : Extracts the edges.

# %%
def preprocess(image_file, resize_dim, blur_kernel, morph_kernel, visualize):
    if visualize:
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    # LOADING AND RESIZING
    image = cv2.imread(image_file)
    image = cv2.resize(image, resize_dim)

    if visualize:
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

    # INCREASING THE CONTRAST
    image = cv2.convertScaleAbs(image, alpha=0.5)  # by setting alpha < 1, image is slightly darken
                                                   # highlights the document borders

    if visualize:
        axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Increased Contrast')
        axes[1].axis('off')

    # GRAYSCALE
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if visualize:
        axes[2].imshow(image, cmap='gray')
        axes[2].set_title('Grayscale Image')
        axes[2].axis('off')

    # GAUSSIAN BLUR
    image = cv2.GaussianBlur(image, blur_kernel, 0)

    if visualize:
        axes[3].imshow(image, cmap='gray')
        axes[3].set_title('Gaussian Blur')
        axes[3].axis('off')

    # DILATE
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    image = cv2.dilate(image, kernel)

    if visualize:
        axes[4].imshow(image, cmap='gray')
        axes[4].set_title('Dilate Applied')
        axes[4].axis('off')

    # CANNY EDGES
    edges = cv2.Canny(image, 50, 150)

    if visualize:
        axes[5].imshow(edges, cmap='gray')
        axes[5].set_title('Canny Edges')
        axes[5].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return image, edges


# %% [markdown]
# ## Hough Transform and RANSAC

# %% [markdown]
# The **Hough Transform** is a voting-based technique used to identify straight lines in an image. It transforms each edge pixel from the Cartesian image space $(x, y)$ into a parameter space $(\rho, \theta)$, where:
# 
# $$
# \rho = x \cdot \cos(\theta) + y \cdot \sin(\theta)
# $$
# 
# - $\rho$: the perpendicular distance from the origin to the line.  
# - $\theta$: the angle of the perpendicular from the origin to the line.
# 
# I defined:
# - $diagonal = \sqrt{H^2 + W^2}$, maximum possible distance
# - $\rho \in [-diagonal, diagonal]$, covering all line distances in the image
# - $\theta \in [0^\circ, 180^\circ]$, converted to radians.
# 
# Each edge pixel votes for all possible $(\rho, \theta)$ values using the formula above. Votes are accumulated in a 2D matrix called the **accumulator**. The peaks in this matrix correspond to potential straight lines in the original image.
# 

# %%
def hough_transform(image):
    height, width = image.shape

    # in a quadrilateral the maximum possible distance is diagonal
    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))

    # covering all line distances
    rhos = np.arange(-diagonal, diagonal + 1)
    
    # degress converted to radian
    thetas = np.deg2rad(np.arange(0, 180))

    # H : the accumulator array (rows: rhos, columns: thetas)
    # each cell H[r, t] counts how many edge pixels voted for that (rho,theta) pair
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    
    # get indices of edge pixels
    y_idxs, x_idxs = np.nonzero(image)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    for x, y in zip(x_idxs, y_idxs):
        rhos_values = x * cos_t + y * sin_t
        rhos_indices = np.round(rhos_values).astype(int) + diagonal
        for idx, r_idx in enumerate(rhos_indices):
            H[r_idx, idx] += 1

    return H, rhos, thetas

# %% [markdown]
# The Hough Transform may produce multiple noisy or redundant lines. To address this issue, I used a variant of **RANSAC**. It works as follows:
# 
# 1. Randomly select a candidate line as a model.
# 2. Identify other lines that are similar (within defined thresholds for $\rho$ and $\theta $)
# 3. If a model has enough **inliers**, retain it and discard the inliers from future iterations.
# 4. Repeat until a set number of lines are detected or the pool is exhausted.
# 
# This step clusters similar rows and filters out spurious detections, producing a cleaner and more consistent set of document edge rows.

# %%
def ransac(hough_lines, num_iterations, theta_threshold, rho_threshold, min_inliers, max_lines):
    detected_lines = []

    while len(hough_lines) >= min_inliers and len(detected_lines) < max_lines:
        # keep trying to detect lines until either:
        #   - not enough data remains (min_inliers)
        #   - you've reached the maximum number of lines (max_lines)

        best_model = None     # stores the best line candidate in current round
        best_inliers = set()  # stores supporting lines (inliers) for that model

        for _ in range(num_iterations):
            if not hough_lines:
                break

            hypothesis_model = random.choice(list(hough_lines))
            candidate_theta, candidate_rho = hypothesis_model

            inliers = set()
            for line in hough_lines:
                theta, rho = line
                if abs(theta - candidate_theta) < theta_threshold and abs(rho - candidate_rho) < rho_threshold:
                    # if both angle and distance are within the thresholds, they are counted as inliers
                    inliers.add(line)

            if len(inliers) > len(best_inliers) and len(inliers) >= min_inliers:
                best_model = (candidate_theta, candidate_rho)
                best_inliers = inliers

        if best_model is None:
            break

        detected_lines.append((best_model, best_inliers))
        hough_lines -= best_inliers

    return detected_lines

# %% [markdown]
# After adding both RANSAC and Hough Transform functionality, a pipeline needed:
# 
#     (1) Hough Transform – to detect candidate lines from the edge map.
# 
#     (2) Thresholding – to keep only the strong line detections.
# 
# <p align="center">Reconstructing the line equation in Cartesian coordinates using:</p>
# 
# $$
# x = \rho \cdot \cos(\theta) , y = \rho \cdot \sin(\theta)
# $$
# 
#     (3) RANSAC – to filter these lines for consistency and robustness.
# 
#     (4) Visualization

# %%
def hough_transform_with_ransac(image_file, resize_dim, image, edges, threshold, visualize=False):
    H, rhos, thetas = hough_transform(edges)
    
    candidate_idx = np.argwhere(H > threshold)
    hough_lines = set()
    image_with_hough_lines = np.copy(image)
    
    for idx in candidate_idx:
        rho_idx, theta_idx = idx
        rho = rhos[rho_idx]
        theta = np.rad2deg(thetas[theta_idx]) # converts theta from radians to degrees
        hough_lines.add((theta, rho))
        
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        cv2.line(image_with_hough_lines, (x1, y1), (x2, y2), (255, 0, 0), 2) # blue, will apear white in grayscale images
    
    if visualize:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        orgImage = cv2.imread(image_file)
        orgImage = cv2.resize(orgImage, resize_dim)
        axes[0].imshow(cv2.cvtColor(orgImage, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(H, cmap='jet', aspect='auto') # a "jet" colormap and logarithmic scaling for visibility
        axes[1].set_title('Hough Accumulator')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(image_with_hough_lines, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Detected Lines with Hough Transformation')
        axes[2].axis('off')
    
    detected_lines = ransac(hough_lines, num_iterations=100, theta_threshold=5, rho_threshold=5, min_inliers=2, max_lines=10)
    
    if visualize:
        lines_image_ransac = np.copy(image)
        for (theta, rho), _ in detected_lines:
            a = np.cos(np.deg2rad(theta))
            b = np.sin(np.deg2rad(theta))
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(lines_image_ransac, (x1, y1), (x2, y2), (255, 0, 0), 2) # blue, will apear white in grayscale images
        
        axes[3].imshow(cv2.cvtColor(lines_image_ransac, cv2.COLOR_BGR2RGB))
        axes[3].set_title(f'Detected Lines with RANSAC')
        axes[3].axis('off')
        plt.show()
    
    return detected_lines


# %% [markdown]
# ## Geometric Transformations and SSIM

# %% [markdown]
# Once reliable line segments have been extracted using **Hough Transform** and **RANSAC**, the next objective is to determine the actual boundary of the document by identifying a **quadrilateral structure**. This involves extracting the document's corner points, validating their geometric correctness, warping the image to a frontal view, and evaluating the result using the **Structural Similarity Index Measure**.
# 
# ---
# 
# ### Finding the Intersections
# 
# The filtered lines from the RANSAC process are first converted from their **polar representation** $(\theta, \rho)$ into **Cartesian line equations** in the general form:
# 
# $$
# Ax + By + C = 0
# $$
# 
# Where:
# - $A = \cos(\theta)$  
# - $B = \sin(\theta)$  
# - $C = -\rho$
# 
# Each line is now represented by the coefficients $(A, B, C)$, which define the orientation and position of the line in the 2D image plane.
# 
# 
# To find the intersection point of two lines $L_1: A_1x + B_1y + C_1 = 0$ and $L_2: A_2x + B_2y + C_2 = 0$, I solved the system of linear equations:
# 
# $$
# \begin{cases}
# A_1x + B_1y + C_1 = 0 \\
# A_2x + B_2y + C_2 = 0
# \end{cases}
# $$
# 
# Using the method of determinants, the solution for $(x, y)$ is:
# 
# $$
# \text{denominator} = A_1 B_2 - A_2 B_1
# $$
# 
# If the denominator is close to **zero**, then:
# 
# - The two lines are **parallel** or **overlapping**.
# - This is because the determinant of the coefficient matrix is zero, which indicates the system of equations does **not have a unique solution**.
# - In geometric terms, it means the lines have the **same or proportional direction vectors** and thus never intersect (unless they are the same line).
# 
# If the lines are not parallel, the intersection point is:
# 
# $$
# x = \frac{B_1 C_2 - B_2 C_1}{A_1 B_2 - A_2 B_1}, \quad
# y = \frac{C_1 A_2 - C_2 A_1}{A_1 B_2 - A_2 B_1}
# $$

# %%
def find_intersections(lines, image_shape):
    # instances in lines have the following form : (A, B, C) where Ax + By + C = 0
    intersections = []
    H, W = image_shape[:2]

    n = len(lines)
    for i in range(n):
        A1, B1, C1 = lines[i]
        for j in range(i+1, n):
            A2, B2, C2 = lines[j]

            denom = A1 * B2 - A2 * B1
            # if the determinant is very close to zero --> the lines are parallel or identical --> they do not intersect
            if abs(denom) < 1e-8:
                continue

            x = (B1 * C2 - B2 * C1) / denom
            y = (C1 * A2 - C2 * A1) / denom

            # Check if the intersection is within image bounds
            if 0 <= x < W and 0 <= y < H:
                intersections.append((x, y))

    return intersections

# %% [markdown]
# ### Controlling if Points Form a Convex Quadrilateral
# 
# This function checks whether a given set of four points forms a **convex quadrilateral**. This is important because applying a perspective transformation requires the detected corners to form a convex shape.
# 
# 1. **Centroid Calculation**  
#    The average point of the four corners is calculated to find out the relative position of each point from the center as:
#    $$
#    C_x = \frac{1}{4} \sum_{i=1}^{4} x_i,\quad C_y = \frac{1}{4} \sum_{i=1}^{4} y_i
#    $$
# 
# 3. **Angle Calculation**  
#    For each point, I calculated the angle between the vector from the centroid to the point and the x-axis using:
# 
#    $$
#    \theta_i = \text{atan2}(y_i - C_y,\ x_i - C_x)
#    $$
# 
#    This helps me **sort the points in a consistent order**.
# 
# 4. **Convexity Check Using Cross Product**  
#    For every triplet of consecutive points (O, A, B), I calculated the **cross product**:
#    $$
#    \text{cross}(O, A, B) = (A_x - O_x)(B_y - O_y) - (A_y - O_y)(B_x - O_x)
#    $$
# 
#    - If all cross products have the **same sign** (all positive or all negative), the quadrilateral is **convex**.
#    - If the sign changes, it means the shape makes a "dent" — it's **not convex**.

# %%
def is_convex_quadrilateral(points):
    # if there are less or more than 4 points --> cannot be a quadrilateral
    if len(points) != 4:
        return False
    
    points = np.array(points, dtype=np.float32)
    cx, cy = np.mean(points, axis=0)

    angles = []
    for p in points:
        dx = p[0] - cx
        dy = p[1] - cy
        angles.append(np.arctan2(dy, dx))

    # the points are sorted based on the angles so that they are ordered consistently (clockwise or counterclockwise)
    sorted_indices = np.argsort(angles)
    points_sorted = points[sorted_indices]

    # 2D cross product of vectors OA and OB
    def cross(o, a, b):
        return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])

    sign = None
    for i in range(4):
        o = points_sorted[i]
        a = points_sorted[(i+1) % 4]
        b = points_sorted[(i+2) % 4]
        c = cross(o, a, b)
        if c != 0:                       # if the cross product is non-zero,
            if sign is None:             # the function checks its sign
                sign = np.sign(c)
            elif np.sign(c) != sign:     # if any cross product has a different sign than the others,
                return False             # the quadrilateral is not convex

    return True

# %% [markdown]
# ### Ordering Points Clockwise
# 
# Before applying a perspective warp, it is essential to get the four corner points of a document are ordered consistently (always counter clockwise or always clockwise). Before adding this step sometimes I detect the quadrilateral correctly but warpped it in wrong direction because the points are counter clockwise ordered. The points to be passed in the following order: **Top-left → Top-right → Bottom-right → Bottom-left**

# %%
def order_points_clockwise(points):
    points = np.array(points, dtype="float32")

    reordered_points = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum (x + y),
    # the bottom-right has the largest sum
    s = points.sum(axis=1)
    reordered_points[0] = points[np.argmin(s)]  # top-left
    reordered_points[2] = points[np.argmax(s)]  # bottom-right

    # the top-right point has the smallest difference (x - y),
    # the bottom-left has the largest difference
    diff = np.diff(points, axis=1)
    reordered_points[1] = points[np.argmin(diff)]  # top-right
    reordered_points[3] = points[np.argmax(diff)]  # bottom-left

    return reordered_points


# %% [markdown]
# ### Finding the Best Quadrilateral
# It is possible to create more than one quadrilateral from selected points. Eventhough there could be some exceptional cases, selecting the largest area convex quadrilateral would be the best practise.

# %%
def find_best_quadrilateral(intersections):
    if len(intersections) < 4:
        return None

    best_area = 0
    best_quad = None

    for combo in itertools.combinations(intersections, 4):   # taking all combinations of points
        if is_convex_quadrilateral(combo):                   # controling if this random 4 point can generate a quadrilateral
            points = np.array(combo, dtype=np.float32)
            area = cv2.contourArea(points)                   # calculating the area
            if area > best_area:                             # keeping track of the largest area
                best_area = area
                best_quad = order_points_clockwise(points)   # fixing the orders of points

    return best_quad

# %% [markdown]
# ### Warping the Document
# In this step, the goal is to transform the detected quadrilateral region into a front-facing rectangle. This process, known as perspective transformation or warping, helps eliminate distortions caused by the camera angle.
# 
# This is done using a **homography matrix**, which defines a projective transformation between two planes. In the updated code, we manually compute the homography matrix using a system of linear equations, which is then used to apply the perspective transformation to the image.
# 
# - The math behind it:
# 
# The transformation matrix \( M \) is a 3×3 matrix that maps input coordinates \((x, y)\) to new coordinates \((x', y')\):
# 
# $$
# \begin{bmatrix}
# x' \\
# y' \\
# w'
# \end{bmatrix}
# =
# \begin{bmatrix}
# h_{11} & h_{12} & h_{13} \\
# h_{21} & h_{22} & h_{23} \\
# h_{31} & h_{32} & h_{33}
# \end{bmatrix}
# \cdot
# \begin{bmatrix}
# x \\
# y \\
# 1
# \end{bmatrix}
# $$
# 
# To get the actual pixel coordinates in the output image:
# 
# $$
# x_{\text{new}} = \frac{x'}{w'}, \quad y_{\text{new}} = \frac{y'}{w'}
# $$
# 
# The transformation matrix is computed using the function `compute_perspective_transform()`. The function solves for the matrix that satisfies the equations that map the source points to the destination points.
# 
# Once the homography matrix \( M \) is computed, the function `apply_perspective_transform()` applies the matrix to the original image. For each pixel in the output image, it maps the corresponding source pixel using the inverse of the homography matrix. This way, the image is warped correctly, preserving the alignment of the document's content.
# 

# %%
def compute_perspective_transform(src, dst):
    A = np.zeros((8, 8), dtype=np.float32)
    B = np.zeros((8, 1), dtype=np.float32)

    for i in range(4):
        x_s, y_s = src[i]
        x_d, y_d = dst[i]

        A[2 * i] = [x_s, y_s, 1, 0, 0, 0, -x_d * x_s, -x_d * y_s]
        A[2 * i + 1] = [0, 0, 0, x_s, y_s, 1, -y_d * x_s, -y_d * y_s]

        B[2 * i] = x_d
        B[2 * i + 1] = y_d

    M = np.linalg.solve(A, B)
    M = np.append(M, 1.0)  # 3x3
    M = M.reshape(3, 3)

    return M

# %%
def apply_perspective_transform(image, M, output_size):
    h, w = output_size[1], output_size[0]
    warped = np.zeros((h, w, image.shape[2]), dtype=np.uint8)

    M_inv = np.linalg.inv(M) # destination to source

    for y_d in range(h):
        for x_d in range(w):
            src = np.dot(M_inv, [x_d, y_d, 1])
            src /= src[2]
            x_s, y_s = int(round(src[0])), int(round(src[1]))

            if 0 <= x_s < image.shape[1] and 0 <= y_s < image.shape[0]:
                warped[y_d, x_d] = image[y_s, x_s]

    return warped

# %%
def warp_document(image, quad, output_size):
    try: 
        w, h = output_size
        dst_pts = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        quad = np.array(quad, dtype=np.float32)
        M = compute_perspective_transform(quad, dst_pts)
        warped = apply_perspective_transform(image, M, (w, h))
        return warped   
    except  Exception as e:
        print("Invalid quadrilateral provided for perspective transform.")


# %% [markdown]
# ### Normalizing Colors
# After all other steps the SSIM value is still too low. One reason may be that images still suffer from lighting or poor contrast. To make text and backgrounds more distinguishable, I applied color normalization.
# 
# This function improves visual clarity by enhancing local contrast and balancing the brightness of the image. It uses the LAB color space, which separates lightness from color information. This way brightness is adjusted moreo without effecting color.

# %%
def normalize_colors(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)                # converting image from bgr to lab

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # Contrast Limited Adaptive Histogram Equalization to Lightness channel
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))

    image = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)             # convertint image back to bgr

    return image


# %% [markdown]
# ### Calculating SSIM Score
# 
# In this step all explained features are demonstrated and then the final result against the ground-truth using SSIM is evaluated.

# %%
def calculate_ssim_score (distoreted_image_path, digital_image_path, resize_dim, detected_lines, visualize=True):
    if not detected_lines:
        print("No lines detected.")
        return None

    distorted_image = cv2.imread(distoreted_image_path)
    distorted_image = cv2.resize(distorted_image, resize_dim)
    digital_image = cv2.imread(digital_image_path)

    # converting each detected (theta, rho) into line coefficients (A, B, C)
    line_coefficients = []
    for (theta, rho), _ in detected_lines:
        A = np.cos(np.deg2rad(theta))
        B = np.sin(np.deg2rad(theta))
        C = -rho
        line_coefficients.append((A, B, C))

    intersections = find_intersections(line_coefficients, distorted_image.shape)
    best_quad = find_best_quadrilateral(intersections)

    if best_quad is not None:

        if visualize:
            image_with_quad = distorted_image.copy()
            quad_int = best_quad.astype(int)
            for i in range(4):
                pt1 = tuple(quad_int[i])
                pt2 = tuple(quad_int[(i + 1) % 4])
                cv2.line(image_with_quad, pt1, pt2, (0, 255, 0), 3)

            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(image_with_quad, cv2.COLOR_BGR2RGB))
            plt.title('Detected Quadrilateral')
            plt.axis('off')

        distorted_image = warp_document(distorted_image, best_quad, output_size=resize_dim)
        distorted_image = normalize_colors(distorted_image)
        
        # digital and distoreted image should have same size
        distorted_image = cv2.resize(distorted_image, (digital_image.shape[1], digital_image.shape[0]))

        # grayscale version of images will be used for SSIM computation
        original_gray = cv2.cvtColor(digital_image, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(original_gray, processed_gray, full=True) # diff will not be used

        if visualize:
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB))
            plt.title('Scanned View')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(digital_image, cv2.COLOR_BGR2RGB))
            plt.title('Ground Truth')
            plt.axis('off')

            plt.show()

        print(f"SSIM between distorted image and digital image: {score:.4f}")
        return score

    else:
        print("No suitable quadrilateral found.")
        return None


# %% [markdown]
# ## Virtual Results

# %% [markdown]
# ### Curved

# %% [markdown]
# #### First Experiment

# %%
image_path_curved = 'asset/Warpdoc/distorted/curved/0001.jpg'

# %%
image_curved, edges_curved = preprocess(image_path_curved, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_curved = hough_transform_with_ransac(image_path_curved, resize_dim=(1024,1024), image=image_curved, edges=edges_curved, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_curved,
    digital_image_path='asset/Warpdoc/digital/curved/0001.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_curved,
    visualize=True
)

# %% [markdown]
# #### Second Experiment

# %%
image_path_curved_2 = 'asset/WarpDoc/distorted/curved/0022.jpg'

# %%
image_curved_2, edges_curved_2 = preprocess(image_path_curved_2, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_curved_2 = hough_transform_with_ransac(image_path_curved_2, resize_dim=(1024,1024), image=image_curved_2, edges=edges_curved_2, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_curved_2,
    digital_image_path='asset/Warpdoc/digital/curved/0022.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_curved_2,
    visualize=True
)

# %% [markdown]
# #### Third Experiment

# %%
image_path_curved_3 = 'asset/WarpDoc/distorted/curved/0095.jpg'

# %% [markdown]
# As can be seen in the attachment below containing the preprocessing steps, in this example the edges were not able to be highlighted due to the structure of the image.

# %%
image_curved_3, edges_curved_3 = preprocess(image_path_curved_3, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %% [markdown]
# The edges could not be detected correctly even when using exactly the same coefficient values ​​as the previous two examples.

# %%
detected_lines_curved_3 = hough_transform_with_ransac(image_path_curved_3, resize_dim=(1024,1024), image=image_curved_3, edges=edges_curved_3, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_curved_3,
    digital_image_path='asset/Warpdoc/digital/curved/0023.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_curved_3,
    visualize=True
)

# %% [markdown]
# ### Fold

# %% [markdown]
# #### First Experiment

# %%
image_path_fold = 'asset/Warpdoc/distorted/fold/0003.jpg'

# %%
image_fold, edges_fold = preprocess(image_path_fold, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_fold = hough_transform_with_ransac(image_path_fold, resize_dim=(1024,1024), image=image_fold, edges=edges_fold, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_fold,
    digital_image_path='asset/Warpdoc/digital/fold/0003.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_fold,
    visualize=True
)

# %% [markdown]
# #### Second Experiment

# %%
image_path_fold_2 = 'asset/Warpdoc/distorted/fold/0045.jpg'

# %%
image_fold_2, edges_fold_2 = preprocess(image_path_fold_2, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_fold_2 = hough_transform_with_ransac(image_path_fold_2, resize_dim=(1024,1024), image=image_fold_2, edges=edges_fold_2, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_fold_2,
    digital_image_path='asset/Warpdoc/digital/fold/0045.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_fold_2,
    visualize=True
)

# %% [markdown]
# #### Third Experiment

# %%
image_path_fold_3 = 'asset/Warpdoc/distorted/fold/0052.jpg'

# %% [markdown]
# As can be seen in the attachment below containing the preprocessing steps, in this example the edges were not able to be highlighted due to the structure of the image.

# %%
image_fold_3, edges_fold_3 = preprocess(image_path_fold_3, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %% [markdown]
# The edges could not be detected correctly even when using exactly the same coefficient values ​​as the previous two examples.

# %%
detected_lines_fold_3 = hough_transform_with_ransac(image_path_fold_3, resize_dim=(1024,1024), image=image_fold_3, edges=edges_fold_3, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_fold_3,
    digital_image_path='asset/Warpdoc/digital/fold/0052.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_fold_3,
    visualize=True
)

# %% [markdown]
# ### Incomplete

# %% [markdown]
# #### First Experiment

# %%
image_path_incomplete = 'asset/WarpDoc/distorted/incomplete/0027.jpg'

# %%
image_incomplete, edges_incomplete = preprocess(image_path_incomplete, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_incomplete = hough_transform_with_ransac(image_path_incomplete, resize_dim=(1024,1024), image=image_incomplete, edges=edges_incomplete, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_incomplete,
    digital_image_path='asset/WarpDoc/digital/incomplete/0027.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_incomplete,
    visualize=True
)

# %% [markdown]
# #### Second Experiment

# %%
image_path_incomplete_2 = 'asset/WarpDoc/distorted/incomplete/0087.jpg'

# %%
image_incomplete_2, edges_incomplete_2 = preprocess(image_path_incomplete_2, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_incomplete_2 = hough_transform_with_ransac(image_path_incomplete, resize_dim=(1024,1024), image=image_incomplete_2, edges=edges_incomplete_2, threshold=80, visualize=True)

# %% [markdown]
# Although the structure of the edges was detected closely, the SSIM value achived is bad because the selected line as left vertical line is not correct.

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_incomplete_2,
    digital_image_path='asset/WarpDoc/digital/incomplete/0087.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_incomplete_2,
    visualize=True
)

# %% [markdown]
# #### Third Experiment

# %%
image_path_incomplete_3 = 'asset/WarpDoc/distorted/incomplete/0092.jpg'

# %%
image_incomplete_3, edges_incomplete_3 = preprocess(image_path_incomplete_3, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_incomplete_3 = hough_transform_with_ransac(image_path_incomplete_3, resize_dim=(1024,1024), image=image_incomplete_3, edges=edges_incomplete_3, threshold=80, visualize=True)

# %% [markdown]
# Although the structure of the edges was detected properly, the SSIM value achived is bad because the selected borders include the outside area.

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_incomplete_3,
    digital_image_path='asset/Warpdoc/digital/incomplete/0092.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_incomplete_3,
    visualize=True
)

# %% [markdown]
# ### Perspective

# %% [markdown]
# #### First Experiment

# %%
image_path_perspective = 'asset/WarpDoc/distorted/perspective/0027.jpg'

# %%
image_perspective, edges_perspective = preprocess(image_path_perspective, resize_dim=(1024,1024),blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)


# %%
detected_lines_perspective = hough_transform_with_ransac(image_path_perspective, resize_dim=(1024,1024), image=image_perspective, edges=edges_perspective, threshold=80, visualize=True)

# %% [markdown]
# When choosing the quadrilateral shape, the point in the lower left corner could not be determined correctly because I took the quadrilateral with the largest area.

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_perspective,
    digital_image_path='asset/WarpDoc/digital/perspective/0027.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_perspective,
    visualize=True
)

# %% [markdown]
# I repeated this test with a larger threshold to get rid of the false line detected below.

# %%
detected_lines_perspective = hough_transform_with_ransac(image_path_perspective, resize_dim=(1024,1024), image=image_perspective, edges=edges_perspective, threshold=100, visualize=True)

# %% [markdown]
# With a larger threshold, the wrong line was eliminated, the corners of the paper were detected perfectly and the SSIM value was increased.

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_perspective,
    digital_image_path='asset//WarpDoc/digital/perspective/0027.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_perspective,
    visualize=True
)

# %% [markdown]
# #### Second Experiment

# %%
image_path_perspective_2 = 'asset/WarpDoc/distorted/perspective/0064.jpg'

# %%
image_perspective_2, edges_perspective_2 = preprocess(image_path_perspective_2, resize_dim=(1024,1024),blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)


# %%
detected_lines_perspective_2 = hough_transform_with_ransac(image_path_perspective_2, resize_dim=(1024,1024), image=image_perspective_2, edges=edges_perspective_2, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_perspective_2,
    digital_image_path='asset/WarpDoc/digital/perspective/0064.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_perspective_2,
    visualize=True
)

# %% [markdown]
# #### Third Experiment

# %%
image_path_perspective_3 = 'asset//WarpDoc/distorted/perspective/0092.jpg'

# %% [markdown]
# As can be seen in the attachment below containing the preprocessing steps, in this example the edges were not able to be highlighted due to the structure of the image.

# %%
image_perspective_3, edges_perspective_3 = preprocess(image_path_perspective_3, resize_dim=(1024,1024),blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)


# %% [markdown]
# The edges could not be detected correctly even when using exactly the same coefficient values ​​as the previous two examples.

# %%
detected_lines_perspective_3 = hough_transform_with_ransac(image_path_perspective_3, resize_dim=(1024,1024), image=image_perspective_3, edges=edges_perspective_3, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_perspective_3,
    digital_image_path='asset/Warpdoc/digital/perspective/0092.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_perspective_3,
    visualize=True
)

# %% [markdown]
# ### Random

# %% [markdown]
# #### First Experiment

# %%
image_path_random = 'asset/Warpdoc/distorted/random/0078.jpg'

# %%
image_random, edges_random = preprocess(image_path_random, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_random = hough_transform_with_ransac(image_path_random, resize_dim=(1024,1024), image=image_random, edges=edges_random, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_random,
    digital_image_path='asset/Warpdoc/digital/random/0078.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_random,
    visualize=True
)

# %% [markdown]
# As can be seen from upper experiment result the selected threshold value is not suitable for "Random" images because of its structure and calculated SSIM value is unreliable because of the white areas. I rerun this experiements with a smaller threshold value.

# %%
detected_lines_random = hough_transform_with_ransac(image_path_random, resize_dim=(1024,1024), image=image_random, edges=edges_random, threshold=60, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_random,
    digital_image_path='asset/Warpdoc/digital/random/0078.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_random,
    visualize=True
)

# %% [markdown]
# #### Second Experiment

# %%
image_path_random_2 = 'asset/WarpDoc/distorted/random/0067.jpg'

# %%
image_random_2, edges_random_2 = preprocess(image_path_random_2, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_random_2 = hough_transform_with_ransac(image_path_random_2, resize_dim=(1024,1024), image=image_random_2, edges=edges_random_2, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_random_2,
    digital_image_path='asset/WarpDoc/digital/random/0067.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_random_2,
    visualize=True
)

# %% [markdown]
# Eventhough the upper result is acceptable to control the most appropriate threshold value for random images, this test is reruned with a smaller threshold value.

# %%
detected_lines_random_2 = hough_transform_with_ransac(image_path_random_2, resize_dim=(1024,1024), image=image_random_2, edges=edges_random_2, threshold=60, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_random_2,
    digital_image_path='asset/WarpDoc/digital/random/0067.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_random_2,
    visualize=True
)

# %% [markdown]
# SSIM result is a little bit increased. I can assume that I need smaller threshold values for "Random" images.

# %% [markdown]
# #### Third Experiment

# %%
image_path_random_3 = 'asset/Warpdoc/distorted/random/0052.jpg'

# %%
image_random_3, edges_random_3 = preprocess(image_path_random_3, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)

# %%
detected_lines_random_3 = hough_transform_with_ransac(image_path_random_3, resize_dim=(1024,1024), image=image_random_3, edges=edges_random_3, threshold=60, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_random_3,
    digital_image_path='asset/Warpdoc/digital/random/0052.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_random_3,
    visualize=True
)

# %% [markdown]
# ### Rotate

# %% [markdown]
# #### First Experiment

# %%
image_path_rotate = 'asset/WarpDoc/distorted/rotate/0005.jpg'

# %%
image_rotate, edges_rotate = preprocess(image_path_rotate, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)


# %%
detected_lines_rotate = hough_transform_with_ransac(image_path_rotate, resize_dim=(1024,1024), image=image_rotate, edges=edges_rotate, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_rotate,
    digital_image_path='asset/WarpDoc/digital/rotate/0005.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_rotate,
    visualize=True
)

# %% [markdown]
# #### Second Experiment

# %%
image_path_rotate_2 = 'asset/WarpDoc/distorted/rotate/0097.jpg'

# %%
image_rotate_2, edges_rotate_2 = preprocess(image_path_rotate_2, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)


# %%
detected_lines_rotate_2 = hough_transform_with_ransac(image_path_rotate_2, resize_dim=(1024,1024), image=image_rotate_2, edges=edges_rotate, threshold=80, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_rotate_2,
    digital_image_path='asset/WarpDoc/digital/rotate/0097.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_rotate_2,
    visualize=True
)

# %% [markdown]
# The edges of the images in the Rotate images type are actually clear. It is noticeable that the two results above are greatly affected by the noisy data around. Based on this comment, I repeated these tests with a larger threshold value.

# %% [markdown]
# #### Third Experiment

# %%
detected_lines_rotate = hough_transform_with_ransac(image_path_rotate, resize_dim=(1024,1024), image=image_rotate, edges=edges_rotate, threshold=120, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_rotate,
    digital_image_path='asset/Warpdoc/digital/rotate/0005.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_rotate,
    visualize=True
)

# %% [markdown]
# #### Fourth Experiment

# %%
detected_lines_rotate_2 = hough_transform_with_ransac(image_path_rotate_2, resize_dim=(1024,1024), image=image_rotate_2, edges=edges_rotate, threshold=120, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_rotate_2,
    digital_image_path='asset/WarpDoc/digital/rotate/0097.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_rotate_2,
    visualize=True
)

# %% [markdown]
# By comparing fourth experiment with second experiment and third experiement with first experiement, it is easily seen that this approach works better for "Rotate" images when threshold value is larger.

# %% [markdown]
# #### Fifth Experiment

# %%
image_path_rotate_5 = 'asset/Warpdoc/distorted/rotate/0052.jpg'

# %%
image_rotate_5, edges_rotate_5 = preprocess(image_path_rotate_5, resize_dim=(1024,1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=True)


# %%
detected_lines_rotate_5 = hough_transform_with_ransac(image_path_rotate_5, resize_dim=(1024,1024), image=image_rotate_5, edges=edges_rotate_5, threshold=90, visualize=True)

# %%
calculate_ssim_score(
    distoreted_image_path=image_path_rotate_5,
    digital_image_path='asset/Warpdoc/digital/rotate/0052.jpg',
    resize_dim=(1024,1024),
    detected_lines=detected_lines_rotate_5,
    visualize=True
)

# %% [markdown]
# ## Statistical Results

# %% [markdown]
# A function was written to evaluate the approach for perspective correction for all data. Although the SSIM metric may be misleading in some cases, since there is no better alternative, it will be made on the SSIM value. The accepted coefficients were found by trial and error during the assignment and were accepted as the most optimal. 
# 
# There is a coefficient called threshold. This value is used to eliminate lines that received less votes than the threshold and is effected by the structure of the images. It will be given according to the results in the "Virtual Results" section.
# 
# - Curved Images --> threshold = 80,
# - Fold Images --> threshold = 80,
# - Incomplete Images --> threshold = 80,
# - Perspective Images --> threshold = 80,
# - Random Images --> threshold = 60,
# - Rotate Images --> threshold = 100

# %%
def evaluate_performance(folder, threshold, resize_dim=(1024, 1024), blur_kernel=(7, 7), morph_kernel=(7, 7), visualize=False):
    distorted_dir = f"asset/WarpDoc/distorted/{folder}/"
    digital_dir = f"asset/WarpDoc/digital/{folder}/"
    ssim_scores = []
    processed_count = 0
    skipped_count = 0

    print("="*50)
    print("Starting SSIM Evaluation")
    print(f"Distorted images from: {distorted_dir}")
    print(f"Digital image from: {digital_dir}")
    print("="*50)

    image_files = sorted([f for f in os.listdir(distorted_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for image_file in tqdm(image_files, desc="Processing images"):
        distorted_path = os.path.join(distorted_dir, image_file)
        digital_path = os.path.join(digital_dir, image_file)

        try:
            image, edges = preprocess(distorted_path, resize_dim, blur_kernel, morph_kernel, visualize)

            detected_lines = hough_transform_with_ransac(distorted_path, resize_dim, image, edges, threshold, visualize)

            score = calculate_ssim_score(distorted_path, digital_path, resize_dim, detected_lines, visualize)

            if score is not None:
                ssim_scores.append(score)
                processed_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            print(f"Skipping {image_file} due to error: {e}")
            skipped_count += 1

    ssim_array = np.array(ssim_scores)

    print(f"\nSSIM Evaluation Results for {folder}")
    print("="*50)
    print(f"Total images processed: {processed_count}")
    print(f"Total images skipped:   {skipped_count}")
    print(f"Average SSIM:           {np.mean(ssim_array):.4f}")
    print(f"Median SSIM:            {np.median(ssim_array):.4f}")
    print(f"Standard Deviation:     {np.std(ssim_array):.4f}")
    print(f"Max SSIM:               {np.max(ssim_array):.4f}")
    print(f"Min SSIM:               {np.min(ssim_array):.4f}")
    print("="*50)

    plt.figure(figsize=(10, 5))
    plt.hist(ssim_array, bins=20, edgecolor='black')
    plt.title(f"SSIM Score Distribution for {folder}")
    plt.xlabel("SSIM Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Curved

# %%
evaluate_performance(folder='curved', threshold=80)

# %% [markdown]
# ### Fold

# %%
evaluate_performance(folder='fold', threshold=80)

# %% [markdown]
# ### Incomplete

# %%
evaluate_performance(folder='incomplete', threshold=80)

# %% [markdown]
# ### Perspective

# %%
evaluate_performance(folder='perspective', threshold=80)

# %% [markdown]
# ### Random

# %%
evaluate_performance(folder='random', threshold=60)

# %% [markdown]
# ### Rotate

# %%
evaluate_performance(folder='rotate', threshold=100)


