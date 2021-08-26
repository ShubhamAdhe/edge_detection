import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

# Image related code #
#image = mpimg.imread('image.png')

# Source and destination points for perspective transform and inverse perspective transform
# Four source co-ordinates which mark a trapazoid of image/video where we want to detect edges
#4 points from source image (bottom left , upper left, bottom right, upper right)
src = np.float32(
    [[6,743],
    [201,430],
    [1500,755],
    [1000,443]])

# Four destination co-ordinates which represent a rectangle, to imitate the top view of image/video
dst = np.float32(
    [[10,600],
    [10, 100],
    [1200,600],
    [1200,100]])

# This function calculates perspective transform matrix
# and wrapped the image using this matrix
def perspective_warp(img):

    img_size = (img.shape[1],img.shape[0])

    # Compute perspective transform matrix M
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Wraps the given image - uses nearest neighbor interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped

# Calculates inverse perspective transform
# Use to visualize detected edges on actual video
def inv_perspective_warp(img):

    img_size = (img.shape[1],img.shape[0])

    # Calculates inverse perspective transform by switching input parameters (i.e. dst and src)
    M = cv2.getPerspectiveTransform(dst,src)
    
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size,flags=cv2.INTER_NEAREST)
    
    return warped

def pipeline(img, s_thresh=(100, 255), sx_thresh=(100, 255)):
    #img = undistort(img)
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) # adjust pixel intensity from 0(black) - 255 (white)
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    #plt.imshow(combined_binary)
    return combined_binary*255 # multiply by 255 to make all 1`s as 255, so that white color pixels will be dispalyed on image

def find_lane_pixels(binary_warped):
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) 

    # These will be the starting point for the left and right lines
    midpoint = np.int(binary_warped.shape[0]//2)
    leftx_base = midpoint
    rightx_base = midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    # In how many vertical sections the image is divided
    vert_windows = 9
    
    # Choose the number of horizontal sliding windows
    # In how many horizontal sections the image is divided
    horz_windows = 7
    
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 1000

    # Set height of window - based on vert_windows and image shape
    window_height = np.int(binary_warped.shape[0]//vert_windows)

    # Identify the x and y positions of all nonzero pixels (activated pixels) in the image
    nonzero = binary_warped.nonzero() #Returns a tuple of two arrays, one for each dimension of binary_warped
    nonzeroy = np.array(nonzero[0]) #1st touple is Y-coordinates (height of image) position of non zero pixel/elements (Y-coordinate's top is zeroth postion and bottom is maximum position)
    nonzerox = np.array(nonzero[1]) #2nd touple is X-coordinates (width of image) position of non zero pixel/elements
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    # First step through windows vertically
    for window in range(vert_windows):
        left_edge_detect_flag = 0
        right_edge_detect_flag = 0
        
        # Identify window boundaries in y, where height increases as we go from top (zero) to bottom (max)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        # Step through windows horizontally
        for hwindow in range(horz_windows):         
           # Identify window boundaries in x
           # Left half of the image (left to midpoint)
            win_xleft_low = binary_warped.shape[1]//2 - (hwindow+1)*2*margin
            win_xleft_high = binary_warped.shape[1]//2 - (hwindow)*2*margin
            
            # Right half of the image (midpoint to right)
            win_xright_low = binary_warped.shape[1]//2 + (hwindow)*2*margin
            win_xright_high = binary_warped.shape[1]//2 + (hwindow+1)*2*margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels position in x and y within the window
            # Here pixel positions are stored in 1D array
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
           
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # If you found pixel more than set threshold (minpix), recenter next window on this position
            # Find the first window where pixels are more than set threshold and check flag so that only 1st instance of edge detection is collected
            if ((len(good_left_inds) > minpix) and left_edge_detect_flag==0):
                # Append indices to the lists
                left_lane_inds.append(good_left_inds)
                left_edge_detect_flag = 1
                
            if ((len(good_right_inds) > minpix) and right_edge_detect_flag==0):
                # Append indices to the lists
                right_lane_inds.append(good_right_inds)
                right_edge_detect_flag = 1
                                
             # if we detect both windows from left and right where pixel are more than set threshold, exit the loop
            if ((left_edge_detect_flag == 1) and (right_edge_detect_flag == 1)):
                break
    
    # Concatenate the arrays of indices from whole image (previously was a list of lists of pixels)
    if (len(left_lane_inds) != 0):
        left_lane_inds = np.concatenate(left_lane_inds)
    else:
        left_lane_inds = [1000] #put a randmon array to avoid error
    
    if (len(right_lane_inds) != 0):
        right_lane_inds = np.concatenate(right_lane_inds)
    else:
        right_lane_inds = [1000] #put a randmon array to avoid error
    
    # Extract only interested left and right line pixel positions from whole image
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit` (To connect the white pixels with a line on left and right sides)
    # You're fitting for Y value, rather than X, because the lane lines in the warped image are nearly in vertical direction and may have the same x value for more than one y value.
    left_fit = np.polyfit(lefty, leftx, 2) #returns 3 coeffiecients of 2nd degree polynomial (i.e. in ay**2+by+c returns values of a, b and c in left_fit) (a, b ,c represents shape and position of curve)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(start=0, stop=binary_warped.shape[0]-1, num=binary_warped.shape[0] ) #array[start of the line, stop point of the line, num is number of intervals] (ploty gives a straight line or Y-coordinates)

    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] # 'left_fitx' has the X-coordinates position of point on the image which is the result of solving the polynomial (write the polynomial in the form of X=ay**2+by+c)
        
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    # Visualization #
    #Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    return out_img, (left_fitx, right_fitx), ploty

def get_curve(img, leftx, rightx, ploty):
    
    width_of_road = 3.7 # width of road in meters
    length_of_road = 30.5 # lenght of road in meters
    
    y_eval = np.max(ploty)
    
    # Measuring how long and wide the section of lane is that we're projecting in our warped image and converting them in meter per pixel
    ym_per_pix = length_of_road/720 # meters per pixel in y dimension (30.5m long)
    xm_per_pix = width_of_road/720 # meters per pixel in x dimension (3.7m wide)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)

    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculation of R_curve (radius of curvature)
    # Formula for radius of curvature: {[1 + (dx/dy)^2]^3/2} / (d2x/dy2),  where dx/dy = 2ay+b & d2x/dy2 = 2a
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, center)

def draw_lanes(img, left_fit, right_fit,ploty):
    
    color_img = np.zeros_like(img)
    
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    cv2.fillPoly(color_img, np.int_(points), (0,200,255))
    inv_perspective = inv_perspective_warp(color_img)
    #print(len(img))
    #print(len(inv_perspective))
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

# Image related code #
# binary_warped = pipeline(image)
# warped = perspective_warp(binary_warped)
# out_img, left_curverad, right_curverad = fit_polynomial(warped)
# plt.imshow(out_img)
# plt.show()


def vid_pipeline(img):
    global running_avg
    
    # Step 1 : Convert to Binary image
    binary_warped = pipeline(img)
    
    # Step 2 : Wrap the area on interest from video
    wrapped = perspective_warp(binary_warped)
    
    # Step 3 : Edge detection and polynomial fitting
    out_img, curves, ploty  = fit_polynomial(wrapped)
    
    # Step 4 : Curvature calculation
    curverad = get_curve(img, curves[0], curves[1],ploty)
    lane_curve = np.mean([curverad[0], curverad[1]])
    
    # Step 5 : Draw lanes and visualization on original video
    img = draw_lanes(img, curves[0], curves[1],ploty)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 0)
    fontSize=0.5
    #cv2.putText(img, 'Lane Curvature: {:.0f} m'.format(lane_curve), (570, 620), font, fontSize, fontColor, 2)
    #cv2.putText(img, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (570, 650), font, fontSize, fontColor, 2)
    return img

# Give file path of input video
myclip = VideoFileClip('input_video.mp4')

output_vid = 'output.mp4'
clip = myclip.fl_image(vid_pipeline)

# Uncomment to save the output video file
#clip.write_videofile(output_vid, audio=False)

# Uncomment to preview video while execution of program
clip.preview(fps=15, audio=False)



 

