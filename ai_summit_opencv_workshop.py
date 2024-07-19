import cv2
import numpy as np
from skimage import exposure
import sys
 
def main():
    def nothing(x):
        pass

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            text = f'mouse at ({str(x)},{str(y)}), pixel color: ({str(b)},{str(g)},{str(r)})'
            print(text)

    def align_images(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
        # convert both the input image and template to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # use ORB to detect keypoints and extract (binary) local invariant features
        orb = cv2.ORB_create(maxFeatures)
        (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
        (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
        # match the features
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, descsB, None)
        # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
        matches = sorted(matches, key=lambda x:x.distance)
        # keep only the top matches
        keep = int(len(matches) * keepPercent)
        matches = matches[:keep]
        # check to see if we should visualize the matched keypoints
        if debug:
            matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
                matches, None)
            matchedVis = imutils.resize(matchedVis, width=1000)
            cv2.imshow("Matched Keypoints", matchedVis)
            cv2.waitKey(0)
        # allocate memory for the keypoints (x, y)-coordinates from the
        # top matches -- we'll use these coordinates to compute our homography matrix
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        # loop over the top matches
        for (i, m) in enumerate(matches):
            # indicate that the two keypoints in the respective images map to each other
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt
        # compute the homography matrix between the two sets of matched points
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
        # use the homography matrix to align the images
        (h, w) = template.shape[:2]
        aligned = cv2.warpPerspective(image, H, (w, h))
        # return the aligned image
        return aligned
        
    # create a window
    windowName = "testing_window"
    cv2.namedWindow(windowName) 
    # Create windows with trackbar and associate a callback function
    image_resize_trackbar_create = 0
    gaussian_blur_trackbar_create = 0
    bilateral_trackbar_create = 0
    threshold_trackbar_create = 0
    adaptive_threshold_trackbar_create = 0
    image_difference_trackbar_create = 0
    # create switch names
    image_realign_onoff_ = 'Realign'
    image_resize_onoff_ = 'Resize'
    gaussian_blur_onoff_ = 'Smoothing'
    bilateral_onoff_ = 'Bilateral'
    image_thresholding_onoff_ = 'Threshold'
    # switches
    cv2.createTrackbar(image_realign_onoff_, windowName, 0, 1, nothing)
    cv2.createTrackbar(image_resize_onoff_, windowName, 0, 1, nothing)
    cv2.createTrackbar(gaussian_blur_onoff_, windowName, 0, 1, nothing)
    cv2.createTrackbar(bilateral_onoff_, windowName, 0, 1, nothing)
    cv2.createTrackbar(image_thresholding_onoff_, windowName, 0, 1, nothing)
    
    bad_image_path = r'01_missing_hole_01_rotated.jpg' # bad image sample
    reference_image_path = r'01_golden_reference.jpg'
    
    img = cv2.imread(bad_image_path)
    ref_img = cv2.imread(reference_image_path) 
    img_ori = img.copy()

    while(True):
        image_realign_onoff = cv2.getTrackbarPos(image_realign_onoff_, windowName)
        if image_realign_onoff:
            if not image_difference_trackbar_create:
                image_aligned_ = "image_aligned"
                cv2.namedWindow(image_aligned_)
                image_difference_ = "Difference"
                cv2.createTrackbar(image_difference_, image_aligned_, 0, 1, nothing)
                cv2.setTrackbarMin(image_difference_, image_aligned_, 0) 
                image_difference_trackbar_create = 1
            aligned = align_images(img, ref_img)
            aligned_temp = cv2.resize(aligned, (0, 0), fx = 0.25, fy = 0.25)
            aligned_height , aligned_width, aligned_channels = aligned_temp.shape
            ref_img_temp = cv2.resize(ref_img, (aligned_width, aligned_height))
            img_temp = cv2.resize(img, (aligned_width, aligned_height))
            stacked = np.vstack([ref_img_temp, img_temp, aligned_temp])
            cv2.imshow(image_aligned_,stacked)     

            image_difference_onoff = cv2.getTrackbarPos(image_difference_, image_aligned_)
            if image_difference_onoff:
                align_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) 
                ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) 
                # histogram matching of input image
                aligned_normalized = exposure.match_histograms(align_gray, ref_img_gray)
                aligned_normalized = exposure.rescale_intensity(aligned_normalized, out_range=(0, 255)).astype(np.uint8)
                aligned_blurred = cv2.GaussianBlur(aligned_normalized, (9,9), 0)
                ref_img_blurred = cv2.GaussianBlur(ref_img_gray, (9,9), 0) #ref_img_normalized
                aligned_gray_temp = cv2.resize(aligned_blurred, (0, 0), fx = 0.5, fy = 0.5)
                aligned_gray_height , aligned_gray_width = aligned_gray_temp.shape
                ref_img_temp = cv2.resize(ref_img_blurred, (aligned_gray_width, aligned_gray_height))
                img_diff = cv2.absdiff(aligned_gray_temp, ref_img_temp)
                
                _,mask = cv2.threshold(img_diff,70,255,cv2.THRESH_BINARY)
                aligned_resized = cv2.resize(aligned, (aligned_gray_width, aligned_gray_height))
                mask_bgr = cv2.merge([0*mask, 0*mask, mask])
                output = cv2.addWeighted(mask_bgr, 1, np.asarray(aligned_resized), 1.0, 0)
                cv2.imshow("image_difference_mask",img_diff) 
                cv2.imshow("image_difference_output",output) 

        # Convert to grayscale. 
        if image_realign_onoff:
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) 
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray_ori = gray.copy()
        gray_copy = gray.copy()
        gray = gray_ori

        image_resize_onoff = cv2.getTrackbarPos(image_resize_onoff_, windowName)
        if image_resize_onoff:
            if not image_resize_trackbar_create:
                image_resized_ = "image_resized"
                cv2.namedWindow(image_resized_)
                scale_factor_ = "Scale"
                cv2.createTrackbar(scale_factor_, image_resized_, 1, 10, nothing)
                cv2.setTrackbarMin(scale_factor_, image_resized_, 1) 
                image_resize_trackbar_create = 1
            scale_factor = cv2.getTrackbarPos(scale_factor_, image_resized_) / 10
            gray_copy = cv2.resize(gray_copy, (0, 0), fx = scale_factor, fy = scale_factor)
            cv2.imshow("image_resized",gray_copy)
        else:
            gray_copy = cv2.resize(gray_copy, (0, 0), fx = 0.5, fy = 0.5)
        gray_copy_height , gray_copy_width = gray_copy.shape

        gaussian_blur_onoff = cv2.getTrackbarPos(gaussian_blur_onoff_, windowName)
        if gaussian_blur_onoff:
            if not gaussian_blur_trackbar_create:
                gray_blurred_ = "gray_blurred"
                cv2.namedWindow(gray_blurred_)
                blur_kernel_size_ = "KernelSize"
                cv2.createTrackbar(blur_kernel_size_, gray_blurred_, 3, 9, nothing)
                cv2.setTrackbarMin(blur_kernel_size_, gray_blurred_, 3) 
                gaussian_blur_trackbar_create = 1
            blur_kernel_size = cv2.getTrackbarPos(blur_kernel_size_, gray_blurred_)
            if blur_kernel_size % 2 == 1:
                gray_blurred = cv2.GaussianBlur(gray_copy, (blur_kernel_size,blur_kernel_size), 0)
                gray_copy = gray_blurred.copy()
                gray_blurred = cv2.resize(gray_blurred, (gray_copy_width, gray_copy_height))
                cv2.imshow(gray_blurred_,gray_blurred)
            else:
                print(f"[WARN] invalid kernel size {blur_kernel_size}, please select odd number!")

        bilateral_onoff = cv2.getTrackbarPos(bilateral_onoff_, windowName)
        if bilateral_onoff:
            if not bilateral_trackbar_create:
                gray_sharpened_ = "gray_sharpened"
                cv2.namedWindow(gray_sharpened_)
                bilateral_filter_ = 'Bilateral'
                bilateral_sigma_ = 'Sigma'
                cv2.createTrackbar(bilateral_filter_, gray_sharpened_, 3, 9, nothing)
                cv2.createTrackbar(bilateral_sigma_, gray_sharpened_, 10, 150, nothing)
                cv2.setTrackbarMin(bilateral_filter_, gray_sharpened_, 3) 
                cv2.setTrackbarMin(bilateral_sigma_, gray_sharpened_, 10) 
                bilateral_trackbar_create = 1
            bilateral_filter = cv2.getTrackbarPos(bilateral_filter_, gray_sharpened_)
            bilateral_sigma = cv2.getTrackbarPos(bilateral_sigma_, gray_sharpened_)
            gray_sharpened = cv2.bilateralFilter(gray_copy, bilateral_filter, bilateral_sigma, bilateral_sigma) 
            gray_copy = gray_sharpened.copy()
            gray_sharpened = cv2.resize(gray_sharpened, (gray_copy_width, gray_copy_height))
            cv2.imshow(gray_sharpened_,gray_sharpened)

        image_thresholding_onoff = cv2.getTrackbarPos(image_thresholding_onoff_, windowName)
        if image_thresholding_onoff:
            if not threshold_trackbar_create:
                gray_thresholded_ = "gray_thresholded"
                cv2.namedWindow(gray_thresholded_)
                threshold_min_ = 'Min Thresh'
                cv2.createTrackbar(threshold_min_, gray_thresholded_, 0, 255, nothing)
                cv2.setTrackbarMin(threshold_min_, gray_thresholded_, 0) 
                adaptive_thresholding_ = 'Adaptive'
                cv2.createTrackbar(adaptive_thresholding_, gray_thresholded_, 0, 1, nothing)
                cv2.setTrackbarMin(adaptive_thresholding_, gray_thresholded_, 0) 
                threshold_trackbar_create = 1
            threshold_min = cv2.getTrackbarPos(threshold_min_, gray_thresholded_)
            adaptive_thresholding = cv2.getTrackbarPos(adaptive_thresholding_, gray_thresholded_)
            if adaptive_thresholding:
                if not adaptive_threshold_trackbar_create:
                    block_size_ = 'Block Size'
                    cv2.createTrackbar(block_size_, gray_thresholded_, 3, 29, nothing)
                    cv2.setTrackbarMin(block_size_, gray_thresholded_, 3) 
                    adaptive_threshold_trackbar_create = 1
                block_size = cv2.getTrackbarPos(block_size_, gray_thresholded_)
                if block_size % 2 == 1:
                    thresh = cv2.adaptiveThreshold(gray_copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)
                else:
                    print(f"[WARN] invalid kernel size {block_size}, please select odd number!")
            else:
                ret, thresh = cv2.threshold(gray_copy, threshold_min, 255, cv2.THRESH_BINARY) #  threshold_max2
            gray_copy = thresh.copy()
            thresh = cv2.resize(thresh, (gray_copy_width, gray_copy_height))
            cv2.imshow(gray_thresholded_,thresh)

        img_ori = cv2.resize(img_ori, (gray_copy_width, gray_copy_height))
        gray_ori = cv2.resize(gray_ori, (gray_copy_width, gray_copy_height))
        
        # Display the image 
        cv2.imshow(windowName, img_ori) 
        cv2.setMouseCallback(windowName, mouse)

        # Create a button for pressing and changing the window 
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    # Close the window 
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)
