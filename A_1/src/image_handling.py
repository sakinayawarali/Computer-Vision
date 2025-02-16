from tkinter import Image
import numpy as np
import cv2 as cv  # noqa: F401


def uint8_to_float(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with uint8 values in the range [0, 255] and
    return a copy of the image with data type float32 and values in the range [0, 1]
    """
    return (image.astype(np.float32)/255.0)

def float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image with float32 values in the range [0, 1] and
    return a copy of the image with uint8 values in the range [0, 255]. Values outside the range
    should be clipped (i.e. a float of 1.1 should be converted to a uint8 of 255, and a float of
    -0.1 should be converted to a uint8 of 0).
    """
    # Find the maximum and minimum values in the array
    min_val,max_val = image.min(), image.max()
    # Handling edge cases (All values are the same, then normalisation would cause div by 0 error)
    if min_val == max_val:
        return np.zeros_like(image, dtype=np.unit8)
    # Normalised values = original_value - min / max - min
    normalized = (image - min_val) / (max_val - min_val)
    # Scale to 0-255, and convert
    return (normalized * 255).astype(np.uint8)

def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image cropped to the
    given rectangle. Any part of the rectangle that falls outside the image should be considered
    black (i.e. 0 intensity in all channels).
    """
    img_h, img_w, channels = image.shape

    rectangle = np.zeros((h, w, channels), dtype = image.dtype)

    # Valid Region
    x1, x2 = max(0, x), min(x+w, img_w)
    y1, y2 = max(0,y), min(y + h, img_h)

    # Placing the valid region
    crop_x1, crop_x2 = max(0, -x), min(w, img_w - x)
    crop_y1, crop_y2 = max(0, -y), min(h, img_h - y)

    rectangle[crop_y1:crop_y2, crop_x1:crop_x2] = image[y1:y2, x1:x2]

    return rectangle

def scale_by_half_using_numpy(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image taking every
    other pixel in each row and column. For example, if the original image has shape (H, W, 3),
    the returned image should have shape (H // 2, W // 2, 3).
    """
    return image[::2, ::2]

def scale_by_half_using_cv(image: np.ndarray) -> np.ndarray:
    """Using cv.resize, take an image and return a copy of the image scaled down by a factor of 2,
    mimicking the behavior of scale_by_half_using_numpy_slicing. Pay attention to the
    'interpolation' argument of cv.resize (see the OpenCV documentation for details).
    """
    return cv.resize(image, (0,0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)

def horizontal_mirror_image(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image flipped
    horizontally (i.e. a mirror image). The behavior should match cv.flip(image, 1).
    """
    return image[:, ::-1]   


def rotate_counterclockwise_90(image: np.ndarray) -> np.ndarray:
    """Without using any cv functions, take an image and return a copy of the image rotated
    counterclockwise by 90 degrees. The behavior should match
    cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE).
    """
    # Transpose (swap rows and cols)
    transposed = np.transpose(image, (1, 0, 2))

    # Reverse the order of the rows
    return transposed[::-1, :, :]

def swap_b_r(image: np.ndarray) -> np.ndarray:
    """Given an OpenCV image in BGR channel format, return a copy of the image with the blue and red
    channels swapped. You may use any numpy or opencv functions you like.
    """
    return image[:, :, [2, 1, 0]]

def blues(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the blue
    channel
    """
    # Blank image
    img = np.zeros_like(image) # fills zeros, but same shape as image

    # Fill the blue channel
    img[:, :, 0] = image[:, :, 0]
    return img


def greens(image: np.ndarray) -> np.ndarray:
    # Blank image
    img = np.zeros_like(image) # fills zeros, but same shape as image

    # Fill the green channel
    img[:, :, 1] = image[:, :, 1]
    return img


def reds(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a copy of the image with only the red
    channel
    """
    # Blank image
    img = np.zeros_like(image) # fills zeros, but same shape as image

    # Fill the red channel
    img[:, :, 2] = image[:, :, 2]
    return img


def scale_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    """Take an OpenCV image in BGR channel format. Convert to HSV and multiply the saturation
    channel by the given scale factor, then convert back to BGR.
    """
    # First convert BGR to HSV

    # Step 1: Normalise the values
    red = image[:, :, 2]/255.0
    green = image[:, :, 1]/255.0
    blue = image[:, :, 0]/255.0

    # Step 2: Compute V (It is the max of the 3 channels b/c brightest part of the colour)
    v = np.maximum.reduce([red, green, blue])

    # Step 3: Compute chroma 
    min = np.minimum.reduce([red, green, blue])
    c = v - min

    # Step 4: Compute S (Measure of how much colour there is compared to brightness)
    # If v> 0, else s = 0
    s = np.where(v > 0, c/v, 0)
    s = np.clip(s * scale, 0, 1)

    # Step 5: Compute H (Angle of the colour on wheel)
    h = np.zeros_like(v, dtype = np.float32)

    # red is dominant
    mask_r = (v == red) & (c > 0)
    # green is dominant 
    mask_g = (v == green) & (c > 0)
    # blue is dominant 
    mask_b = (v == blue) & (c > 0)

    h[mask_r] = (60 * ((green - blue) / c) % 6)[mask_r]
    h[mask_g] = (60 * ((blue - red) / c + 2) % 6)[mask_g]
    h[mask_b] = (60 * ((red - green) / c + 4) % 6)[mask_b]

    h[c == 0] = 0  # edge case = hue to 0 where chroma is zero

    # HSV -> BGR
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c

    r_new = np.zeros_like(h)
    g_new = np.zeros_like(h)
    b_new = np.zeros_like(h)

    mask_0 = (0 <= h) & (h < 60)
    mask_1 = (60 <= h) & (h < 120)
    mask_2 = (120 <= h) & (h < 180)
    mask_3 = (180 <= h) & (h < 240)
    mask_4 = (240 <= h) & (h < 300)
    mask_5 = (300 <= h) & (h < 360)

    r_new[mask_0] = c[mask_0]
    g_new[mask_0] = x[mask_0]
    b_new[mask_0] = 0

    r_new[mask_1] = x[mask_1]
    g_new[mask_1] = c[mask_1]
    b_new[mask_1] = 0

    r_new[mask_2] = 0
    g_new[mask_2] = c[mask_2]
    b_new[mask_2] = x[mask_2]

    r_new[mask_3] = 0
    g_new[mask_3] = x[mask_3]
    b_new[mask_3] = c[mask_3]

    r_new[mask_4] = x[mask_4]
    g_new[mask_4] = 0
    b_new[mask_4] = c[mask_4]

    r_new[mask_5] = c[mask_5]
    g_new[mask_5] = 0
    b_new[mask_5] = x[mask_5]

    # Add m to match brightness
    r_new = (r_new + m) * 255
    g_new = (g_new + m) * 255
    b_new = (b_new + m) * 255

    # Stack channels
    bgr_img = np.stack([b_new, g_new, r_new], axis=-1).astype(np.uint8)

    return bgr_img

def grayscale(image: np.ndarray) -> np.ndarray:
    """Using numpy, reproduce the OpenCV function cv.cvtColor(image, cv.COLOR_BGR2GRAY) to convert
    the given image to grayscale. The returned image should still be in BGR channel format.
    """
    # formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
    r, g, b = image[:, :, 2], image[:, :, 1], image[:, :, 0]

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)

    grayscaled = np.stack([gray, gray, gray], axis=-1)

    return grayscaled


def tile_bgr(image: np.ndarray) -> np.ndarray:
    """Take an OpenCV image in BGR channel format and return a 2x2 tiled copy of the image, with the
    original image in the top-left, the blue channel in the top-right, the green channel in the
    bottom-left, and the red channel in the bottom-right. If the original image has shape (H, W, 3),
    the returned image has shape (2 * H, 2 * W, 3).
    """
    h, w, channels = image.shape

    b, g , r = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    blue_image = np.stack([b, np.zeros_like(b), np.zeros_like(b)], axis=-1)
    green_image = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=-1)
    red_image = np.stack([np.zeros_like(r), np.zeros_like(r), r], axis=-1)

    # Could also use concatenate but axis would be 0 for hor, ver case axis would be 1
    top_row = np.hstack([image, blue_image])
    bottom_row = np.hstack([green_image, red_image])

    return np.vstack([top_row, bottom_row])

def main():
    # Write your testing code here and provide standard input/output calls to run the functions
    # If use gives 1, run function 1, if 2, run function 2, etc.
    # If user gives 0, exit the program, etc

    # TODO: Implement the main function
    image = cv.imread('bouquet.png')
    while True:
        print("\nImage Processing Functions:")
        print("1. Convert uint8 to float")
        print("2. Convert float to uint8")
        print("3. Crop image")
        print("4. Scale by half (numpy)")
        print("5. Scale by half (cv)")
        print("6. Mirror image")
        print("7. Rotate 90° counterclockwise")
        print("8. Swap blue and red channels")
        print("9. Show blue channel")
        print("10. Show green channel")
        print("11. Show red channel")
        print("12. Scale saturation")
        print("13. Convert to grayscale")
        print("14. Create tiled BGR image")
        print("0. Exit")

        choice = input("\nEnter your choice (0-14): ")
        
        try:
            choice = int(choice)
            if choice == 0:
                break

            # Show original image for comparison
            cv.imshow('Original', image)
            
            if choice == 1:
                result = uint8_to_float(image)
                print(f"Converted to float32. Range: [{result.min():.2f}, {result.max():.2f}]")
                # Convert back to uint8 for display
                display_result = (result * 255).astype(np.uint8)
            
            elif choice == 2:
                # First convert to float for testing
                float_image = image.astype(np.float32) / 255.0
                result = float_to_uint8(float_image)
                display_result = result
            
            elif choice == 3:
                x, y = 100, 100
                w, h = 200, 200
                result = crop(image, x, y, w, h)
                display_result = result
            
            elif choice == 4:
                result = scale_by_half_using_numpy(image)
                display_result = result
            
            elif choice == 5:
                result = scale_by_half_using_cv(image)
                display_result = result
            
            elif choice == 6:
                result = horizontal_mirror_image(image)
                display_result = result
            
            elif choice == 7:
                result = rotate_counterclockwise_90(image)
                display_result = result
            
            elif choice == 8:
                result = swap_b_r(image)
                display_result = result
            
            elif choice == 9:
                result = blues(image)
                display_result = result
            
            elif choice == 10:
                result = greens(image)
                display_result = result
            
            elif choice == 11:
                result = reds(image)
                display_result = result
            
            elif choice == 12:
                scale = float(input("Enter saturation scale factor (e.g., 1.5): "))
                result = scale_saturation(image, scale)
                display_result = result
            
            elif choice == 13:
                result = grayscale(image)
                display_result = result
            
            elif choice == 14:
                result = tile_bgr(image)
                display_result = result
            
            # Show the result
            cv.imshow('Result', display_result)
            cv.waitKey(0)
            cv.destroyAllWindows()

        except ValueError:
            print("Please enter a valid number")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

main()