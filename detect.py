import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
args = vars(ap.parse_args())
image = cv2.imread(args['image'])

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Do Gaussian Blurring and Canny Edge Detection
img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

img_canny = cv2.Canny(img_blur, 10, 250)

# Now let us find the contours
(cnts, _) = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 0, 255), 2)
cv2.imshow("Original", image)
print("In the image {} contours are found!".format(len(cnts)))
cv2.imshow("Coins detected", coins)
cv2.imshow("Canny Edge Detection", np.hstack((img_gray, img_canny)))

# Calculate the number of rows and columns for the subplots
num_contours = len(cnts)
num_cols = 3
num_rows = (num_contours // num_cols) + (1 if num_contours % num_cols != 0 else 0)

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 15))

# Flatten the axes array for easy indexing
axes = axes.flatten()

# Now let us separate the coins and show them
for (i, c) in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    (Cx, Cy), rad = cv2.minEnclosingCircle(c)
    sub_image = image[y:y + w, x:x + h]
    # Since we want to display using matplotlib we convert the BGR to RGB
    sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (int(Cx), int(Cy)), int(rad), 255, -1)
    mask = mask[y:y + w, x:x + h]

    masked_subimage = cv2.bitwise_and(sub_image, sub_image, mask=mask)
    # Create index for the subplots
    axes[i].imshow(masked_subimage)
    axes[i].set_title("Coin Number {}".format(i + 1))
    axes[i].axis('off')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()
cv2.waitKey(0)
