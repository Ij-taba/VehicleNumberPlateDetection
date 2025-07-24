import cv2
import selectivesearch

img = cv2.imread('model\DSC_0872_JPG.rf.2499cabc40b18c5341d9b7247ec9272d.jpg')

img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)

canditates = set()
for r in regions:
    x, y, w, h = r['rect']
    rect = (x, y, w, h)
    if r['size'] < 200:
        continue
    canditates.add(rect)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =======================================================
def wrap_region(image, region, target_size=(224, 224)):
        # Extract the bounding box coordinates (x, y, width, height)
        x, y, w, h = region
        # Crop the region from the image
        cropped_image = image[y:y+h, x:x+w]
        # Resize the cropped region to the target size
        resized_image = cv2.resize(cropped_image, target_size)
        return resized_image
     # Example usage:
image = cv2.imread('WhatsApp Image 2024-12-24 at 10.04.28_aaeff151.jpg')
region = (50, 50, 100, 100)  # Example region (x, y, w, h)
wrapped_image = wrap_region(image, region)
cv2.imshow('Wrapped Region', wrapped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()