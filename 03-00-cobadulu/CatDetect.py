# import the necessary packages
import cv2

# Define the path to the input image and Haar cascade file
image_path = "C:\SMA_PRAXIS\03-00-cobadulu\kucing\kucing.jpg"
cascade_path = "haarcascade_frontalcatface.xml"

# load the input image and convert it to grayscale
image = cv2.imread(image_path)
if image is None:
    raise Exception("Image not found or unable to load!")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the cat detector Haar cascade, then detect cat faces in the input image
detector = cv2.CascadeClassifier(cascade_path)
if detector.empty():
    raise Exception("Cascade file not found or unable to load!")

rects = detector.detectMultiScale(gray, scaleFactor=1.3,
                                  minNeighbors=10, minSize=(75, 75))

# loop over the cat faces and draw a rectangle surrounding each
for (i, (x, y, w, h)) in enumerate(rects):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# show the detected cat faces
cv2.imshow("Cat Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
