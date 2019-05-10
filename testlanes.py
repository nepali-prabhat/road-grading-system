import cv2
import numpy as np
import matplotlib.pyplot as plt
video = cv2.VideoCapture('test2.mp4')
def getEdges(image):
	#convert to grey scale.
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	#blur image for smooting
	blur = cv2.GaussianBlur(gray, (5,5),0)
	#edge detection
	canny = cv2.Canny(blur,50,150)
	return canny
#different images have different points, so we need to calculate it automaitcally in future.
#for now, we speculate
def toGetCoordinates(image):
	plt.imshow(image)
	plt.show()

def getLaneRegion(image):
	height = image.shape[0]
	polygons = np.array([[(0,500),(0,height),(1050,height),(600,250)]])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask,polygons,255)
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image

def getLaneLines(image, lines):
	lineImage = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2 = line.reshape(4)
			cv2.line(lineImage,(x1,y1),(x2,y2),(255,100,100),10)
	return lineImage

i = 0
while True:
	ret, frame = video.read()
	if not ret:
		video = cv2.VideoCapture('test2.mp4')
		continue
	frameMatrix = np.copy(frame)
	edges = getEdges(frameMatrix)
	if i == 0:
		toGetCoordinates(edges)
	i+=1
	laneRegion = getLaneRegion(edges)
	#HoighLinesP(image, rho, theta, threshold(per box least intersection needed, placeholder, minLineLength, maxLineGap))
	laneRegionHoughLines = cv2.HoughLinesP(laneRegion, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	laneLinesTraces = getLaneLines(laneRegion,laneRegionHoughLines)
	#needs same dimention images to add
	laneRegionTraced = cv2.addWeighted(laneRegion, 0.8, laneLinesTraces, 1, 1)
	cv2.imshow('lane region edges traced',laneRegionTraced)
	
	key = cv2.waitKey(25)
	if(key==27):
		break
video.release()
cv2.destroyAllWindows()