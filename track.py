
#Import reqquired libraries
import numpy as np
import cv2
import math
import time
from imutils.video import VideoStream
import datetime
import argparse
ROI_points=[]
image = None
MS_input = []
list1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
list2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
global vs
# import the necessary packages

global frame
# import the necessary packages
from imutils.video import VideoStream
import datetime
import argparse
import imutils

def ROI_capture(event, u, v, flags, param):

	#capture Region of Interest(ROI) by mouse click
	global MS_input, ROI_points, image

	#circle ROI points with red color and draw it on the image
	if MS_input and event == cv2.EVENT_LBUTTONDOWN and len(ROI_points) < 4:
		ROI_points.append((u, v))
		cv2.circle(image, (u, v), 4, (0, 0, 255), 2)

		cv2.imshow("frame", image)


def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 0.

def main():

	#declare global varibales
	global MS_input, ROI_points, image

	count=0
	(dX,dY) = (0,0)
	#initialize counter to save frames
	counter = 0

	#call ROI_capture function to capture ROI points
	cv2.namedWindow("frame")
	cv2.setMouseCallback("frame", ROI_capture)

	#termination criteria for camshift algorithm.
	termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

	#bounding box for roi
	BBox_roi = None

	#bounding box for refrence point
	BBox_ref = None

	#store previous angle for angle difference calculation
	angle_buf = 0;
	x2 = 0
	y2 = 0
	# keep looping over the frames
	while 1:
		frame = vs.read()

		image = frame

		'''
		if not cap:
			break
		'''
		#check bounding box(ROI) is calculated or not
		if BBox_roi is not None:

			#convert frame from RGB to HSV space
			hsv_space = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			Projection_back = cv2.calcBackProject([hsv_space], [0], BBox_hist, [0, 180], 1)

			#apply camshift algorithm for each ROI point
			(points, BBox_roi) = cv2.CamShift(Projection_back, BBox_roi, termination)
			pts = np.int32(cv2.boxPoints(points))

			#Draw bounding box around ROI
			cv2.polylines(image, [pts], True, (0, 255, 255), 2)

			#calculate centroid of ROI
			p0 = pts[0]
			p1 = pts[1]
			p2 = pts[2]
			p3 = pts[3]
			cx1 = (p0[0]+p3[0])/2
			cy1 = (p0[1]+p3[1])/2
			cx2 = (p1[0]+p2[0])/2
			cy2 = (p1[1]+p2[1])/2
			x1 = (cx1 + cx2)/2
			y1 = (cy1 + cy2)/2

			list1[count] = x1
			list2[count] = y1

			if count<20:
				count = count+1
			if count==20:
				dX = list1[19] - list1[0]

				if abs(dX)>20:
					print(dX)
					print('east') if sign(dX) == -1 else print('West')

				count=0
				dY = list2[19]-list2[0]
				if abs(dY)>20:
					#print(dy)
					print('North') if sign(dY) == -1 else print('South')


			#check bounding box(around reference point) is calculated or not
			
			if BBox_ref is not None:

				#convert frame from RGB to HSV space
				hsv_space = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
				Projection_back = cv2.calcBackProject([hsv_space], [0], BBox_hist_ref, [0, 180], 1)

				#apply camshift algorithm for each boundary points of reference point
				(points, BBox_ref) = cv2.CamShift(Projection_back, BBox_ref, termination)
				pts = np.int32(cv2.cv.BoxPoints(points))

				#Draw bounding box around bounded refernce point
				cv2.polylines(image, [pts], True, (0, 255, 255), 2)

				#calculate reference point
				p0 = pts[0]
				p1 = pts[1]
				p2 = pts[2]
				p3 = pts[3]
				cx1 = (p0[0]+p3[0])/2
				cy1 = (p0[1]+p3[1])/2
				cx2 = (p1[0]+p2[0])/2
				cy2 = (p1[1]+p2[1])/2
				x2 = (cx1 + cx2)/2
				y2 = (cy1 + cy2)/2
				cv2.circle(image, (x2, y2), 4, (0, 255, 255), 2)

				#plot reference point on the image
				cv2.line(image,(x1,y1),(x2,y2),(255,0,0),5)

				#calculate slope and current angle
				if x2 == x1:
					deg = 90
				else:
					slope = float(y1-y2)/float(x2-x1)
					deg = math.degrees(math.atan(slope))

				if y2<=y1 and x2>=x1:
					degree = deg
				elif y2<=y1 and x2<=x1:
					degree = 180+deg
				elif x1>x2 and y1<y2:
					degree = 180+deg
				else:
					degree = 360+deg

				#calculate angle difference
				angle_diff= degree - angle_buf
				angle_buf = degree

				#put the text on the image
				cv2.putText(image, str(angle_diff), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 5)

			#display frame
			cv2.imshow("frame", image)
			file = "/home/ashutosh/image/frame"+ str(counter) + ".png"
			cv2.imwrite(file, image)
			counter = counter+1


			#press buttun for selecting reference point ROI
			button = cv2.waitKey(1) & 0xFF

			#push "s" select reference point ROI
			
			if len(ROI_points) < 4 and button == ord("s"):

				#change status of mouse input and duplicate the frame
				MS_input = True
				orig = image.copy()

				#select ROI points until it reaches four
				while len(ROI_points) < 4:
					cv2.imshow("frame", image)
					cv2.waitKey(1)

				#create array and perform vertical addition
				ROI_points = np.array(ROI_points)
				add = ROI_points.sum(axis = 1)

				#select minimum and maximum point in current ROI
				lower = ROI_points[np.argmin(add)]
				upper = ROI_points[np.argmax(add)]

				#visit all the pixels under current ROI and convert RGB to HSV
				region = orig[lower[1]:upper[1], lower[0]:upper[0]]
				region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

				#calculate the current ROI histogram for future tracking
				BBox_hist_ref = cv2.calcHist([region], [0], None, [16], [0, 180])
				BBox_hist_ref = cv2.normalize(BBox_hist_ref, BBox_hist_ref, 0, 255, cv2.NORM_MINMAX)
				BBox_ref = (lower[0], lower[1], upper[0], upper[1])

			#press "q" to quit
			elif button == ord("q"):
				break
			

		#display frame
		cv2.imshow("frame", image)

		#press buttun for selecting actual ROI
		button = cv2.waitKey(1) & 0xFF

		#push "i" select reference point ROI
		if len(ROI_points) < 4 and button == ord("i"):

			#change status of mouse input and duplicate the frame
			MS_input = True
			orig = image.copy()

			#select ROI points until it reaches four
			while len(ROI_points) < 4:
				cv2.imshow("frame", image)
				cv2.waitKey(1)

			#create array and perform vertical addition
			ROI_points = np.array(ROI_points)
			add = ROI_points.sum(axis = 1)

			#select minimum and maximum point in Factual ROI
			lower = ROI_points[np.argmin(add)]
			upper = ROI_points[np.argmax(add)]

			#visit all the pixels under actual ROI and convert RGB to HSV
			region = orig[lower[1]:upper[1], lower[0]:upper[0]]
			region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

			#calculate the current ROI histogram for future tracking
			BBox_hist = cv2.calcHist([region], [0], None, [16], [0, 180])
			BBox_hist = cv2.normalize(BBox_hist, BBox_hist, 0, 255, cv2.NORM_MINMAX)
			BBox_roi = (lower[0], lower[1], upper[0], upper[1])
			ROI_points = []

		#press "q" to quit
		elif button == ord("q"):
			break

	#wipe out buffer
	frame.release()
	cv2.destroyAllWindows()
'''
if __name__ == "__main__":
	main()
'''

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	main()
	# draw the timestamp on the frame
	timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

#initialize Region of Interest points, image buffer to NIL and mouse input buffer to False
ROI_points = []
image = None
MS_input = False

def ROI_capture(event, u, v, flags, param):

	#capture Region of Interest(ROI) by mouse click
	global MS_input, ROI_points, image

	#circle ROI points with red color and draw it on the image
	if MS_input and event == cv2.EVENT_LBUTTONDOWN and len(ROI_points) < 4:
		ROI_points.append((u, v))
		cv2.circle(image, (u, v), 4, (0, 0, 255), 2)
		cv2.imshow("frame", image)
