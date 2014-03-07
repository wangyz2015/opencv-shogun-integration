import numpy as np
import cv2
import math

IMAGE_WIDHT = 28
IMAGE_HEIGHT = 28
SUDOKU_SIZE= 9
N_MIN_ACTVE_PIXELS = 20
THRESHOLD = 100

class Sudoku_CV():
    def __init__(self, path):
        self.path = path
        self.image_sudoku_original = cv2.imread(path)

        self.image_sudoku_gray = cv2.cvtColor(self.image_sudoku_original,cv2.COLOR_BGR2GRAY)
        self.blur = cv2.GaussianBlur(self.image_sudoku_gray,(9,9),0)
        self.thresh = cv2.adaptiveThreshold(self.image_sudoku_gray,255,1,1,11,15)

        self.sudoku = np.zeros(shape=(9,9))

    def getOuterPoints(self,rcCorners):
            ar = [];
            ar.append(rcCorners[0,0,:]);
            ar.append(rcCorners[1,0,:]);
            ar.append(rcCorners[2,0,:]);
            ar.append(rcCorners[3,0,:]);

            x_sum = sum(rcCorners[x, 0, 0] for x in range(len(rcCorners)) ) / len(rcCorners)
            y_sum = sum(rcCorners[x, 0, 1] for x in range(len(rcCorners)) ) / len(rcCorners)

            def algo(v):
                return (math.atan2(v[0] - x_sum, v[1] - y_sum) + 2 * math.pi) % 2*math.pi
            ar.sort(key=algo)
            return (  ar[3], ar[0], ar[1], ar[2])

    def findRectangle_Sudoku(self):
        contours0,hierarchy = cv2.findContours(self.thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        h, w = self.image_sudoku_original.shape[:2]
        cont = 0
        for contour in contours0:
            approximation = cv2.approxPolyDP(contour, 4, True)
            if cv2.contourArea(approximation)>1110:
                if(not (len(approximation)==4)):
                    continue;
                if(not cv2.isContourConvex(approximation) ):
                    continue;         
                for i in range(len(approximation)):
                    cv2.line(self.image_sudoku_original, (approximation[(i%4)][0][0], approximation[(i%4)][0][1]), (approximation[((i+1)%4)][0][0], approximation[((i+1)%4)][0][1]),  (255, 0, 0))
                cont = cont+1;
                
                points1 = np.array([
                        np.array([0.0,0.0] ,np.float32),
                        np.array([0.0,0.0] ,np.float32) + np.array([252,0], np.float32),
                        np.array([0.0,0.0] ,np.float32) + np.array([252,252], np.float32),
                        np.array([0.0,0.0] ,np.float32) + np.array([0.0,252], np.float32),
                ],np.float32)    

                outerPoints = self.getOuterPoints(approximation)
                points2 = np.array(outerPoints,np.float32)
               
                pers = cv2.getPerspectiveTransform(points2,  points1 );
                warp = cv2.warpPerspective(self.image_sudoku_original, pers, (SUDOKU_SIZE*IMAGE_HEIGHT, SUDOKU_SIZE*IMAGE_WIDHT));
        print "cont " + str(cont)
        self.warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

        cv2.imshow("sudoku_rectangle", self.image_sudoku_original);
        cv2.imshow("warp_gray", self.warp_gray);
        return self.warp_gray

    def Recognize_number(self, x, y, knn):
        im_rectangle = self.warp_gray[x*IMAGE_HEIGHT:(x+1)*IMAGE_HEIGHT][:, y*IMAGE_WIDHT:(y+1)*IMAGE_WIDHT]
        thresh_im = cv2.adaptiveThreshold(im_rectangle,255,1,1,21,9)
        n_active_pixels =0
        cv2.imshow('thresh_imt',thresh_im)
        for i in range(im_rectangle.shape[0]):
            for j in range(im_rectangle.shape[1]):
                dist_center = math.sqrt( (IMAGE_WIDHT/2 - i)**2  + (IMAGE_HEIGHT/2 - j)**2);
                if dist_center > 10:
                    thresh_im[i,j] = 0;
                if thresh_im[i, j] > THRESHOLD:
                    n_active_pixels = n_active_pixels+1
                    
        if n_active_pixels> N_MIN_ACTVE_PIXELS:
                
            im = thresh_im.reshape(1, IMAGE_WIDHT*IMAGE_HEIGHT).astype(np.float64)
            predict = knn.predict(im)
            print "Predictions", predict
            self.sudoku[x, y] = predict
        else:
            print "Predictions: ", 0
            self.sudoku[x, y] = 0;
        cv2.imshow('im',im_rectangle)
        cv2.imshow('thresh_im',thresh_im)
        cv2.waitKey(0)

    def Recognize_sudoku(self, knn):
        for i in range(SUDOKU_SIZE):
            for j in range(SUDOKU_SIZE):
                self.Recognize_number(i, j, knn);

        print_sudoku();
        return self.sudoku

    def print_sudoku(self):
        for i in range(9):
            print "-----------------------------------------------------------------------------------------------------------------------------"
            for j in range(9):
                if j%3==0 :
                    print "|\t",
                print self.sudoku[i, j], "\t",

            print "\t|\n" "-----------------------------------------------------------------------------------------------------------------------------"
        
