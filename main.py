import cv2
import numpy as np

cap = cv2.VideoCapture('video.avi')
imgTarget = cv2.imread('TargetImage.png')
building = cv2.imread('building2.jpg')

# make building and image target the same size
hT, wT, cT = imgTarget.shape
building = cv2.resize(building, (wT, hT))

# find key points in target image
sift = cv2.SIFT_create(nfeatures=1000)
kp1, des1 = sift.detectAndCompute(imgTarget, None)
# show key points
# imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)
# cv2.imwrite('imgTargetKeypoint.jpg',imgTarget)

while True:
    sucess , imgVideo = cap.read()
    imgAug = imgVideo.copy()
    # find key points in imgVideo image
    kp2, des2 = sift.detectAndCompute(imgVideo, None)
    # show key points
    # imgVideo = cv2.drawKeypoints(imgVideo, kp2, None)
    # cv2.imwrite('imgVideotKeypoint.jpg',imgVideo)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.8 *n.distance:
            good.append(m)

    print('number of good matches is:', len(good))

    # show feature matches
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgVideo, kp2, good, None, flags=2)
    cv2.imwrite('imgFeatures.jpg', imgFeatures)
    cv2.imshow('imgFeatures', imgFeatures)


    if len(good) > 5:
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # find homography
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)


        # boundary
        pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgVideo, [np.int32(dst)], True, (255, 255, 0), 3)
        # cv2.imshow('img2', img2)
        # cv2.imwrite('img2.jpg', img2)


        # image warping, we warp the building image in the shape of boundaries
        imgWarp = cv2.warpPerspective(building, matrix, (imgVideo.shape[1], imgVideo.shape[0]))
        # cv2.imshow('imgWarp', imgWarp)
        # cv2.imwrite('imgWarp.jpg', imgWarp)


        # create mask as same size as imgVideo
        maskNew = np.zeros((imgVideo.shape[0], imgVideo.shape[1]), np.uint8)
        # cv2.imshow('maskNew', maskNew)
        # cv2.imwrite('maskNew.jpg', maskNew)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255))
        maskInv = cv2.bitwise_not(maskNew)
        # cv2.imshow('maskInv', maskInv)
        # cv2.imwrite('maskInv.jpg', maskInv)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask= maskInv)
        # cv2.imshow('imgAug', imgAug)
        # cv2.imwrite('imgAug.png', imgAug)

        # adding
        imgAug = cv2.bitwise_or(imgWarp, imgAug)


    # show result! :D
    cv2.imshow('imgAug', imgAug)

    # cv2.imwrite('imgAug.jpg', imgAug)
    # cv2.imshow('ImageTarget', imgTarget)
    # cv2.imshow('building', building)
    # cv2.imshow('imgVideo', imgVideo)

    cv2.waitKey(1)
