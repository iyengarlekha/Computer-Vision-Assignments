import numpy as np
import cv2
from matplotlib import pyplot as plt
import random

def augment(xys):
    axy = np.ones( (len(xys), 1, 3) )
    axy[:, :, :-1] = xys
    return axy

def estimate(pairs, print_=False):
    A = np.zeros((1,6))
    b = np.zeros((1,1))
    for i in range(len(pairs)):
        # print pairs[i][0]
        temp = np.append(pairs[i][0], np.zeros((1,3)), axis=1)
        # print temp
        temp1 = np.append(np.zeros((1,3)), pairs[i][0], axis=1)
        # print temp1
        A = np.append(A,temp, axis=0)
        A = np.append(A,temp1, axis=0)

        b = np.append(b, pairs[i][1].reshape((3,1))[:-1], axis =0)
    
    A = A[1:, :]
    b = b[1:, :]

    if print_:
        print (A.shape)
        print (b.shape)

    try:
        q = np.linalg.solve(A, b)
    except np.linalg.linalg.LinAlgError:
        q = None
        pass

    return q


def is_inlier(coeffs, xy, threshold):
    a = coeffs.reshape((2,3)).dot(xy[0].T).reshape((1,2))
    b = np.delete(xy[1], 2, axis=1).reshape((1,2))
    # print
    return np.linalg.norm(a-b) < threshold


def ransac(data, estimate, is_inlier, sample_size = 3, max_iterations = 100, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        inliers = []
        s = random.sample(data, int(sample_size))
        m = estimate(s)

        try:
            if not m.any():
                continue
        except AttributeError:
            continue

        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
                inliers.append(data[j])

        # print(s)
        # print('estimate:', m,)
        # print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            best_inliners = inliers

    # print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic, best_inliners

def get_sift_descripter(image_name):
    img = cv2.imread(image_name)
    # gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(img, None)
    # print("Image: {} kps: {}, descriptors: {}".format(str(image_name), len(kps), descs.shape))

    img_disp=cv2.drawKeypoints(img,kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imshow('image',img_disp)
    # cv2.waitKey(0)

    return kps, descs, img


def match_descriptors(des1, des2):
    # f_match - first match from des2 for each descriptor in des1
    # s_match - second match from des2 each descriptor in des1

    first_match = [cv2.DMatch(i, -1, 0, -1) for i in range(0, len(des1))]
    second_match = [cv2.DMatch(i, -1, 0, -1) for i in range(0, len(des1))]
    # for a in first_match:
    #     print(str(a.imgIdx) + ' -- ' + str(a.queryIdx) + ' -- ' + str(a.trainIdx))
    for i in range(0, len(des1)):
        for j in range(0, len(des2)):

            dist = cv2.norm(des1[i], des2[j], normType=cv2.NORM_L2)

            if first_match[i].trainIdx == -1:
                first_match[i].trainIdx = j
                first_match[i].distance = dist

            elif second_match[i].trainIdx == -1:
                second_match[i].trainIdx = j
                second_match[i].distance = dist

            elif dist < first_match[i].distance:
                second_match[i].trainIdx = first_match[i].trainIdx
                second_match[i].distance = first_match[i].distance
                first_match[i].trainIdx = j
                first_match[i].distance = dist

            elif dist < second_match[i].distance:
                second_match[i].trainIdx = j
                second_match[i].distance = dist

    return first_match, second_match


# def match_descriptors(des1, des2):
#     matches = []
#     for q_idx, _d1 in enumerate(des1):
#         min1, min2 = cv2.DMatch(), cv2.DMatch()
#         for t_idx, _d2 in enumerate(des2):
#             dis = cv2.norm(_d1, _d2, cv2.NORM_L2)
#             if dis < min1.distance:
#                 temp = cv2.DMatch(q_idx,t_idx,0,dis)
#                 min2 = min1
#                 min1 = temp
#             elif dis < min2.distance:
#                 temp = cv2.DMatch(q_idx, t_idx, 0, dis)
#                 min2 = temp
#                 matches.append([min1,min2])
#     return matches

def main():

    image_2 = 'scene.pgm'
    image_1 = 'book.pgm'

    kp1, des1, img1 = get_sift_descripter(image_1)
    kp2, des2, img2 = get_sift_descripter(image_2)
	
    # print (len(des1[1]))
    print (len(des1))

    # BFMatcher with default params
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2, k=2)
    first, second = match_descriptors(des1, des2)
    # Apply ratio test
    good = []
    for m,n in zip(first, second):
        if m.distance < 0.9 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None)

    # print (len(good))
    cv2.imshow('image',img3)
    cv2.waitKey(0)
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

    src_pts = augment(src_pts)
    dst_pts = augment(dst_pts)
    input_to_ransac = []
    for i in range(len(src_pts)):
        input_to_ransac.append((src_pts[i], dst_pts[i]))
    
    model, best_inliner_count, inliners = ransac(input_to_ransac, estimate, lambda x, y: is_inlier(x, y, 10))

    M = model.reshape((2,3))

    # print (M)


    # Refit
    # print (len(inliners))
    model = np.zeros((6,1))
    singular = 0

    for i in range(len(inliners)-2):
        refit_list = [inliners[i], inliners[i+1], inliners[i+2]]

        temp_model = estimate(refit_list)
        if type(temp_model) == type(None):
            singular += 1
        else:
            # print (model)
            model += temp_model

    model = model / (len(inliners) - singular)
    model = model.reshape((2, 3))
    print ("H matrix: ")
    print (model)

    # transform
    im_out = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))

    cv2.imshow('image', im_out)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

