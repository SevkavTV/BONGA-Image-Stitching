import time

import cv2 as cv
import numpy as np

ratio = 0.4  # threshold of distance between descriptors
# do not recommend to change


def find_best_height(img, temp):
    w = max(img.shape)
    h = min(img.shape)

    w_divs = []
    while temp < w:
        if w % temp == 0:
            w_divs.append(temp)
        temp += 1

    min_dif = w_divs[-1]
    best_common_div = w_divs[-1]
    best_mltplr = 0
    for x in w_divs:
        mltplr = int(h / x)
        if min_dif > h - x * mltplr:
            min_dif = h - x * mltplr
            best_common_div = x
            best_mltplr = mltplr
    print(f"best com div: {best_common_div}")
    print(f"{best_mltplr}*{best_common_div}={best_mltplr*best_common_div}", end="\n\n")
    return (best_mltplr * best_common_div, best_common_div)


def check_img_cmpetability(img1, img2):
    return img1.shape == img2.shape


def get_buffered_img(path: str, bw=False):
    if bw:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Something is wrong with image path: {path}")
            exit()
        print(f"Opened path: {path}\nImage is buffered and black-white", end="\n\n\n")
        return img
    else:
        img = cv.imread(path)
        if img is None:
            print(f"Something is wrong with image path: {path}")
            exit()
        print(f"Opened path: {path}\nImage is buffered and colourful", end="\n\n\n")
        return img


def output_buffered_img(img, path):
    cv.imwrite(path, img)
    print(f"Image saved to {path}", end="\n\n\n")


def crop_img_bottom(img, height=2934):
    # 2935 is min to get rid of date and time
    return img[:height, :]


def get_keypoints(img1, img2):
    akaze = cv.AKAZE_create()
    print("AKAZE created")
    # find the keypoints and descriptors with AKAZE
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    print("Key poins found")
    return ((kp1, des1), (kp2, des2))


def get_matched_points(des1, des2, n_of_points):
    # create BFMatcher object
    print("Beginning brute forse")
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(des1, des2, 2)

    # Apply ratio test
    good = []
    for m, n in nn_matches:
        if m.distance < ratio * n.distance:
            good.append([m])
    good = sorted(good, key=lambda x: x[0].distance)
    good = good[:n_of_points]
    print(f"Good matches found.\nGood matches: {len(good)}")
    return good


def get_linked_points_img(img1, kp1, img2, kp2, good):
    img_3 = None
    img_3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, img_3)
    return img_3


def convert_kp_to_coords(kp1, kp2, matches):
    coords_1 = []
    coords_2 = []
    for kp in matches:
        # Get the matching keypoints for each of the images
        img1_idx = kp[0].queryIdx
        img2_idx = kp[0].trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        coords_1.append((x1, y1))
        coords_2.append((x2, y2))
    return (coords_1, coords_2)


def get_transformation_matrix(coords):
    source = np.round(np.float32(coords[0]), 0)
    destination = np.round(np.float32(coords[1]), 0)
    h, status = cv.findHomography(source, destination)
    return h


def transform_img(img, matrix):
    return cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))


def get_laplacian(img, kernel_size):
    # Apply Gaussian Blur
    blur = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Apply Laplacian operator in some higher datatype
    laplacian = cv.Laplacian(blur, cv.CV_32F)

    # But this tends to localize the edge towards the brighter side.
    result = laplacian / laplacian.max()

    return result


def show_result(img):
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.resizeWindow("image", 600, 600)
    cv.imshow("image", img)
    cv.waitKey(0)


def get_grid(img, cnt, width):
    color = [57, 255, 20]
    h = int(img.shape[0] / cnt)
    w = int(img.shape[1] / cnt)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for i in range(0, img.shape[0] - 1, h):
        if i + width < img.shape[0]:
            img[i : i + width, :] = color
    for j in range(0, img.shape[1] - 1, w):
        if j + width < img.shape[1]:
            img[:, j : j + width] = color
    return img


def increase_sharpness(src_bw, dst_bw, src_col, dst_col, min_cnt, kernel_size):
    shape = src_bw.shape
    cnt = find_best_height(src_bw, min_cnt)[1]
    result = np.zeros((shape[0], shape[1], 3)).astype(np.uint8)

    if shape[0] % cnt != 0 and shape[1] % cnt != 0:
        print("img cant be splitted.")
        exit()
    w = int(shape[1] / cnt)
    h = int(shape[0] / cnt)
    h_shift = 0
    v_shift = 0

    print(f"Shape: {shape}")
    print(f"CNT: {cnt}")
    print(f"w: {w}\nh: {h}")

    start = time.time()
    dummy = True
    for progress in range(int((shape[0] * shape[1]) / (w * h))):
        laplace_shard_1 = get_laplacian(src_bw, kernel_size)[
            h_shift : h + h_shift, v_shift : w + v_shift
        ]
        laplace_shard_2 = get_laplacian(dst_bw, kernel_size)[
            h_shift : h + h_shift, v_shift : w + v_shift
        ]
        cleares_shard = dst_col[h_shift : h + h_shift, v_shift : w + v_shift]

        # 1st approach: amount of positive values
        if len(laplace_shard_1[laplace_shard_1 > 0]) > len(
            laplace_shard_2[laplace_shard_2 > 0]
        ):
            cleares_shard = src_col[h_shift : h + h_shift, v_shift : w + v_shift]

        # 2nd approach: sum of positive values
        """
        if sum(laplace_shard_1[laplace_shard_1>0]) > sum(laplace_shard_2[laplace_shard_2>0]):
            cleares_shard = src_col[h_shift:h + h_shift, v_shift:w + v_shift]
        """

        result[h_shift : h + h_shift, v_shift : w + v_shift] = cleares_shard
        # print(f'|{(v_shift)} {(h_shift)}|', end='  ')

        v_shift += w
        if v_shift == shape[1]:
            v_shift = 0
            h_shift += h
        """
        show_result(laplace_shard_1)
        show_result(laplace_shard_2)
        show_result(cleares_shard)
        show_result(result)
        """
        if dummy:
            dummy = False
            print(
                f"\nEstimated processing time: {round((time.time() - start)*cnt*cnt/60,1)} min"
            )
        print(f"Progress: {round(progress/(cnt**2-1)*100, 2)}%")
    return result


def main():
    # the only numbers you need to change
    h = 2880  # height of the img
    min_cnt = 500  # minimal number of spliting the image
    kernel_size = 3  # gaussian blur power.   must be odd number!
    max_points = 20  # max key points

    # uploading images
    src_bw = crop_img_bottom(get_buffered_img("images/117.jpg", bw=True), h)
    src_col = crop_img_bottom(get_buffered_img("images/117.jpg"), h)

    dst_bw = crop_img_bottom(get_buffered_img("images/116.jpg", bw=True), h)
    dst_col = crop_img_bottom(get_buffered_img("images/116.jpg"), h)

    if not check_img_cmpetability(src_bw, dst_bw):
        print("images are different size")
        exit()

    # getting key points and descriptors
    data = get_keypoints(src_bw, dst_bw)
    kp1 = data[0][0]
    kp2 = data[1][0]
    desc1 = data[0][1]
    desc2 = data[1][1]

    # getting coodrinates of best matched points
    matched_points = get_matched_points(desc1, desc2, max_points)
    linked_points_img = get_linked_points_img(src_bw, kp1, dst_bw, kp2, matched_points)
    coords = convert_kp_to_coords(kp1, kp2, matched_points)

    # computing transformation matrix and transforming source image
    homography = get_transformation_matrix(coords)
    transformed = transform_img(src_col, homography)
    show_result(linked_points_img)
    show_result(transformed)

    # computing the resunt matrix
    result = increase_sharpness(
        transform_img(src_bw, homography),
        dst_bw,
        transform_img(src_col, homography),
        dst_col,
        min_cnt,
        kernel_size,
    )
    show_result(result)
    output_buffered_img(
        result,
        "output/final_result_LoG_ks"
        + str(kernel_size)
        + "_mincnt"
        + str(min_cnt)
        + ".jpg",
    )


if __name__ == "__main__":
    main()
