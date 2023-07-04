# -*- coding: utf-8 -*-
"""
Detect words on the page
return array of words' bounding boxes
# -*- coding: utf-8 -*-
Detect words on the page and return an array of words' bounding boxes.

This module provides functions for detecting words in an image and returning their bounding boxes.
    The main function 'detection' performs the word detection process using various image processing techniques,
    such as edge detection and contour analysis.

Functions:
- detection(image: any, join: bool = False) -> any: Detects the words' bounding boxes in the given image.
- sort_words(boxes: any) -> any: Sorts the word bounding boxes from left to right, top to bottom.
- union(a: any, b: any) -> any: Calculates the union of two rectangles.
- _intersect(a: any, b: any) -> any: Checks if two rectangles intersect.
- _group_rectangles(rec: any) -> any: Groups intersecting rectangles together.
- _text_detect(img: any, image: any, join: bool = False) -> any: Detects text using contours.
- text_detect_watershed(thresh: any) -> any: (Not in use) Detects text using the watershed algorithm.

Note:
- The module depends on the 'numpy' and 'cv2' (OpenCV) libraries.
- The functions may require additional configuration or modifications based on specific use cases.

Example Usage:

"""
import numpy as np
import cv2

from htr_api.autoSegment_utils import implt, ratio, resize


def detection(image: any, join: bool = False) -> any:
    """Detecting the words bounding boxes.
    Return: numpy array of bounding boxes [x, y, x+w, y+h]
    """
    # Preprocess image for word detection
    blurred = cv2.GaussianBlur(image, (5, 5), 18)
    edge_img = _edge_detect(blurred)
    _, edge_img = cv2.threshold(edge_img, 50, 255, cv2.THRESH_BINARY)
    bw_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    return _text_detect(bw_img, image, join)


def sort_words(boxes: any) -> any:
    """Sort boxes - (x, y, x+w, y+h) from left to right, top to bottom."""
    mean_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)

    boxes.view("i8,i8,i8,i8").sort(order=["f1"], axis=0)
    current_line = boxes[0][1]
    lines = []
    tmp_line = []
    for box in boxes:
        if box[1] > current_line + mean_height:
            lines.append(tmp_line)
            tmp_line = [box]
            current_line = box[1]
            continue
        tmp_line.append(box)
    lines.append(tmp_line)

    for line in lines:
        line.sort(key=lambda box: box[0])

    return lines


def _edge_detect(im: any) -> any:
    """
    Edge detection using sobel operator on each layer individually.
    Sobel operator is applied for each image layer (RGB)
    """
    return np.max(
        np.array(
            [
                _sobel_detect(im[:, :, 0]),
                _sobel_detect(im[:, :, 1]),
                _sobel_detect(im[:, :, 2]),
            ]
        ),
        axis=0,
    )


def _sobel_detect(channel: any) -> any:
    """Sobel operator."""
    sobel_x = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobel_y = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobel_x, sobel_y)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)


def union(a: any, b: any) -> any:
    """
    Calculate the union of two rectangles.

    Given two rectangles defined by their coordinates and dimensions,
    this function calculates the union of the two rectangles.

    Args:
        a (any): The first rectangle, represented as a list or tuple [x, y, width, height].
        b (any): The second rectangle, represented as a list or tuple [x, y, width, height].

    Returns:
        any: The union rectangle, represented as a list [x, y, width, height].

    Example:
        Calculate the union of two rectangles:
        ```
        a = [10, 20, 30, 40]
        b = [50, 60, 70, 80]
        result = union(a, b)
        print(result)  # Output: [10, 20, 110, 120]
        ```
    """
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return [x, y, w, h]


def _intersect(a: any, b: any) -> any:
    """
    Check if two rectangles intersect.

    Given two rectangles defined by their coordinates and dimensions,
    this function checks if the two rectangles intersect with each other.

    Args:
        a (any): The first rectangle, represented as a list or tuple [x, y, width, height].
        b (any): The second rectangle, represented as a list or tuple [x, y, width, height].

    Returns:
        any: True if the rectangles intersect, False otherwise.

    Example:
        Check if two rectangles intersect:
        ```
        a = [10, 20, 30, 40]
        b = [50, 60, 70, 80]
        result = _intersect(a, b)
        print(result)  # Output: False
        ```

    Note:
        The function considers rectangles that share an edge or a corner as intersecting.
        If one rectangle is entirely contained within the other, they are also considered intersecting.
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return False
    return True


def _group_rectangles(rec: any) -> any:
    """
    Uion intersecting rectangles.
    Args:
        rec - list of rectangles in form [x, y, w, h]
    Return:
        list of grouped ractangles
    """
    tested = [False for _ in range(len(rec))]
    final = []
    i = 0
    while i < len(rec):
        if not tested[i]:
            j = i + 1
            while j < len(rec):
                if not tested[j] and _intersect(rec[i], rec[j]):
                    rec[i] = union(rec[i], rec[j])
                    tested[j] = True
                    j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final


def _text_detect(img: any, image: any, join: bool = False) -> any:
    """Text detection using contours."""
    small = resize(img, 2000)

    # Finding contours
    # mask = np.zeros(small.shape, np.uint8)
    ### (5, 100) for line segmention  (5,30) for word segmentation
    kernel = np.ones((5, 100), np.uint16)
    img_dilation = cv2.dilate(small, kernel, iterations=1)
    # print(11111111111111)

    cnt, hierarchy = cv2.findContours(
        np.copy(small), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    index = 0
    boxes = []
    # Go through all contours in top level
    while index >= 0:
        x, y, w, h = cv2.boundingRect(cnt[index])
        cv2.drawContours(img_dilation, cnt, index, (255, 255, 255), cv2.FILLED)
        mask_roi = img_dilation[y : y + h, x : x + w]
        # Ratio of white pixels to area of bounding rectangle
        r = cv2.countNonZero(mask_roi) / (w * h)

        # Limits for text
        if (
            r > 0.1
            and 1600 > w > 10
            and 1600 > h > 10
            and h / w < 3
            and w / h < 10
            and (60 // h) * w < 1000
        ):
            boxes += [[x, y, w, h]]

        index = hierarchy[0][index][0]

    if join:
        # Need more work
        boxes = _group_rectangles(boxes)

    # image for drawing bounding boxes
    small = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
    bounding_boxes = np.array([0, 0, 0, 0])
    for x, y, w, h in boxes:
        cv2.rectangle(small, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bounding_boxes = np.vstack((bounding_boxes, np.array([x, y, x + w, y + h])))

    implt(small, t="Bounding rectangles")

    boxes = bounding_boxes.dot(ratio(image, small.shape[0])).astype(np.int64)
    return boxes[1:]


def text_detect_watershed(thresh: any) -> any:
    """NOT IN USE - Text detection using watershed algorithm.
    Based on: http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html
    """
    img = cv2.cvtColor(cv2.imread("test/n.jpg"), cv2.COLOR_BGR2RGB)
    print(img)
    img = resize(img, 3000)
    thresh = resize(thresh, 3000)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers += 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    implt(markers, t="Markers")
    image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for mark in np.unique(markers):
        # mark == 0 --> background
        if mark == 0:
            continue

        # Draw it on mask and detect biggest contour
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == mark] = 255

        cnts = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        c = max(cnts, key=cv2.contourArea)

        # Draw a bounding rectangle if it contains text
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(mask, c, 0, (255, 255, 255), cv2.FILLED)
        mask_roi = mask[y : y + h, x : x + w]
        # Ratio of white pixels to area of bounding rectangle
        r = cv2.countNonZero(mask_roi) / (w * h)

        # Limits for text
        if r > 0.2 and 2000 > w > 15 and 1500 > h > 15:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    implt(image)
