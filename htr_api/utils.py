from io import BytesIO
from pdf2image import convert_from_bytes
import cv2
from numpy import ones, uint8

from core.settings import BASE_DIR
from htr_api import page, words
from . import test_resnet
import numpy as np
from os import mkdir
from pdf2docx import Converter
from os.path import isfile
import PIL.Image as Image


# Convert PDF to Images
def pdf_to_images(pdf_file):
    images = convert_from_bytes(pdf_file)
    return images


def get_threshold_image(image, threshold):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresh


def saveImageFromPDF(images, filename):
    try:
        mkdir(f"media/pdf2img/{filename.strip('.pdf')}")
    except:
        print("Couldn't create folder")
    for i in range(len(images)):
        cv2.imwrite(
            f"media/pdf2img/{filename.strip('.pdf')}/{i}.png", np.array(images[i])
        )


def saveImageFromInMemoryImage(image, filename, fileType):
    image = Image.open(BytesIO(bytearray(image)))
    try:
        mkdir(f"media/pdf2img/{filename.strip(f'.{fileType}')}")
    except:
        print("Couldn't create folder")
    cv2.imwrite(
        f"media/pdf2img/{filename.strip(f'.{fileType}')}/{0}.png", np.array(image)
    )


def thresholdValue(
    threshold, dilate_x, dilate_y, filename, number_of_pages, upload_htr, file_type
):
    cordinates = []
    for i in range(number_of_pages):
        image = cv2.imread(f"media/pdf2img/{filename.strip(f'.{file_type}')}/{i}.png")
        image = np.array(image)
        img_h, img_w, img_c = image.shape
        threshold_image = get_threshold_image(image, threshold)
        kernel = ones((dilate_x, dilate_y), uint8)
        dilated = cv2.dilate(threshold_image, kernel, iterations=1)
        (contours, heirarchy) = cv2.findContours(
            dilated.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        for line in contours:
            _x, _y, _w, _h = cv2.boundingRect(line)
            cordinates.append(
                {
                    "x": _x,
                    "y": _y,
                    "w": _w,
                    "h": _h,
                    "p": i,
                    "img_h": img_h,
                    "img_w": img_w,
                    "upload_htr": upload_htr,
                }
            )
    return cordinates


def extract_text_from_images(cordinates, filename, fileType):
    extracted_text = []

    for cordinate in cordinates:
        image = cv2.imread(
            f"media/pdf2img/{filename.strip(f'.{fileType}')}/{cordinate['p']}.png"
        )
        image = np.array(image)
        extracted_text.append(
            test_resnet.pred_with_image(
                image[
                    cordinate["y"] : cordinate["y"] + cordinate["h"],
                    cordinate["x"] : cordinate["x"] + cordinate["w"],
                ]
            )
        )
    return extracted_text


def convertPDFtoDOCX(filename):
    print(str(BASE_DIR) + filename)
    # convert pdf to docx
    print(isfile(str(BASE_DIR) + filename))
    cv = Converter(str(BASE_DIR) + filename)
    filename = filename.replace(".pdf", ".docx").replace("PDF", "Doc")
    cv.convert(str(BASE_DIR) + filename)  # all pages by default
    cv.close()


def extract_text(
    in_memory_file, threshold_value=80, dilate_x_value=1, dilate_y_value=20
):
    threshold_value = 80 if threshold_value == None else threshold_value
    dilate_x_value = 1 if dilate_x_value == None else dilate_x_value
    dilate_y_value = 20 if dilate_y_value == None else dilate_y_value
    extracted_text = ""
    images = pdf_to_images(in_memory_file.read())
    for image in images:
        image = np.array(image)
        threshold_image = get_threshold_image(image, threshold_value)
        kernel = ones((dilate_x_value, dilate_y_value), uint8)
        dilated = cv2.dilate(threshold_image, kernel, iterations=1)
        (contours, heirarchy) = cv2.findContours(
            dilated.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        for line in contours:
            x, y, w, h = cv2.boundingRect(line)
            if not (w < 50 or h < 15):
                extracted_text = (
                    test_resnet.pred_with_image(image[y : y + h, x : x + w])
                    + " "
                    + extracted_text
                )
    return extracted_text


# Auto Segmentation


def autoSegmentation(filename, number_of_pages, upload_htr, fileType):
    cordinates = []
    for i in range(number_of_pages):
        image = cv2.cvtColor(
            cv2.imread(f"media/pdf2img/{filename.strip(f'.{fileType}')}/{i}.png"),
            cv2.COLOR_BGR2RGB,
        )
        img_h, img_w, img_c = image.shape
        crop = page.detection(image)
        boxes = words.detection(crop)
        lines = words.sort_words(boxes)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cordinates.append(
                    {
                        "x": x1,
                        "y": y1,
                        "w": abs(x2 - x1),
                        "h": abs(y2 - y1),
                        "p": i,
                        "img_h": img_h,
                        "img_w": img_w,
                        "upload_htr": upload_htr,
                    }
                )
    return cordinates
