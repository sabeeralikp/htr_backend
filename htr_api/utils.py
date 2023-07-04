"""
Utility functions for PDF processing and text extraction.

This module contains several functions for working with PDF files, converting PDFs to images,
    applying thresholding and segmentation techniques, extracting text from images, and converting PDFs to DOCX format.

Functions:
- pdf_to_images: Converts a PDF file into a list of images.
- get_threshold_image: Applies thresholding to an input image.
- save_image_from_pdf: Saves images extracted from a PDF file to the specified directory.
- save_image_from_in_memory_image: Saves an image from in-memory data to the specified directory.
- threshold_value: Applies thresholding and dilation operations to images and returns coordinate data.
- extract_text_from_images: Extracts text from images using coordinate data.
- convert_pdf_to_docx: Converts a PDF file to DOCX format.
- extract_text: Extracts text from a PDF file using thresholding and segmentation techniques.
- auto_segmentation: Performs auto-segmentation on a PDF file and returns coordinate data.
"""
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
def pdf_to_images(pdf_file: any) -> any:
    """
    Converts a PDF file into a list of images.

    Args:
        pdf_file: The PDF file to be converted.

    Returns:
        A list of images extracted from the PDF file.
    """
    images = convert_from_bytes(pdf_file)
    return images


def get_threshold_image(image: any, threshold: any) -> any:
    """
    Applies thresholding to an input image.

    Args:
        image: The input image.
        threshold: The threshold value for the conversion.

    Returns:
        The thresholded image.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresh


def save_image_from_pdf(images: any, filename: any) -> None:
    """
    Save images extracted from a PDF file to the specified directory.

    Args:
        images: The list of images extracted from the PDF file.
        filename: The filename of the PDF file.

    Raises:
        OSError: If the folder creation fails.

    Returns:
        None
    """
    try:
        mkdir(f"media/pdf2img/{filename.strip('.pdf')}")
    except OSError as e:
        print("Couldn't create folder", e)
    for i in range(len(images)):
        cv2.imwrite(
            f"media/pdf2img/{filename.strip('.pdf')}/{i}.png", np.array(images[i])
        )


def save_image_from_in_memory_image(image: any, filename: any, file_type: any) -> None:
    """
    Saves an image from in-memory data to the specified directory.

    Args:
        image: The in-memory image data.
        filename: The filename for the saved image.
        file_type: The file type or extension of the image.

    Raises:
        OSError: If the folder creation fails.

    Returns:
        None
    """
    image = Image.open(BytesIO(bytearray(image)))
    try:
        mkdir(f"media/pdf2img/{filename.strip(f'.{file_type}')}")
    except OSError as e:
        print("Couldn't create folder", e)
    cv2.imwrite(
        f"media/pdf2img/{filename.strip(f'.{file_type}')}/{0}.png", np.array(image)
    )


def threshold_value(
    threshold: any,
    dilate_x: any,
    dilate_y: any,
    filename: any,
    number_of_pages: any,
    upload_htr: any,
    file_type: any,
) -> any:
    """
    Applies thresholding to images and retrieves the coordinates of bounding rectangles.

    Args:
        threshold: The threshold value for the conversion.
        dilate_x: The dilation factor for the x-axis.
        dilate_y: The dilation factor for the y-axis.
        filename: The filename of the image.
        number_of_pages: The number of pages in the image.
        upload_htr: The upload HTR value.
        file_type: The file type or extension of the image.

    Returns:
        A list of dictionaries representing the coordinates of bounding rectangles.

    """
    cordinates = []
    for i in range(number_of_pages):
        image = cv2.imread(f"media/pdf2img/{filename.strip(f'.{file_type}')}/{i}.png")
        image = np.array(image)
        img_h, img_w, _ = image.shape
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


def extract_text_from_images(cordinates: any, filename: any, file_type: any) -> any:
    """
    Extracts text from images based on the provided coordinates.

    Args:
        cordinates: A list of dictionaries representing the coordinates of bounding rectangles.
        filename: The filename of the image.
        file_type: The file type or extension of the image.

    Returns:
        A list of extracted text from the specified image regions.

    """
    extracted_text = []

    for cordinate in cordinates:
        image = cv2.imread(
            f"media/pdf2img/{filename.strip(f'.{file_type}')}/{cordinate['p']}.png"
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


def convert_pdf_to_docx(filename: any) -> None:
    """
    Converts a PDF file to DOCX format.

    Args:
        filename: The filename of the PDF file to be converted.

    Returns:
        None

    """
    print(str(BASE_DIR) + filename)
    # convert pdf to docx
    print(isfile(str(BASE_DIR) + filename))
    cv = Converter(str(BASE_DIR) + filename)
    filename = filename.replace(".pdf", ".docx").replace("PDF", "Doc")
    # all pages by default
    cv.convert(str(BASE_DIR) + filename)
    cv.close()


def extract_text(
    in_memory_file: any,
    threshold_value: int = 80,
    dilate_x_value: int = 1,
    dilate_y_value: int = 20,
) -> any:
    """
    Extracts text from a PDF file.

    Args:
        in_memory_file: The in-memory representation of the PDF file.
        threshold_value: The threshold value for image binarization (default: 80).
        dilate_x_value: The dilation factor for x-axis (default: 1).
        dilate_y_value: The dilation factor for y-axis (default: 20).

    Returns:
        The extracted text as a string.

    """
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


def auto_segmentation(
    filename: any, number_of_pages: any, upload_htr: any, file_type: any
) -> any:
    """
    Performs automatic segmentation of text regions in a PDF document.

    Args:
        filename: The name of the PDF file.
        number_of_pages: The total number of pages in the PDF.
        upload_htr: The HTR (Handwritten Text Recognition) upload option.
        file_type: The type of the PDF file.

    Returns:
        A list of dictionaries containing the coordinates and other information
        of the segmented text regions. Each dictionary has the following keys:
            - 'x': The x-coordinate of the top-left corner of the region.
            - 'y': The y-coordinate of the top-left corner of the region.
            - 'w': The width of the region.
            - 'h': The height of the region.
            - 'p': The page number of the region.
            - 'img_h': The height of the original image.
            - 'img_w': The width of the original image.
            - 'upload_htr': The HTR upload option for the region.

    """
    cordinates = []
    for i in range(number_of_pages):
        image = cv2.cvtColor(
            cv2.imread(f"media/pdf2img/{filename.strip(f'.{file_type}')}/{i}.png"),
            cv2.COLOR_BGR2RGB,
        )
        img_h, img_w, _ = image.shape
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
