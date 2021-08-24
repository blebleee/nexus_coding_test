import os
import click
import logging
import re

import cv2 
import numpy as np 
import pytesseract
from pdf2image import convert_from_path

def preprocess(image):
    '''
    @input: numpy ndarray
    @desc: preprocessing the input image for better OCR
    @return: numpy ndarray 
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return thresh

def postprocess(text):
    p = re.compile('\A.*?@bnf\.fr',re.MULTILINE|re.DOTALL)
    text = p.sub('',text)

    p = re.compile(r'(-\n|\n\w|[^\w]{2,})',re.UNICODE)
    text = p.sub(' ',text).lower()

    return text

def OCR(input_path, output_path, verbose=True):
    #set up logger
    logging.basicConfig(filename=output_path, filemode='w', format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ALLOW_EXT = ['png', 'jpg', 'jpeg', 'pdf']
    ext = input_path.split('.')[-1].lower()
    if ext not in ALLOW_EXT:
        raise OSError('This type of file is not supported')
    
    if ext == 'pdf':
        image = convert_from_path(input_path)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(input_path)

    preprocessed_image = preprocess(image)
    text = pytesseract.image_to_string(Image.fromarray(preprocessed_img))
    if not verbose:
        logger.info(text)
    else:
        text = postprocess(text)
        logger.info(text)

@click.command()
@click.option('-input', type=str)
@click.option('-output', type=str)
@click.option('-verbose', is_flag=True)
def Reader(input, output, verbose):
    OCR(input, output, verbose)

if __name__ == '__main__':
    Reader()
