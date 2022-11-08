
import os
from threading import Thread
from queue import Queue

import numpy as np
import cv2

ALLOWED_EXTENSIONS = set((".jpg", ".jpeg", ".bmp", ".png", ".tiff"))
SIZE = (120,40)


class bulk_image_reader(Thread):
    def __init__(self, parent_dir="", files=None, img_q=None):
        Thread.__init__(self)
        self.img_q = img_q
        self.parent_dir = parent_dir
        self.files = files

    def run(self):

        files_list = [os.path.join(self.parent_dir, f)
                      for f in self.files if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]

        #print("From Thread for : {}. # of files to read {}".format(self.parent_dir, len(files_list)))
        
        if len(files_list) > 0:
            images = self._read_all_images(files_list)
            self.img_q.put((images, files_list))
        
        print("From Thread for : {}. Done !!! ".format(self.parent_dir))

    def _read_all_images(self, files_list):
        '''  Read all images  '''
        images = []
        for f in files_list:
            img = read_and_preprocess_image(f)
            images.append(img)
            
        return images

 
def read_and_preprocess_image(f):
    '''
        Read and preprocess the image
    '''
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img,(3,3),0)
    _,img = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img,(3,3),0)
    return img

def read_all_images(parent_dir_name):
    '''
        Read all images in the training directories
    '''
    images_q = Queue()
    th_pool = []

    if not os.path.exists(parent_dir_name):
        raise Exception(
            "The directory '{}' does not exist !".format(parent_dir_name))

    print("Trying to read images from : ", parent_dir_name)

    for curr_path, sub_dirs, fnames in os.walk(parent_dir_name):
        if len(fnames) > 0:
            reader = bulk_image_reader(parent_dir=curr_path, files=fnames, img_q=images_q)
            th_pool.append(reader)
            reader.start()

    for th in th_pool:
        th.join()

    images_list = []
    names_list = []

    while(not images_q.empty()):
        img_list, lbl_list = images_q.get()
        images_list.extend(img_list)
        names_list.extend(lbl_list)

    return np.asarray(images_list), names_list
