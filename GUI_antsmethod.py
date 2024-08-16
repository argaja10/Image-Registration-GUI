import math, random
import ants
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
from skimage import io, transform
from visualization import plot_landmarks, blend_images, read_coordinates_from_csv, plot_landmarks_with_overlay
import pandas as pd
import random
import os
from evaluation import euclidean_distance_metric, k_pixel_threshold, relative_TRE, robustness
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QSpinBox, QDoubleSpinBox
from PyQt5.QtGui import QPixmap
import sys
import cv2 as cv

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
# os.environ['ANTS_RANDOM_SEED']='42'
# Method 3: Using intensity-based antspy library to register images using SyNRA transformation method

# Existing image processing functions
def get_mean_and_std(x):
    x_mean, x_std = cv.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2)) 
    return x_mean, x_std

def get_normalized_image(input_img, template_img):
    template_img = cv.cvtColor(template_img, cv.COLOR_RGB2LAB)
    template_mean, template_std = get_mean_and_std(template_img)
    input_img = cv.cvtColor(input_img, cv.COLOR_RGB2LAB)
    img_mean, img_std = get_mean_and_std(input_img)    
    input_img = input_img.astype(np.float32)
    input_img -= img_mean
    input_img *= (template_std / img_std)
    input_img += template_mean
    input_img = np.clip(input_img, 0, 255)
    input_img = input_img.round().astype(np.uint8)
    normalized_img = cv.cvtColor(input_img, cv.COLOR_LAB2RGB)
    return normalized_img

def ants_registration(fixed_image, moving_image, target_landmarks, type_of_transform):
    fixed_image_gray = cv.cvtColor(fixed_image, cv.COLOR_RGB2GRAY)
    moving_image_gray = cv.cvtColor(moving_image, cv.COLOR_RGB2GRAY)

    # Convert grayscale images to ANTs images
    fixed_image = ants.from_numpy(fixed_image_gray)
    moving_image = ants.from_numpy(moving_image_gray)
    
    # set seed
    np.random.seed(42)
    random.seed(42)
    
    # Perform affine registration on grayscale images
    registration_result = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyNRA')

    # Get the warped moving image
    #warped = registration_result['warpedmovout']
    warped = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=registration_result['fwdtransforms'])

    warped = warped.numpy()
    
    target_landmarks = target_landmarks[:,::-1]
    target_landmarks= pd.DataFrame(target_landmarks, columns=['x', 'y'])

    #warped_moving = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=registration_result['fwdtransforms'])
    transformed_points = ants.apply_transforms_to_points(dim=2, points=target_landmarks, transformlist=registration_result['invtransforms'])

    transformed_points = transformed_points.to_numpy()[:,::-1]
    transformed_points = [tuple(r) for r in transformed_points]
    
    return warped, transformed_points
    
# PyQt5 GUI Implementation
class AntsRegistrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SynRA Image Registration")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # I/O Inputs
        self.source_img_label = QLabel("Source Image:", self)
        self.source_img_path = QLineEdit(self)
        self.source_img_button = QPushButton("Browse...", self)
        self.source_img_button.clicked.connect(self.select_source_image)

        self.target_img_label = QLabel("Target Image:", self)
        self.target_img_path = QLineEdit(self)
        self.target_img_button = QPushButton("Browse...", self)
        self.target_img_button.clicked.connect(self.select_target_image)

        self.source_landmarks_label = QLabel("Source Landmarks (optional):", self)
        self.source_landmarks_path = QLineEdit(self)
        self.source_landmarks_button = QPushButton("Browse...", self)
        self.source_landmarks_button.clicked.connect(self.select_source_landmarks)

        self.target_landmarks_label = QLabel("Target Landmarks (optional):", self)
        self.target_landmarks_path = QLineEdit(self)
        self.target_landmarks_button = QPushButton("Browse...", self)
        self.target_landmarks_button.clicked.connect(self.select_target_landmarks)

        # Layout setup for I/O inputs and parameters
        self.layout.addWidget(self.source_img_label)
        self.layout.addWidget(self.source_img_path)
        self.layout.addWidget(self.source_img_button)

        self.layout.addWidget(self.target_img_label)
        self.layout.addWidget(self.target_img_path)
        self.layout.addWidget(self.target_img_button)

        self.layout.addWidget(self.source_landmarks_label)
        self.layout.addWidget(self.source_landmarks_path)
        self.layout.addWidget(self.source_landmarks_button)

        self.layout.addWidget(self.target_landmarks_label)
        self.layout.addWidget(self.target_landmarks_path)
        self.layout.addWidget(self.target_landmarks_button)

        # Process button
        self.process_button = QPushButton("Register Images", self)
        self.process_button.clicked.connect(self.process_registration)
        self.layout.addWidget(self.process_button)

        # Result label
        self.result_label = QLabel("", self)
        self.layout.addWidget(self.result_label)
    
    def select_source_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Source Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.source_img_path.setText(file_name)

    def select_target_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Target Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.target_img_path.setText(file_name)

    def select_source_landmarks(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Source Landmarks (CSV)", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.source_landmarks_path.setText(file_name)

    def select_target_landmarks(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Target Landmarks (CSV)", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.target_landmarks_path.setText(file_name)

    def process_registration(self):
        source_img_path = self.source_img_path.text()
        target_img_path = self.target_img_path.text()
        source_landmarks_path = self.source_landmarks_path.text()
        target_landmarks_path = self.target_landmarks_path.text()

        if not source_img_path or not target_img_path:
            self.result_label.setText("Please provide both source and target images.")
            return
        
        try: 
            fixed_image = np.array(Image.open(source_img_path))
            target_image = np.array(Image.open(target_img_path))
            target = get_normalized_image(target_image, fixed_image)
        
            source_landmarks = read_coordinates_from_csv(source_landmarks_path)
            target_landmarks = read_coordinates_from_csv(target_landmarks_path)
            target_landmarks = np.array(target_landmarks)
    
            warped_moving, transformed_points = ants_registration(fixed_image, target, target_landmarks, type_of_transform='SyNRA')
            
            plt.imshow(warped_moving, cmap='gray')
            plt.show()
            
            plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='Before Registration')
            plt.show()
            
            plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='After Registration')
            plt.show()
            
            # Evaluation:
            distances, sum_distances, avg_distances = euclidean_distance_metric(source_landmarks, transformed_points)
            print(f"Avg distance between source and transformed images: {avg_distances}")
            rTRE = relative_TRE(distances, fixed_image)
            kpte = k_pixel_threshold(distances, 50)
            print(f"K-Pixel threshold value: {kpte}")
            robust = robustness(source_landmarks, target_landmarks, transformed_points)
            print(f"Robustness: {robust}")

            self.result_label.setText(f"Registration completed successfully. \nAvg distance between source and transformed images: {avg_distances} \nK-Pixel threshold value: {kpte} \nRobustness: {robust}")
        except Exception as e:
            self.result_label.setText(f"Error: {e}")

def main():
    app = QApplication(sys.argv)
    ex = AntsRegistrationApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
