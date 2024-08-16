import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.registration import optical_flow_tvl1
from skimage.color import rgb2gray
import cv2 as cv
from visualization import plot_landmarks, blend_images, read_coordinates_from_csv, plot_landmarks_with_overlay
from PIL import Image
from evaluation import euclidean_distance_metric, k_pixel_threshold, relative_TRE, robustness
import os, argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QSpinBox, QDoubleSpinBox
from PyQt5.QtGui import QPixmap

# Existing image processing functions
def get_mean_and_std(x):
    x_mean, x_std = cv.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2)) 
    return x_mean, x_std

def get_normalized_image(input_img, template_img):
    template_img = cv.cvtColor(template_img, cv.COLOR_BGR2LAB)
    template_mean, template_std = get_mean_and_std(template_img)
    input_img = cv.cvtColor(input_img, cv.COLOR_BGR2LAB)
    img_mean, img_std = get_mean_and_std(input_img)    
    input_img = input_img.astype(np.float32)
    input_img -= img_mean
    input_img *= (template_std / img_std)
    input_img += template_mean
    input_img = np.clip(input_img, 0, 255)
    input_img = input_img.round().astype(np.uint8)
    normalized_img = cv.cvtColor(input_img, cv.COLOR_LAB2BGR)
    return normalized_img

def resize_images(source, target):
    # Resize target image to the size of source image
    target_resized = transform.resize(target, source.shape, anti_aliasing=True)
    return target_resized

def compute_optical_flow(source, target):
    # Compute the optical flow between source and target images
    v, u = optical_flow_tvl1(source, target,attachment=5, tightness=0.1, num_warp=30, num_iter=200, tol=1e-4, prefilter=True)
    return v, u

def warp_image(target, v, u):
    # Warp the target image using the estimated optical flow
    nr, nc = target.shape[0], target.shape[1]
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    warp_coords = np.array([row_coords + v, col_coords + u])
    target_warped = transform.warp(target, warp_coords, mode='edge')
    return target_warped

def register_images(source, target, downsample_factor, medianBlurKernelSize):
    downsample_factor = float(downsample_factor)    
    source_dwn = cv.medianBlur(cv.resize(source, (0, 0), fx = 1/downsample_factor, fy = 1/downsample_factor), medianBlurKernelSize)
    target_dwn = cv.medianBlur(cv.resize(target, (0, 0), fx = 1/downsample_factor, fy = 1/downsample_factor), medianBlurKernelSize)
    target_resized = resize_images(source_dwn, target_dwn)
    v, u = compute_optical_flow(source_dwn, target_resized)
    
    h_old, w_old = u.shape[:2]
    h_new, w_new = target.shape[:2]
    from scipy.interpolate import interp2d
    #print(u.shape, v.shape)
    # Interpolating and rescaling u and v
    f_u = interp2d(np.linspace(0,w_old-1,w_old),np.linspace(0,h_old-1,h_old),10*u,kind='linear')
    f_v = interp2d(np.linspace(0,w_old-1,w_old),np.linspace(0,h_old-1,h_old),10*v,kind='linear')
    
    u_rescaled = f_u(np.linspace(0,w_old-1,w_new),np.linspace(0,h_old-1,h_new))
    v_rescaled = f_v(np.linspace(0,w_old-1,w_new),np.linspace(0,h_old-1,h_new))
    
    target_warped = warp_image(target, v_rescaled, u_rescaled)
    
    return target_warped, v_rescaled, u_rescaled

def transform_landmarks(landmarks, v, u):
    # Transform the landmark coordinates using the optical flow vectors
    transformed_landmarks = landmarks.copy()
    for i, (x, y) in enumerate(landmarks):
        dx, dy = u[int(y), int(x)], v[int(y), int(x)]
        transformed_landmarks[i] = [x - dx, y - dy]
    return transformed_landmarks

# PyQt5 GUI Implementation
class OFRegistrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Optical Flow Image Registration")

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

        # OF Parameters
        self.downsample_factor_label = QLabel("Downsample Factor:", self)
        self.downsample_factor = QSpinBox(self)
        self.downsample_factor.setRange(1, 20)
        self.downsample_factor.setValue(10)

        self.median_blur_size_label = QLabel("Median Blur Size:", self)
        self.median_blur_size = QSpinBox(self)
        self.median_blur_size.setRange(1, 10)
        self.median_blur_size.setValue(5)

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

        self.layout.addWidget(self.downsample_factor_label)
        self.layout.addWidget(self.downsample_factor)

        self.layout.addWidget(self.median_blur_size_label)
        self.layout.addWidget(self.median_blur_size)

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

        downsample_factor = self.downsample_factor.value()
        median_blur_size = self.median_blur_size.value()
        
        if not source_img_path or not target_img_path:
            self.result_label.setText("Please provide both source and target images.")
            return
        
        try: 
            source = cv.imread(source_img_path)
            target = cv.imread(target_img_path)
            normalized_target = get_normalized_image(target, source)
            source = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
            normalized_target = cv.cvtColor(normalized_target, cv.COLOR_BGR2GRAY)
            
            # Register the images
            registered_image, v_rescaled, u_rescaled = register_images(source, normalized_target, downsample_factor, median_blur_size)
            plt.imshow(registered_image, cmap='gray')
            plt.show()
            
            source_image = np.array(Image.open(source_img_path))
            #source_landmarks_path = './TestData/01-HE.csv'
            source_landmarks = read_coordinates_from_csv(source_landmarks_path)
            
            #target_landmarks_path = './TestData/01-CC10.csv'
            target_landmarks = read_coordinates_from_csv(target_landmarks_path)
            
            transformed_landmarks = transform_landmarks(target_landmarks, v_rescaled, u_rescaled) 
            plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='Before Registration')
            plt.show()
            plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='After Registration')
            plt.show()

            # Evaluation:
            distances, sum_distances, avg_distances = euclidean_distance_metric(source_landmarks, transformed_landmarks)
            print(f"Avg distance between source and transformed images: {avg_distances}")
            rTRE = relative_TRE(distances, source_image)
            kpte = k_pixel_threshold(distances, 50)
            print(f"K-Pixel threshold value: {kpte}")
            robust = robustness(source_landmarks, target_landmarks, transformed_landmarks)
            print(f"Robustness: {robust}")

            self.result_label.setText(f"Registration completed successfully. Avg distance: {avg_distances}, K-Pixel threshold: {kpte}, Robustness: {robust}")
        except Exception as e:
            self.result_label.setText(f"Error during registration: {str(e)}")

def main():
    app = QApplication(sys.argv)
    ex = OFRegistrationApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
