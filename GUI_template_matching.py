import sys
import os
import numpy as np
import cv2 as cv
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout, QSpinBox, QDoubleSpinBox
from PyQt5.QtGui import QPixmap
from PIL import Image
from matplotlib import pyplot as plt

# Assuming these are the imported functions from the provided code
from evaluation import euclidean_distance_metric, k_pixel_threshold, relative_TRE, robustness
from visualization import plot_landmarks_with_overlay, read_coordinates_from_csv

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

def SIFT_registration(source_image, target_image, normalized_target_image, source_landmarks, target_landmarks, downsample_factor, median_blur_size, lowe_ratio, ransac_threshold):
    downsample_factor = float(downsample_factor)
    median_blur_size = int(median_blur_size)
    lowe_ratio = float(lowe_ratio)
    ransac_threshold = float(ransac_threshold)
    source_image = cv.cvtColor(source_image, cv.COLOR_RGB2GRAY)
    source_image = cv.medianBlur(cv.resize(source_image, (0, 0), fx = 1/downsample_factor, fy = 1/downsample_factor), median_blur_size)
    normalized_target_image = cv.cvtColor(normalized_target_image, cv.COLOR_RGB2GRAY)
    normalized_target_image = cv.medianBlur(cv.resize(normalized_target_image, (0, 0), fx = 1/downsample_factor, fy = 1/downsample_factor), median_blur_size)

    S = np.array([[downsample_factor, 0, 0], [0, downsample_factor, 0], [0, 0, 1]])
    Sinv = np.array([[1/downsample_factor, 0, 0], [0, 1/downsample_factor, 0], [0, 0, 1]])

    MIN_MATCH_COUNT = 10

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(normalized_target_image, None)
    kp2, des2 = sift.detectAndCompute(source_image, None)
    # cv.setRNGSeed(2391)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 30)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.trainIdx != m.queryIdx and m.distance < lowe_ratio * n.distance:
            good.append(m)
    
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_threshold)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    matches_image = cv.drawMatches(normalized_target_image, kp1, source_image, kp2, good, None, **draw_params)

    points1 = np.float32([kp1[m.queryIdx].pt for m in good])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good])

    homography_matrix, mask = cv.findHomography(points1, points2, cv.RANSAC, ransac_threshold)

    h, w = source_image.shape[:2]
    homography_matrix_rescaled = np.matmul(S, homography_matrix)
    homography_matrix_rescaled = np.matmul(homography_matrix_rescaled, Sinv)
    print(homography_matrix_rescaled)
    registered_image = cv.warpPerspective(target_image, homography_matrix_rescaled, (int(1/downsample_factor)*w, int(1/downsample_factor)*h))

    transformed_landmarks = cv.perspectiveTransform(np.array(target_landmarks).reshape(-1, 1, 2), homography_matrix_rescaled)
    transformed_landmarks = transformed_landmarks.reshape(-1, 2)
    transformed_landmarks_list = [tuple(np.round(coord, 1)) for coord in transformed_landmarks]
    
    return registered_image, transformed_landmarks_list, matches_image

# PyQt5 GUI Implementation
class SIFTRegistrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SIFT Image Registration")

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

        # SIFT Parameters
        self.downsample_factor_label = QLabel("Downsample Factor:", self)
        self.downsample_factor = QSpinBox(self)
        self.downsample_factor.setRange(1, 20)
        self.downsample_factor.setValue(10)

        self.median_blur_size_label = QLabel("Median Blur Size:", self)
        self.median_blur_size = QSpinBox(self)
        self.median_blur_size.setRange(1, 10)
        self.median_blur_size.setValue(5)

        self.lowe_ratio_label = QLabel("Lowe Ratio:", self)
        self.lowe_ratio = QDoubleSpinBox(self)
        self.lowe_ratio.setRange(0.0, 1.0)
        self.lowe_ratio.setValue(0.80)

        self.ransac_threshold_label = QLabel("RANSAC Threshold:", self)
        self.ransac_threshold = QSpinBox(self)
        self.ransac_threshold.setRange(1, 10)
        self.ransac_threshold.setValue(5)

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

        self.layout.addWidget(self.lowe_ratio_label)
        self.layout.addWidget(self.lowe_ratio)

        self.layout.addWidget(self.ransac_threshold_label)
        self.layout.addWidget(self.ransac_threshold)

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
        lowe_ratio = self.lowe_ratio.value()
        ransac_threshold = self.ransac_threshold.value()

        if not source_img_path or not target_img_path:
            self.result_label.setText("Please provide both source and target images.")
            return

        try:
            source_image = np.array(Image.open(source_img_path))
            target_image = np.array(Image.open(target_img_path))
            normalized_target_image = get_normalized_image(target_image, source_image)

            source_landmarks = read_coordinates_from_csv(source_landmarks_path) if source_landmarks_path else []
            target_landmarks = read_coordinates_from_csv(target_landmarks_path) if target_landmarks_path else []

            registered_image, transformed_landmarks_list, matches_image = SIFT_registration(
                source_image, target_image, normalized_target_image,
                source_landmarks, target_landmarks,
                downsample_factor, median_blur_size, lowe_ratio, ransac_threshold
            )
            fig, axes = plt.subplots(2, 1)

            axes[0].imshow(matches_image)
            axes[0].set_title('Matches Image')

            axes[1].imshow(registered_image)
            axes[1].set_title('Registered Image')

            plt.show()

            plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='Before Registration')
            plt.show()

            plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='After Registration')
            plt.show()

            self.result_label.setText("Registration completed successfully.")
        except Exception as e:
            self.result_label.setText(f"Error: {e}")

def main():
    app = QApplication(sys.argv)
    ex = SIFTRegistrationApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
