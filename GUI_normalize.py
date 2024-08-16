import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QLineEdit, QHBoxLayout

# The existing functions for stain normalization
def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2)) 
    return x_mean, x_std

def get_normalized_image(input_img_path, template_img_path, output_dir):
    template_img = cv2.imread(template_img_path)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
    template_mean, template_std = get_mean_and_std(template_img)
    
    input_img = cv2.imread(input_img_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    img_mean, img_std = get_mean_and_std(input_img)
    
    input_img = input_img.astype(np.float32)
    input_img -= img_mean
    input_img *= (template_std / img_std)
    input_img += template_mean
    input_img = np.clip(input_img, 0, 255)
    input_img = input_img.round().astype(np.uint8)
    
    normalized_img = cv2.cvtColor(input_img, cv2.COLOR_LAB2BGR)
    
    img_name = os.path.basename(input_img_path)
    normalized_img_path = os.path.join(output_dir, f'normalized-{img_name}')
    cv2.imwrite(normalized_img_path, normalized_img)
    
    return normalized_img, normalized_img_path

# PyQt5 GUI Implementation
class StainNormalizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Stain Normalization")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Input image selection
        self.input_img_label = QLabel("Input Image:", self)
        self.input_img_path = QLineEdit(self)
        self.input_img_button = QPushButton("Browse...", self)
        self.input_img_button.clicked.connect(self.select_input_image)

        self.input_img_layout = QHBoxLayout()
        self.input_img_layout.addWidget(self.input_img_label)
        self.input_img_layout.addWidget(self.input_img_path)
        self.input_img_layout.addWidget(self.input_img_button)
        self.layout.addLayout(self.input_img_layout)

        # Template image selection
        self.template_img_label = QLabel("Template Image:", self)
        self.template_img_path = QLineEdit(self)
        self.template_img_button = QPushButton("Browse...", self)
        self.template_img_button.clicked.connect(self.select_template_image)

        self.template_img_layout = QHBoxLayout()
        self.template_img_layout.addWidget(self.template_img_label)
        self.template_img_layout.addWidget(self.template_img_path)
        self.template_img_layout.addWidget(self.template_img_button)
        self.layout.addLayout(self.template_img_layout)

        # Output directory selection
        self.output_dir_label = QLabel("Output Directory:", self)
        self.output_dir_path = QLineEdit(self)
        self.output_dir_button = QPushButton("Browse...", self)
        self.output_dir_button.clicked.connect(self.select_output_directory)

        self.output_dir_layout = QHBoxLayout()
        self.output_dir_layout.addWidget(self.output_dir_label)
        self.output_dir_layout.addWidget(self.output_dir_path)
        self.output_dir_layout.addWidget(self.output_dir_button)
        self.layout.addLayout(self.output_dir_layout)

        # Process button
        self.process_button = QPushButton("Normalize Stain", self)
        self.process_button.clicked.connect(self.process_stain_normalization)
        self.layout.addWidget(self.process_button)

        # Result label
        self.result_label = QLabel("", self)
        self.layout.addWidget(self.result_label)

    def select_input_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.input_img_path.setText(file_name)

    def select_template_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Template Image", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.template_img_path.setText(file_name)

    def select_output_directory(self):
        options = QFileDialog.Options()
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory", "", options=options)
        if dir_name:
            self.output_dir_path.setText(dir_name)

    def process_stain_normalization(self):
        input_img_path = self.input_img_path.text()
        template_img_path = self.template_img_path.text()
        output_dir = self.output_dir_path.text()

        if not input_img_path or not template_img_path or not output_dir:
            self.result_label.setText("Please fill in all fields.")
            return

        try:
            _, normalized_img_path = get_normalized_image(input_img_path, template_img_path, output_dir)
            self.result_label.setText(f"Image normalized successfully! Saved to: {normalized_img_path}")
        except Exception as e:
            self.result_label.setText(f"Error: {e}")

def main():
    app = QApplication(sys.argv)
    ex = StainNormalizationApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
