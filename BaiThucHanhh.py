from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2
import numpy as np
import os

# Đường dẫn đến thư mục chứa ảnh
image_folder_path = 'D:\data_nhakhoa'
X_dental, y_dental = process_dental_images(image_folder_path)

# Chia tập dữ liệu ảnh thành tập huấn luyện và tập kiểm tra
X_train_dental, X_test_dental, y_train_dental, y_test_dental = train_test_split(X_dental, y_dental, test_size=0.3, random_state=42)
# Tạo mô hình CART với tiêu chí Gini
cart_model_iris = DecisionTreeClassifier(criterion="gini", random_state=42)
cart_model_iris.fit(X_train_iris, y_train_iris)

# Dự đoán trên tập kiểm tra IRIS
y_pred_iris = cart_model_iris.predict(X_test_iris)

# Đánh giá độ chính xác trên dữ liệu IRIS
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
print("Accuracy on IRIS dataset:", accuracy_iris)
print("Classification report for IRIS dataset:\n", classification_report(y_test_iris, y_pred_iris))
# Tạo mô hình CART với tiêu chí Gini
cart_model_dental = DecisionTreeClassifier(criterion="gini", random_state=42)
cart_model_dental.fit(X_train_dental, y_train_dental)

# Dự đoán trên tập kiểm tra ảnh nha khoa
y_pred_dental = cart_model_dental.predict(X_test_dental)

# Đánh giá độ chính xác trên dữ liệu ảnh nha khoa
accuracy_dental = accuracy_score(y_test_dental, y_pred_dental)
print("Accuracy on Dental Images dataset:", accuracy_dental)
print("Classification report for Dental Images dataset:\n", classification_report(y_test_dental, y_pred_dental))


