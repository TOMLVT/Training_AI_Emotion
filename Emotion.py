import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# 1. Định nghĩa đường dẫn tới thư mục chứa ảnh ----------------------------------------------------------------------------------------n
image_directory = r'C:\Users\ADMIN\Downloads\archive\images'  # Cập nhật lại đường dẫn 

# 2. Khai báo các lớp cảm xúc----------------------------------------------------------------------------------------
emotion_labels = ['Surprise', 'Sad', 'Neutral', 'Happy', 'Fear', 'Disgust', 'Angry']

# 3. Sử dụng ImageDataGenerator để tải ảnh và nhãn----------------------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,    # Xoay ảnh ngẫu nhiên
    width_shift_range=0.1,  # Dịch chuyển ảnh theo chiều ngang
    height_shift_range=0.1,  # Dịch chuyển ảnh theo chiều dọc
    shear_range=0.2,       # Biến dạng ảnh theo góc
    zoom_range=0.1,        # Phóng to/thu nhỏ ảnh
    horizontal_flip=True,  # Lật ảnh ngang
    fill_mode='nearest'    # Chế độ điền màu khi có phần bị thiếu sau khi biến đổi
)

vali_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    image_directory + '\\train',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)


vali_data = train_datagen.flow_from_directory(
    image_directory + '\\validation',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

# 4. Lấy các giá trị X và y từ generator----------------------------------------------------------------------------------------
X_train, y_train = next(vali_data)

# 5. Xây dựng mô hình CNN----------------------------------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Huấn luyện mô hình----------------------------------------------------------------------------------------
history = model.fit(
    train_data,
    epochs=15,
    batch_size=64,
    validation_data=vali_data,
)

# 7. Đánh giá mô hình trên tập kiểm tra----------------------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_train, y_train)  # Cập nhật tập kiểm tra
print(f"Accuracy on Validation Set: {test_acc:.2%}")

# 8. Dự đoán trên tập kiểm tra----------------------------------------------------------------------------------------
y_pred = model.predict(vali_data)  # Cập nhật dự đoán với validation_data
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = vali_data.classes

# 9. Trực quan hóa kết quả huấn luyện----------------------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Độ chính xác
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Quá trình cải thiện độ chính xác')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Mất mát
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Quá trình giảm thiểu mất mát')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 10. Ma trận nhầm lẫn----------------------------------------------------------------------------------------
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 11. Báo cáo đánh giá chi tiết----------------------------------------------------------------------------------------
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# 12. Tính toán độ chính xác trên từng lớp----------------------------------------------------------------------------------------
precision_per_class = []
recall_per_class = []
f1_per_class = []

# Tính toán precision, recall và f1-score cho từng lớp
for cls in range(7):  # 7 lớp cảm xúc
    cls_idx = np.where(y_true == cls)[0]
    if len(cls_idx) > 0:
        precision = precision_score(y_true[cls_idx], y_pred_classes[cls_idx], average='macro', zero_division=0)
        recall = recall_score(y_true[cls_idx], y_pred_classes[cls_idx], average='macro', zero_division=0)
        f1 = f1_score(y_true[cls_idx], y_pred_classes[cls_idx], average='macro', zero_division=0)
    else:
        precision = recall = f1 = 0
    
    precision_per_class.append(precision)
    recall_per_class.append(recall)
    f1_per_class.append(f1)

# Tính accuracy cho từng lớp
accuracy_per_class = []

for cls in range(7):
    cls_idx = np.where(y_true == cls)[0]
    correct_predictions = np.sum(y_pred_classes[cls_idx] == cls)
    accuracy = correct_predictions / len(cls_idx) if len(cls_idx) > 0 else 0
    accuracy_per_class.append(accuracy)

# Tạo DataFrame với Precision, Recall, F1-Score và Accuracy
df_classification = pd.DataFrame({
    "Class": emotion_labels,  # Cập nhật tên lớp cảm xúc
    "Accuracy": accuracy_per_class,
    "Precision": precision_per_class,
    "Recall": recall_per_class,
    "F1-Score": f1_per_class
})

# Hiển thị bảng kết quả
print(df_classification)

# 13. Vẽ bảng liệt kê----------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_classification.values, 
                 colLabels=df_classification.columns, 
                 loc='center', 
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df_classification.columns))))
plt.show()

# 14. Hiển thị 10 hình ảnh với dự đoán và tên cảm xúc----------------------------------------------------------------------------------------
plt.figure(figsize=(12, 8))
for i in range(10):
    # Đảm bảo chỉ số không vượt quá số lượng mẫu
    idx = i % len(X_train)
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[idx].reshape(48, 48), cmap='gray')  # Đảm bảo reshape đúng
    true_label = emotion_labels[np.argmax(y_train[idx])]  # Lấy nhãn đúng
    pred_label = emotion_labels[y_pred_classes[idx]]  # Lấy nhãn dự đoán
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()

# 15. Lưu mô hình----------------------------------------------------------------------------------------
model.save('emotion_recognition_model.h5')
print("Model saved as 'emotion_recognition_model.h5'")

# 16. Tính các chỉ số đánh giá----------------------------------------------------------------------------------------
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
