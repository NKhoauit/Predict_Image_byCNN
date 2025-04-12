import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from PIL import Image, ImageChops, ImageEnhance
import os

model_path = r'E:\UIT\Hoc_ki_6\Forensic\Project\model_run1.h5'
real_path = r'E:\UIT\Hoc_ki_6\Forensic\Project\Dataset\Au'
fake_path = r'E:\UIT\Hoc_ki_6\Forensic\Project\Dataset\Tp'
image_size = (128, 128)

# === Hàm chuyển đổi ảnh sang dạng ELA (Error Level Analysis) ===
def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp.jpg'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

# === Hàm tiền xử lý ảnh (tạo ELA + resize + chuẩn hoá pixel) ===
def prepare_image(image_path):
    try:
        ela = convert_to_ela_image(image_path)
        return np.array(ela.resize(image_size)) / 255.0
    except:
        print(f"Lỗi ảnh: {image_path}")
        return None

# === Load dữ liệu ảnh thật và giả, gán nhãn tương ứng ===
X, y = [], []
for path, label in [(real_path, 1), (fake_path, 0)]:
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('jpg', 'png')):
                img = prepare_image(os.path.join(root, file))
                if img is not None:
                    X.append(img)
                    y.append(label)

print(f"Tổng số ảnh dùng để huấn luyện: {len(X)}")

# === Tiền xử lý đầu vào cho mô hình CNN ===
X = np.array(X).reshape(-1, 128, 128, 3)
y = to_categorical(y, 2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5)

# === Xây dựng kiến trúc mô hình CNN ===
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(128, 128, 3))) # conv1
    model.add(Conv2D(32, (5, 5), activation='relu'))                            # conv2
    model.add(MaxPool2D((2, 2)))                                                # giảm chiều không gian
    model.add(Dropout(0.25))                                                    # tránh overfitting
    model.add(Flatten())                                                        # chuyển ảnh thành vector
    model.add(Dense(256, activation='relu'))                                    # fully connected
    model.add(Dropout(0.5))                                                     # dropout tiếp
    model.add(Dense(2, activation='softmax'))                                   # lớp đầu ra (2 lớp: real/fake)
    return model

# === Biên dịch và huấn luyện mô hình ===
model = build_model()
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện và dừng sớm nếu không cải thiện accuracy validation
hist = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=2)]
)

# Lưu model sau khi huấn luyện
model.save(model_path)
print(" Mô hình đã lưu vào:", model_path)

# === Vẽ biểu đồ theo dõi quá trình huấn luyện ===
fig, ax = plt.subplots(2, 1)
ax[0].plot(hist.history['loss'], label="Train Loss")
ax[0].plot(hist.history['val_loss'], label="Val Loss")
ax[0].legend()
ax[1].plot(hist.history['accuracy'], label="Train Acc")
ax[1].plot(hist.history['val_accuracy'], label="Val Acc")
ax[1].legend()
plt.tight_layout()
plt.show()
