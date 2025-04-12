import numpy as np
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os

model_path = r'E:\UIT\Hoc_ki_6\Forensic\Project\model_run1.h5'
test_folder = r'E:\UIT\Hoc_ki_6\Forensic\Project\Dataset\test_image'
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

# Load Model đã được train
model = load_model(model_path)
print(" Model đã được load!")

# Dự đoán hình ảnh
def predict_folder(folder_path):
    print(f"\n Dự đoán ảnh trong thư mục: {folder_path}")
    count = 0
    for file in os.listdir(folder_path):                        # Duyệt qua tất cả các file trong thư mục
        if file.lower().endswith(('jpg', 'jpeg', 'png')):       # Kiểm tra định dạng file là hình ảnh (jpg, jpeg, png)
            path = os.path.join(folder_path, file)              # Ghép đường dẫn đầy đủ tới file ảnh
            img = prepare_image(path)                           # Gọi hàm tiền xử lý ảnh: chuyển sang ảnh ELA, resize, normalize
            if img is None:                                     # Nếu ảnh lỗi (không đọc được), bỏ qua
                continue
            img = img.reshape(-1, 128, 128, 3)
            pred = model.predict(img)                           # Dự đoán ảnh bằng mô hình đã huấn luyện
            label = "Real" if np.argmax(pred) == 1 else "Fake"   # Lấy nhãn dự đoán: 1 là real, 0 là fake
            confidence = np.max(pred) * 100                      # Tính độ tin cậy của dự đoán (xác suất lớn nhất)
            print(f"{file:<40} ➜ {label:<5} ({confidence:.2f}%)")  # In kết quả ra màn hình 
            count += 1  
    if count == 0:
        print(" Không có ảnh hợp lệ!")

predict_folder(test_folder)                         # Gọi hàm dự đoán hình ảnh
