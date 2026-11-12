深度学习目标检测算法yolo训练铝材料缺陷数据集 铝型材表面缺陷数据集训练自己的权重识别铝材缺陷中的不导电，擦花，角漏，桔皮，漏底，喷流，漆泡，起坑，杂色，脏点。
@[toc]

铝型材表面缺陷数据集
十类缺陷 分别为 不导电，擦花，角漏，桔皮，漏底，喷流，漆泡，起坑，杂色，脏点。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0dbade7c822b4562aa7ec1c3ed028eb8.png)


标签为txt格式
共分十类["aoxian", "budaodian", "cahua", "jupi", "loudi", "pengshang", "qikeng", "tufen", "tucengkailie", "zangdian"]
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cec766809fd443b2909cc55b8ad4f8fb.png)

训练集：1508张
测试集：95张
验证集：282张
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a5fa02464d844d6d8a02b78933d56695.png)
基于YOLOv8的铝材缺陷监测系统，我们需要经历数据准备、环境搭建、模型训练、推理和界面开发等步骤。以下是详细的指南及代码示例。

### 数据准备

确保你的数据集已准备好，并按照以下结构组织：

```
aluminum_defects/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

- `images/train/`, `images/val/`, `images/test/` 分别存放训练、验证和测试图像。
- `labels/train/`, `labels/val/`, `labels/test/` 存放对应图像的标签文件（YOLO格式）。

每个标签文件包含一行信息，格式如下：`class_id center_x center_y width height`，所有坐标值都是相对于图像尺寸归一化的。

### 数据配置

创建一个名为 `data.yaml` 的文件来描述数据集的路径和类别信息：

```yaml
train: ./aluminum_defects/images/train/
val: ./aluminum_defects/images/val/
test: ./aluminum_defects/images/test/

nc: 10 # 类别数量
names: ['aoxian', 'budaodian', 'cahua', 'jupi', 'loudi', 'pengshang', 'qikeng', 'tufen', 'tucengkailie', 'zangdian'] # 类别名称
```

### 环境搭建

安装必要的库：

```bash
pip install ultralytics opencv-python
```

### 模型训练

使用YOLOv8进行模型训练：

```bash
yolo detect train model=yolov8n.pt data=data.yaml batch=16 epochs=100 imgsz=640 device=cuda
```

这里，`epochs` 设置为100，可以根据需要调整。`device=cuda` 表示使用GPU加速训练。

### 推理代码

训练完成后，可以编写Python脚本对新图像进行推理：

```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

def infer_image(image_path):
    results = model.predict(source=image_path, imgsz=640)
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy()

        image = cv2.imread(image_path)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{r.names[int(label)]} {score:.4f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Inference Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 测试图像路径
infer_image('path/to/test/image.jpg')
```

### 批量推理

如果要进行批量推理，可以修改上述脚本以遍历指定目录下的所有图像：

```python
import os

def infer_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, folder_path, filename)
            infer_image(image_path)

# 定义测试图像目录
test_dir = 'path/to/test/images/'
infer_images_in_folder(test_dir)
```

### 性能评估

手动运行性能评估：

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640
```

这将输出详细的性能评估报告，帮助你了解模型在验证集上的表现。

### 构建用户界面

我们可以使用`tkinter`库来构建一个简单的GUI，用于选择图像或视频进行检测：

```python
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class DefectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("铝材缺陷检测系统")
        
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        self.create_button("选择图片", self.select_image)
        self.create_button("开始检测", self.start_detection)
        
        self.model = YOLO('runs/detect/train/weights/best.pt')
    
    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command)
        button.pack(pady=5)
    
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.image_path = file_path
            self.show_image(file_path)
    
    def show_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((600, 400))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
    
    def start_detection(self):
        if hasattr(self, 'image_path'):
            infer_image(self.image_path)
        else:
            messagebox.showwarning("警告", "请先选择一张图片")

if __name__ == "__main__":
    root = tk.Tk()
    app = DefectDetectionApp(root)
    root.mainloop()
```

