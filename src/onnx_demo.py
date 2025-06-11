from detect import Detect
from segment import Segment
from classify import Classify
from pose import Pose
import cv2

# image = cv2.imread("images/human2.jpg")
# model=Detect(model_path="model/yolo11n.onnx",device="cuda:1")
# result=model.predict(image)

# for box, score, class_name in zip(result.boxes, result.scores, result.class_names):
#     if score > 0.5:
#         print(f"目标: {class_name} - {score:.3f} - {box}")
#     # if class_name == "person":
#     #     print(box)

# #绘制结果
# result.save("result/detect_result6.jpg")

image = cv2.imread("images/human3.jpg")
model=Segment(model_path="model/yolo11l-seg.onnx",imput_size=(1280,1280))
result=model.predict(image)

for box, score, class_name,mask in zip(result.boxes, result.scores, result.class_names,result.masks):
    # print(f"目标{class_name} - 置信度: {score:.3f}")
    # print(f"  边界框: {box}")
    # print(f"  掩码: {mask}")
    print(mask)

result.save("result/seg_result6.jpg")

# image = cv2.imread("images/human3.jpg")
# model=Classify(model_path="model/yolo11n-cls.onnx")
# result=model.predict(image)

# for class_name, score in zip(result.class_names, result.scores):
#     print(f"{class_name}: {score:.4f}")

# top_class = result.class_names[0]
# top_score = result.scores[0]
# print(f"Top class: {top_class} with score: {top_score:.4f}")
# result.save("result/cls_result6.jpg")

# image = cv2.imread("images/human3.jpg")
# model=Pose(model_path="model/yolo11l-pose.onnx")
# result=model.predict(image)

# for box, score, class_name, keypoint in zip(result.boxes, result.scores, result.class_names, result.keypoints):
#     print(f"目标 {class_name} - 置信度: {score:.3f}")
#     print(f"  边界框: {box}")
#     print(f"  关键点: {keypoint}")
# result.save("result/pose_result6.jpg")