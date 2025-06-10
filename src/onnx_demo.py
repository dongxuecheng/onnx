from detect import Detect
from segment import Segment
from classify import Classify
from pose import Pose
import cv2

# image = cv2.imread("images/human2.jpg")
# model=Detect(model_path="model/yolo11n.onnx")
# result=model.predict(image)

# for box, score, class_name in zip(result.boxes, result.scores, result.class_names):
#     #print(f"目标 {i+1}: {class_name} - {score:.3f} - {box}")
#     if class_name == "person":
#         print(box)

# 绘制结果
# result_image = model.draw_results(image, result)
# cv2.imwrite("result/detection_result2.jpg", result_image)

image = cv2.imread("images/human3.jpg")
model=Segment(model_path="model/yolo11l-seg.onnx",imput_size=(1280,1280))
result=model.predict(image)

for box, score, class_name,mask in zip(result.boxes, result.scores, result.class_names,result.masks):
    # print(f"目标{class_name} - 置信度: {score:.3f}")
    # print(f"  边界框: {box}")
    # print(f"  掩码: {mask}")
    print(mask)

result_image = model.draw_results(image, result)
cv2.imwrite("result/seg_result4.jpg", result_image)

# image = cv2.imread("images/human3.jpg")
# model=Classify(model_path="model/yolo11n-cls.onnx")
# result=model.predict(image)

# for class_name, score in zip(result.class_names, result.scores):
#     print(f"{class_name}: {score:.4f}")

# top_class = result.class_names[0]
# top_score = result.scores[0]
# print(f"Top class: {top_class} with score: {top_score:.4f}")
# result_image = model.draw_results(image, result)
# cv2.imwrite("result/cls_result3.jpg", result_image)

# image = cv2.imread("images/human3.jpg")
# model=Pose(model_path="model/yolo11l-pose.onnx")
# result=model.predict(image)

# for box, score, class_name, keypoint in zip(result.boxes, result.scores, result.class_names, result.keypoints):
#     print(f"目标 {class_name} - 置信度: {score:.3f}")
#     print(f"  边界框: {box}")
#     print(f"  关键点: {keypoint}")
# result_image = model.draw_results(image, result)
# cv2.imwrite("result/pose_result3.jpg", result_image)