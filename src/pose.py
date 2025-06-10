import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from base_onnx import BaseONNX, ModelType, TaskResult


class Pose(BaseONNX):
    """姿态估计推理器"""
    
    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        """
        初始化姿态估计推理器
        
        Args:
            model_path: ONNX模型路径
            device: 设备类型 ("auto", "cuda", "cpu")
            **kwargs: 其他参数
        """
        
        super().__init__(model_path, ModelType.POSE, device, **kwargs)
        
        # COCO人体关键点信息
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 骨骼连接定义
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # 颜色配置
        self.keypoint_colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170)
        ]
        
        self.skeleton_colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
            (255, 0, 170), (255, 0, 85), (255, 0, 0)
        ]
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """预处理图像"""
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = min(self.input_size[0]/w, self.input_size[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 调整图片大小
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建填充后的图片
        padded = np.full((*self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # 转换为模型输入格式
        input_tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        metadata = {
            'scale': scale,
            'original_shape': (h, w),
            'resized_shape': (new_h, new_w),
            'input_shape': self.input_size
        }
        
        return input_tensor, metadata
    
    def postprocess(self, outputs: List[np.ndarray], metadata: Dict[str, Any]) -> TaskResult:
        """后处理姿态估计结果 - YOLO11姿态模型（56通道）"""
        result = TaskResult(ModelType.POSE)
        predictions = outputs[0]
        scale = metadata['scale']
        orig_h, orig_w = metadata['original_shape']
        
        # YOLO11姿态模型输出格式: [batch, 56, num_detections] -> [batch, num_detections, 56]
        # 56 = 4(bbox) + 1(conf) + 51(17*3 keypoints)
        if len(predictions.shape) == 3:
            if predictions.shape[1] == 56:  # [1, 56, N] 格式
                predictions = predictions.transpose(0, 2, 1)  # [1, N, 56]
            elif predictions.shape[2] == 56:  # 已经是 [1, N, 56] 格式
                pass
            else:
                raise ValueError(f"姿态模型输出形状错误，期望56通道: {predictions.shape}")
        
        # 验证是姿态模型格式
        num_channels = predictions.shape[2]
        if num_channels != 56:
            raise ValueError(f"姿态模型期望56通道，实际得到: {num_channels}")
        
        boxes, scores, keypoints_list = [], [], []
        
        for i, detection in enumerate(predictions[0]):
            # detection: [x_center, y_center, width, height, conf, kpt1_x, kpt1_y, kpt1_v, ..., kpt17_x, kpt17_y, kpt17_v]
            bbox = detection[:4]  # 边界框坐标
            conf = detection[4]   # 置信度
            keypoints = detection[5:56]  # 51个关键点值 (17*3)
            
            # 检查bbox数据是否有效
            if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                continue
                
            # 检查置信度
            if np.isnan(conf) or np.isinf(conf) or conf <= self.conf_threshold:
                continue
            
            x_center, y_center, width, height = bbox
            
            # 检查bbox尺寸是否有效
            if width <= 0 or height <= 0:
                continue
            
            # 坐标转换到原图尺寸
            x1 = int((x_center - width/2) / scale)
            y1 = int((y_center - height/2) / scale)
            x2 = int((x_center + width/2) / scale)
            y2 = int((y_center + height/2) / scale)
            
            # 确保坐标在合理范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            # 验证转换后的坐标
            if x2 <= x1 or y2 <= y1:
                continue
                
            # 检查最小面积
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < 100:  # 姿态检测需要较大的目标
                continue
            
            # 转换关键点坐标为YOLO格式 (17, 2) numpy数组
            scaled_keypoints = []
            valid_keypoints = 0
            
            for j in range(0, 51, 3):  # 17个关键点，每个3个值
                if j + 2 < len(keypoints):
                    kpt_x, kpt_y, kpt_v = keypoints[j:j+3]
                    
                    # 检查关键点数据是否有效
                    if not (np.isnan(kpt_x) or np.isnan(kpt_y) or np.isnan(kpt_v)):
                        # 转换坐标到原图尺寸
                        scaled_x = kpt_x / scale
                        scaled_y = kpt_y / scale
                        
                        # 确保关键点在图像范围内
                        scaled_x = max(0, min(orig_w, scaled_x))
                        scaled_y = max(0, min(orig_h, scaled_y))
                        
                        # 只保存x, y坐标，丢弃可见性信息（模仿YOLO格式）
                        scaled_keypoints.append([scaled_x, scaled_y])
                        
                        if kpt_v > 0.5:  # 可见性阈值
                            valid_keypoints += 1
                    else:
                        # 无效关键点用[0, 0]表示
                        scaled_keypoints.append([0.0, 0.0])
            
            # 只保留有足够可见关键点的检测结果
            if valid_keypoints >= 3 and len(scaled_keypoints) == 17:
                # 转换为numpy数组格式 (17, 2)
                keypoints_array = np.array(scaled_keypoints, dtype=np.int32)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(conf))
                keypoints_list.append(keypoints_array)
        
        # 非极大值抑制
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes],
                scores, self.conf_threshold, self.iou_threshold
            )
            
            if len(indices) > 0:
                result.boxes = [boxes[i] for i in indices.flatten()]
                result.scores = [scores[i] for i in indices.flatten()]
                result.class_ids = [0] * len(result.boxes)  # 人类类别
                result.class_names = ["person"] * len(result.boxes)
                result.keypoints = [keypoints_list[i] for i in indices.flatten()]
        
        return result
