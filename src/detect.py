import cv2
import numpy as np
import ast  # 导入 ast 模块
from typing import List, Tuple, Dict, Any
from base_onnx import BaseONNX, ModelType, TaskResult


class Detect(BaseONNX):
    """目标检测推理器"""
    
    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        """
        初始化检测推理器
        
        Args:
            model_path: ONNX模型路径
            device: 设备类型 ("auto", "cuda", "cpu")
            **kwargs: 其他参数
        """        
        super().__init__(model_path, ModelType.DETECTION, device, **kwargs)
    

    
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
        """后处理检测结果 - 处理检测模型"""
        result = TaskResult(ModelType.DETECTION)
        predictions = outputs[0]
        scale = metadata['scale']
        orig_h, orig_w = metadata['original_shape']
        
        # 动态获取类别数量
        num_classes = len(self.class_names) if self.class_names else 80
        expected_channels = 4 + num_classes  # 4(bbox) + num_classes
        
        # YOLO检测模型输出格式处理: [batch, channels, num_detections] -> [batch, num_detections, channels]
        if len(predictions.shape) == 3:
            if predictions.shape[1] == expected_channels:  # [1, channels, N] 格式
                predictions = predictions.transpose(0, 2, 1)  # [1, N, channels]
            elif predictions.shape[2] == expected_channels:  # 已经是 [1, N, channels] 格式
                pass
            else:
                raise ValueError(f"检测模型输出形状错误，期望{expected_channels}通道: {predictions.shape}")
        
        # 验证是检测模型格式
        num_channels = predictions.shape[2]
        if num_channels != expected_channels:
            raise ValueError(f"检测模型期望{expected_channels}通道，实际得到: {num_channels}")
        
        boxes, scores, class_ids = [], [], []
        
        for i, detection in enumerate(predictions[0]):
            # detection: [x_center, y_center, width, height, class_0, class_1, ..., class_n]
            bbox = detection[:4]  # 边界框坐标
            class_scores = detection[4:4+num_classes]  # 类别分数
            
            # 检查bbox数据是否有效
            if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                continue
                
            # 检查类别分数是否有效
            if np.any(np.isnan(class_scores)) or np.any(np.isinf(class_scores)):
                continue
                
            # 获取最高分数的类别
            max_score = np.max(class_scores)
            
            # 置信度阈值检查
            if max_score <= self.conf_threshold:
                continue
            
            class_id = np.argmax(class_scores)
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
                
            # 检查最小面积（至少16个像素）
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < 16:
                continue
            
            # 验证类别ID
            if class_id < 0 or class_id >= num_classes:
                continue
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(max_score))
            class_ids.append(int(class_id))
        
        # 非极大值抑制
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes],
                scores, self.conf_threshold, self.iou_threshold
            )
            
            if len(indices) > 0:
                result.boxes = [boxes[i] for i in indices.flatten()]
                result.scores = [scores[i] for i in indices.flatten()]
                result.class_ids = [class_ids[i] for i in indices.flatten()]
                result.class_names = [self.class_names[cid] if cid < len(self.class_names) 
                                    else f"Class_{cid}" for cid in result.class_ids]
        
        return result

