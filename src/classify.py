import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from base_onnx import BaseONNX, ModelType, TaskResult


class Classify(BaseONNX):
    """分类推理器"""
    
    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        """
        初始化分类推理器
        
        Args:
            model_path: ONNX模型路径
            device: 设备类型 ("auto", "cuda", "cpu")
            **kwargs: 其他参数
        """
        
        # 分类模型通常输入大小为224x224
        if 'input_size' not in kwargs:
            kwargs['input_size'] = (224, 224)
        
        super().__init__(model_path, ModelType.CLASSIFICATION, device, **kwargs)
    
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """预处理图像"""
        h, w = image.shape[:2]
        
        # 调整到输入尺寸
        resized = cv2.resize(image, self.input_size)
        
        # 标准化处理
        # ImageNet标准化参数 (确保为float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # 转换为模型输入格式 (RGB, 0-1范围)
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = (input_tensor - mean) / std
        
        # 转换为CHW格式并添加批次维度
        input_tensor = input_tensor.transpose(2, 0, 1).astype(np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        metadata = {
            'original_shape': (h, w),
            'input_shape': self.input_size
        }
        
        return input_tensor, metadata
    
    def postprocess(self, outputs: List[np.ndarray], metadata: Dict[str, Any]) -> TaskResult:
        """后处理分类结果"""
        result = TaskResult(ModelType.CLASSIFICATION)
        predictions = outputs[0]
        
        # 获取概率和类别ID
        if len(predictions.shape) == 2:
            predictions = predictions[0]  # 移除批次维度
        
        # 应用softmax
        exp_preds = np.exp(predictions - np.max(predictions))
        probabilities = exp_preds / np.sum(exp_preds)
        
        # 获取top-5结果
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        result.scores = [float(probabilities[i]) for i in top_indices]
        result.class_ids = [int(i) for i in top_indices]
        result.class_names = [self.class_names[i] if i < len(self.class_names) 
                            else f"Class_{i}" for i in top_indices]

        return result

