import cv2
import numpy as np
import onnxruntime as ort
import ast
import onnx
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum


class ModelType(Enum):
    """模型类型枚举"""
    DETECTION = "detection"         # 目标检测
    CLASSIFICATION = "classification"  # 分类
    SEGMENTATION = "segmentation"      # 分割
    POSE = "pose"                   # 姿态估计


class TaskResult:
    """任务结果统一格式"""
    def __init__(self, task_type: ModelType):
        self.task_type = task_type
        self.boxes = []         # 检测框 [x1, y1, x2, y2]
        self.scores = []        # 置信度
        self.class_ids = []     # 类别ID
        self.class_names = []   # 类别名称
        self.masks = []         # 分割掩码
        self.keypoints = []     # 关键点
        self.features = []      # 特征向量（分类）
        self.extra_data = {}    # 额外数据
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'task_type': self.task_type.value,
            'boxes': self.boxes,
            'scores': self.scores,
            'class_ids': self.class_ids,
            'class_names': self.class_names,
            'masks': [mask.tolist() if isinstance(mask, np.ndarray) else mask for mask in self.masks],
            'keypoints': self.keypoints,
            'features': self.features,
            'extra_data': self.extra_data
        }


class BaseONNX(ABC):
    """ONNX推理基类"""
    
    def __init__(self, model_path: str, model_type: ModelType, device: str = "auto", **kwargs):
        """
        初始化ONNX推理器
        
        Args:
            model_path: ONNX模型路径
            model_type: 模型类型
            device: 设备类型 ("auto", "cuda", "cpu")
            **kwargs: 其他参数
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        
        # 默认参数
        self.conf_threshold = kwargs.get('conf_threshold', 0.25)
        self.iou_threshold = kwargs.get('iou_threshold', 0.45)
        self.input_size = kwargs.get('input_size', (640, 640))
        self.class_names = self._get_model_names()
        
        # 初始化模型
        self._load_model()
        
    def _load_model(self):
        """加载ONNX模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
        print(f"加载ONNX模型: {self.model_path}")
        
        try:
            # 配置执行提供者
            providers = self._get_providers()
            
            # 创建会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # 创建推理会话
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
            
            # 获取模型输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # 显示模型信息
            used_providers = self.session.get_providers()
            print(f"模型加载成功 - {self.model_type.value}")
            print(f"输入名称: {self.input_name}")
            print(f"输入形状: {self.input_shape}")
            print(f"输出名称: {self.output_names}")
            print(f"执行提供者: {used_providers}")
            
            if 'CUDAExecutionProvider' in used_providers:
                print("✓ GPU加速已启用 - 使用CUDA")
            else:
                print("⚠ 使用CPU运行")
                
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _get_providers(self) -> List:
        """获取执行提供者配置"""
        if self.device == "cpu":
            return ['CPUExecutionProvider']
        elif self.device == "cuda":
            return [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
        else:  # auto
            return [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
    
    def _get_model_names(self) -> List[str]:
        model = onnx.load(self.model_path)
        metadata = model.metadata_props
        for prop in metadata:
            if prop.key == "names":   
                label_mapping = ast.literal_eval(prop.value)
                return list(label_mapping.values())


    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[预处理后的张量, 元数据字典]
        """
        pass
    
    @abstractmethod
    def postprocess(self, outputs: List[np.ndarray], metadata: Dict[str, Any]) -> TaskResult:
        """
        后处理模型输出
        
        Args:
            outputs: 模型输出
            metadata: 预处理元数据
            
        Returns:
            TaskResult: 统一的任务结果
        """
        pass
    
    def predict(self, image: np.ndarray) -> TaskResult:
        """
        预测单张图像
        
        Args:
            image: 输入图像
            
        Returns:
            TaskResult: 预测结果
        """
        # 预处理
        input_tensor, metadata = self.preprocess(image)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # 后处理
        result = self.postprocess(outputs, metadata)
        
        return result
    
    def predict_batch(self, images: List[np.ndarray]) -> List[TaskResult]:
        """
        批量预测
        
        Args:
            images: 图像列表
            
        Returns:
            List[TaskResult]: 预测结果列表
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def draw_results(self, image: np.ndarray, result: TaskResult) -> np.ndarray:
        """
        根据任务类型绘制结果
        
        Args:
            image: 原始图像
            result: 任务结果
            
        Returns:
            绘制结果的图像
        """
        if self.model_type == ModelType.DETECTION:
            return self._draw_detection(image, result)
        elif self.model_type == ModelType.CLASSIFICATION:
            return self._draw_classification(image, result)
        elif self.model_type == ModelType.SEGMENTATION:
            return self._draw_segmentation(image, result)
        elif self.model_type == ModelType.POSE:
            return self._draw_pose(image, result)
        else:
            return image.copy()
    
    def _draw_detection(self, image: np.ndarray, result: TaskResult) -> np.ndarray:
        """绘制检测结果"""
        result_image = image.copy()
        
        for i, (box, score, class_id) in enumerate(zip(result.boxes, result.scores, result.class_ids)):
            x1, y1, x2, y2 = map(int, box)
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 获取类别名称
            if i < len(result.class_names):
                class_name = result.class_names[i]
            elif 0 <= class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class_{class_id}"
            
            # 绘制标签
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # 绘制标签文字
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
    def _draw_classification(self, image: np.ndarray, result: TaskResult) -> np.ndarray:
        """绘制分类结果"""
        result_image = image.copy()
        
        # 在图像上显示分类结果
        y_offset = 30
        for i, (score, class_id) in enumerate(zip(result.scores, result.class_ids)):
            if i < len(result.class_names):
                class_name = result.class_names[i]
            elif 0 <= class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class_{class_id}"
            
            text = f"{class_name}: {score:.3f}"
            cv2.putText(result_image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        return result_image
    
    def _draw_segmentation(self, image: np.ndarray, result: TaskResult) -> np.ndarray:
        """绘制分割结果"""
        result_image = image.copy()

        
        # 绘制轮廓（先绘制轮廓，再绘制框，这样框在上层）
        if result.masks:
            for i, (contour_points, class_id) in enumerate(zip(result.masks, result.class_ids)):
                if len(contour_points) > 0:  # 检查是否有轮廓点
                    # 获取类别对应的颜色
                    color = np.random.randint(0, 255, 3).tolist()
                    # 确保点坐标为int32类型用于绘制
                    pts = contour_points.astype(np.int32)
                    
                    # 绘制填充的多边形（半透明效果）
                    overlay = result_image.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    alpha = 0.4
                    result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
                    
                    # 绘制轮廓线
                    cv2.polylines(result_image, [pts], True, color, 2)
        
        # 绘制检测框和标签
        if result.boxes:
            for i, (box, score, class_id, class_name) in enumerate(
                zip(result.boxes, result.scores, result.class_ids, result.class_names)
            ):
                x1, y1, x2, y2 = map(int, box)
                
                # 获取类别对应的颜色
                color = np.random.randint(0, 255, 3).tolist()
                
                # 绘制边界框
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签背景
                label = f"{class_name}: {score:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # 绘制标签文本
                cv2.putText(result_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def _draw_pose(self, image: np.ndarray, result: TaskResult) -> np.ndarray:
        """绘制姿态估计结果"""
        result_image = image.copy()
        
        # 绘制检测框（如果有）
        if result.boxes:
            for i, (box, score) in enumerate(zip(result.boxes, result.scores)):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制置信度
                label = f"Person: {score:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(result_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 绘制关键点和骨骼
        for keypoints in result.keypoints:
            if isinstance(keypoints, np.ndarray) and keypoints.shape == (17, 2):
                # 绘制骨骼连接线
                for i, connection in enumerate(self.skeleton):
                    kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1  # 转换为0-based索引
                    if 0 <= kpt1_idx < len(keypoints) and 0 <= kpt2_idx < len(keypoints):
                        x1, y1 = keypoints[kpt1_idx]
                        x2, y2 = keypoints[kpt2_idx]
                        # 检查关键点是否有效（不为[0, 0]）
                        if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
                            color = self.skeleton_colors[i % len(self.skeleton_colors)]
                            cv2.line(result_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                                   color, 2)
                
                # 绘制关键点
                for i, (x, y) in enumerate(keypoints):
                    if x > 0 or y > 0:  # 有效关键点
                        color = self.keypoint_colors[i % len(self.keypoint_colors)]
                        cv2.circle(result_image, (int(x), int(y)), 4, color, -1)
                        cv2.circle(result_image, (int(x), int(y)), 6, (0, 0, 0), 2)
        
        return result_image

