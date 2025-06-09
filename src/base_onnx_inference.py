import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import json


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


class BaseONNXInference(ABC):
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
        self.class_names = kwargs.get('class_names', [])
        
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
        
        # 绘制检测框（如果有）
        if result.boxes:
            result_image = self._draw_detection(image, result)
        
        # 绘制分割掩码
        for i, mask in enumerate(result.masks):
            if isinstance(mask, np.ndarray):
                # 创建彩色掩码
                color = np.random.randint(0, 255, 3).tolist()
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0] = color
                
                # 与原图混合
                result_image = cv2.addWeighted(result_image, 0.8, colored_mask, 0.2, 0)
        
        return result_image
    
    def _draw_pose(self, image: np.ndarray, result: TaskResult) -> np.ndarray:
        """绘制姿态估计结果"""
        result_image = image.copy()
        
        # 绘制检测框（如果有）
        if result.boxes:
            result_image = self._draw_detection(image, result)
        
        # 绘制关键点
        for keypoints in result.keypoints:
            if isinstance(keypoints, (list, np.ndarray)) and len(keypoints) >= 17 * 3:  # COCO 17个关键点
                # 重新整形为 (17, 3) - x, y, visibility
                kpts = np.array(keypoints).reshape(-1, 3)
                
                # 绘制关键点
                for x, y, v in kpts:
                    if v > 0.5:  # 可见性阈值
                        cv2.circle(result_image, (int(x), int(y)), 3, (0, 0, 255), -1)
                
                # 绘制骨骼连接线
                skeleton = [
                    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                    [2, 4], [3, 5], [4, 6], [5, 7]
                ]
                
                for connection in skeleton:
                    kpt1, kpt2 = connection
                    if kpt1 - 1 < len(kpts) and kpt2 - 1 < len(kpts):
                        x1, y1, v1 = kpts[kpt1 - 1]
                        x2, y2, v2 = kpts[kpt2 - 1]
                        if v1 > 0.5 and v2 > 0.5:
                            cv2.line(result_image, (int(x1), int(y1)), (int(x2), int(y2)), 
                                   (255, 0, 0), 2)
        
        return result_image


class RTSPProcessor:
    """RTSP流处理器"""
    
    def __init__(self, inference: BaseONNXInference):
        """
        初始化RTSP处理器
        
        Args:
            inference: ONNX推理实例
        """
        self.inference = inference
        self.cap = None
    
    def process_stream(self, rtsp_url: str, max_frames: Optional[int] = None, 
                      display: bool = False, save_results: bool = False,
                      output_dir: str = "rtsp_results"):
        """
        处理RTSP视频流
        
        Args:
            rtsp_url: RTSP流地址
            max_frames: 最大处理帧数
            display: 是否显示实时画面
            save_results: 是否保存结果
            output_dir: 输出目录
        """
        print(f"连接RTSP流: {rtsp_url}")
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        if not self.cap.isOpened():
            raise RuntimeError("无法连接到RTSP流")
        
        # 创建输出目录
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        frame_count = 0
        total_inference_time = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取帧，重新连接...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(rtsp_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    continue
                
                frame_count += 1
                start_time = time.time()
                
                # 推理
                result = self.inference.predict(frame)
                
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                # 打印统计信息
                if self.inference.model_type == ModelType.DETECTION:
                    print(f"帧 {frame_count:6d} | "
                          f"推理时间: {inference_time*1000:6.2f}ms | "
                          f"检测目标: {len(result.boxes):2d}")
                else:
                    print(f"帧 {frame_count:6d} | "
                          f"推理时间: {inference_time*1000:6.2f}ms | "
                          f"任务类型: {result.task_type.value}")
                
                # 每100帧打印平均统计
                if frame_count % 100 == 0:
                    avg_inference = (total_inference_time / frame_count) * 1000
                    fps = 1.0 / (avg_inference / 1000) if avg_inference > 0 else 0
                    print(f"前 {frame_count} 帧平均推理时间: {avg_inference:.2f}ms, 理论FPS: {fps:.2f}")
                
                # 显示结果
                if display:
                    result_image = self.inference.draw_results(frame, result)
                    
                    # 添加信息
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if self.inference.model_type == ModelType.DETECTION:
                        info_text = f"Time: {timestamp} | Objects: {len(result.boxes)} | Frame: {frame_count}"
                    else:
                        info_text = f"Time: {timestamp} | Type: {result.task_type.value} | Frame: {frame_count}"
                    
                    cv2.putText(result_image, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    display_image = cv2.resize(result_image, (960, 540))
                    cv2.imshow(f'RTSP {self.inference.model_type.value}', display_image)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 保存结果
                if save_results and (result.boxes or result.scores):
                    result_image = self.inference.draw_results(frame, result)
                    output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(output_path, result_image)
                
                # 检查最大帧数
                if max_frames and frame_count >= max_frames:
                    print(f"达到最大帧数限制: {max_frames}")
                    break
                    
        except KeyboardInterrupt:
            print("\n用户中断，正在退出...")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
        finally:
            if self.cap:
                self.cap.release()
            if display:
                cv2.destroyAllWindows()
            print(f"处理完成，共处理 {frame_count} 帧")
