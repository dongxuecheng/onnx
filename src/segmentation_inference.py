import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from base_onnx_inference import BaseONNXInference, ModelType, TaskResult


class SegmentationInference(BaseONNXInference):
    """分割推理器"""
    
    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        """
        初始化分割推理器
        
        Args:
            model_path: ONNX模型路径
            device: 设备类型 ("auto", "cuda", "cpu")
            **kwargs: 其他参数
        """
        # 设置默认类别名称
        if 'class_names' not in kwargs:
            kwargs['class_names'] = self._get_coco_names()
        
        super().__init__(model_path, ModelType.SEGMENTATION, device, **kwargs)
    
    def _get_coco_names(self) -> List[str]:
        """获取COCO数据集类别名称"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
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
        """后处理分割结果 - 仅处理分割模型（116通道）"""
        result = TaskResult(ModelType.SEGMENTATION)
        predictions = outputs[0]
        scale = metadata['scale']
        orig_h, orig_w = metadata['original_shape']
        
        # YOLO11分割模型输出格式: [batch, 116, num_detections] -> [batch, num_detections, 116]
        # 116 = 4(bbox) + 80(classes) + 32(masks)
        if len(predictions.shape) == 3:
            if predictions.shape[1] == 116:  # [1, 116, N] 格式
                predictions = predictions.transpose(0, 2, 1)  # [1, N, 116]
            elif predictions.shape[2] == 116:  # 已经是 [1, N, 116] 格式
                pass
            else:
                raise ValueError(f"分割模型输出形状错误，期望116通道: {predictions.shape}")
        
        # 验证是分割模型格式
        num_channels = predictions.shape[2]
        if num_channels != 116:
            raise ValueError(f"分割模型期望116通道，实际得到: {num_channels}")
        
        boxes, scores, class_ids, mask_coeffs = [], [], [], []
        
        for i, detection in enumerate(predictions[0]):
            # detection: [x_center, y_center, width, height, class_0, class_1, ..., class_79, mask_0, ..., mask_31]
            bbox = detection[:4]  # 边界框坐标
            class_scores = detection[4:84]  # 80个类别分数 (COCO数据集)
            mask_coeff = detection[84:116]  # 32个掩码系数
            
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
            
            # 验证类别ID - COCO有80个类别
            if class_id < 0 or class_id >= 80:
                continue
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(max_score))
            class_ids.append(int(class_id))
            mask_coeffs.append(mask_coeff)
        
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
                
                # 处理掩码系数（这里只保存系数，实际掩码生成需要原型掩码）
                # 在实际应用中，你可能需要额外的原型掩码来生成最终的分割掩码
                result.extra_data['mask_coeffs'] = [mask_coeffs[i] for i in indices.flatten()]
                
                # 如果有额外的掩码输出，处理它们
                if len(outputs) > 1:
                    self._process_masks(outputs[1:], result, metadata)
        
        return result
    
    def _process_masks(self, mask_outputs: List[np.ndarray], result: TaskResult, metadata: Dict[str, Any]):
        """处理掩码输出 - 生成完整的分割掩码"""
        if 'mask_coeffs' not in result.extra_data or len(mask_outputs) == 0:
            return
            
        proto_masks = mask_outputs[0]  # 掩码原型 [mask_dim, mask_h, mask_w]
        
        # 移除批次维度（如果存在）
        if len(proto_masks.shape) == 4:
            proto_masks = proto_masks[0]
        
        orig_h, orig_w = metadata['original_shape']
        scale = metadata['scale']
        
        # 为每个检测到的目标生成掩码
        for i, (mask_coeff, box) in enumerate(zip(result.extra_data['mask_coeffs'], result.boxes)):
            try:
                # 使用掩码系数和原型生成掩码
                # mask_coeff: [32], proto_masks: [32, mask_h, mask_w]
                mask = np.dot(mask_coeff, proto_masks.reshape(proto_masks.shape[0], -1))
                mask = mask.reshape(proto_masks.shape[1:])  # [mask_h, mask_w]
                
                # 应用sigmoid激活
                mask = 1.0 / (1.0 + np.exp(-mask))
                
                # 将掩码从模型输出尺寸调整到原图尺寸
                # 先调整到输入尺寸
                mask_resized = cv2.resize(mask, self.input_size)
                
                # 然后裁剪到实际使用的区域（去除padding）
                resized_h, resized_w = metadata['resized_shape']
                mask_cropped = mask_resized[:resized_h, :resized_w]
                
                # 最后调整到原图尺寸
                mask_final = cv2.resize(mask_cropped, (orig_w, orig_h))
                
                # 应用阈值
                mask_binary = (mask_final > 0.5).astype(np.uint8)
                
                # 可选：将掩码限制在检测框内
                x1, y1, x2, y2 = box
                mask_in_box = np.zeros_like(mask_binary)
                mask_in_box[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]
                
                result.masks.append(mask_in_box)
                
            except Exception as e:
                print(f"掩码生成失败 (第{i}个目标): {e}")
                # 创建一个空掩码作为fallback
                empty_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                result.masks.append(empty_mask)
    
    def draw_segmentation_results(self, image: np.ndarray, result: TaskResult) -> np.ndarray:
        """绘制分割结果（包含检测框和掩码）"""
        result_image = image.copy()
        
        # 为每个类别生成固定颜色（确保颜色不会太暗）
        colors = self._get_bright_colors(max(80, len(self.class_names)))
        
        # 绘制掩码（先绘制掩码，再绘制框，这样框在上层）
        if result.masks:
            for i, (mask, class_id) in enumerate(zip(result.masks, result.class_ids)):
                if isinstance(mask, np.ndarray) and mask.sum() > 0:
                    # 获取类别对应的颜色
                    color = colors[class_id % len(colors)]
                    
                    # 创建彩色掩码
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[mask > 0] = color
                    
                    # 与原图混合（半透明效果）
                    alpha = 0.4
                    mask_area = mask > 0
                    result_image[mask_area] = cv2.addWeighted(
                        image[mask_area], 1-alpha, 
                        colored_mask[mask_area], alpha, 0
                    )
                    
                    # 绘制掩码轮廓
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(result_image, contours, -1, color, 2)
        
        # 绘制检测框和标签
        if result.boxes:
            for i, (box, score, class_id, class_name) in enumerate(
                zip(result.boxes, result.scores, result.class_ids, result.class_names)
            ):
                x1, y1, x2, y2 = map(int, box)
                
                # 获取类别对应的颜色
                color = colors[class_id % len(colors)]
                
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
    
    def _get_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """生成固定的颜色列表"""
        np.random.seed(42)  # 固定随机种子，确保颜色一致
        colors = []
        for i in range(num_classes):
            colors.append(tuple(np.random.randint(0, 255, 3).tolist()))
        return colors
    
    def _get_bright_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """生成明亮的固定颜色列表，避免黑色或太暗的颜色"""
        np.random.seed(42)  # 固定随机种子，确保颜色一致
        colors = []
        for i in range(num_classes):
            # 确保每个颜色通道至少有一个值较高，避免全黑
            while True:
                color = tuple(np.random.randint(64, 255, 3).tolist())  # 最小值从64开始
                # 确保颜色的亮度足够（至少有一个通道 > 128）
                if max(color) > 128:
                    colors.append(color)
                    break
        return colors
    
    def draw_results(self, image: np.ndarray, result: TaskResult) -> np.ndarray:
        """重写基类方法，使用自定义的分割绘制"""
        return self.draw_segmentation_results(image, result)
    


# 便捷工厂函数
def create_segmentor(model_path: str, **kwargs) -> SegmentationInference:
    """创建分割器"""
    return SegmentationInference(model_path, **kwargs)
