import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from base_onnx import BaseONNX, ModelType, TaskResult


class Segment(BaseONNX):
    """分割推理器"""
    
    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        """
        初始化分割推理器
        
        Args:
            model_path: ONNX模型路径
            device: 设备类型 ("auto", "cuda", "cpu")
            simplify_contours: 是否简化轮廓
            downsample_contours: 是否对轮廓点进行下采样
            **kwargs: 其他参数
        """        
        super().__init__(model_path, ModelType.SEGMENTATION, device, **kwargs)
    

    
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
                
                # 处理掩码输出 - 生成轮廓点坐标（类似YOLO mask.xy格式）
                if len(outputs) > 1:
                    selected_mask_coeffs = [mask_coeffs[i] for i in indices.flatten()]
                    proto_masks = outputs[1]  # 掩码原型 [mask_dim, mask_h, mask_w]
                    
                    # 移除批次维度（如果存在）
                    if len(proto_masks.shape) == 4:
                        proto_masks = proto_masks[0]
                    
                    # 为每个检测到的目标生成轮廓点
                    for i, (mask_coeff, box) in enumerate(zip(selected_mask_coeffs, result.boxes)):
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
                            
                            # 应用阈值得到二值掩码
                            mask_binary = (mask_final > 0.5).astype(np.uint8)
                            
                            # 可选：将掩码限制在检测框内
                            x1, y1, x2, y2 = box
                            mask_in_box = np.zeros_like(mask_binary)
                            mask_in_box[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]
                            
                            # 提取轮廓点坐标，使用更精细的方法
                            contours, _ = cv2.findContours(mask_in_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 使用APPROX_NONE保留所有点
                            
                            # 如果找到轮廓，选择最大的轮廓作为主轮廓
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                
                                # 将轮廓点转换为整数坐标，直接使用int类型避免科学计数法
                                points = largest_contour.reshape(-1, 2)
                                # 转换为整数类型，避免科学计数法显示
                                points = points.astype(np.int32)
                                
                                if len(points) >= 3:  # 至少需要3个点形成有效轮廓
                                    result.masks.append(points)
                                else:
                                    result.masks.append(np.array([], dtype=np.int32).reshape(0, 2))
                            else:
                                result.masks.append(np.array([], dtype=np.int32).reshape(0, 2))
                            
                        except Exception as e:
                            print(f"轮廓点提取失败 (第{i}个目标): {e}")
                            # 创建空的numpy数组作为fallback
                            result.masks.append(np.array([], dtype=np.int32).reshape(0, 2))
        
        return result
    

