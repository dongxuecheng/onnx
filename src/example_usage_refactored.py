"""
重构后的ONNX推理器使用示例
展示如何使用四个不同的推理器类
"""
import cv2
import numpy as np
import os
import time
from detection_inference import create_detector
from classification_inference import create_classifier
from segmentation_inference import create_segmentor
from pose_inference import create_pose_estimator
from base_onnx_inference import RTSPProcessor


def test_detection():
    """测试目标检测"""
    print("=" * 50)
    print("测试目标检测")
    print("=" * 50)
    
    try:
        # 创建检测器
        detector = create_detector(
            model_path="model/yolo11n.onnx",
            device="auto",
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        # 测试单张图片
        if os.path.exists("images/human.jpg"):
            image = cv2.imread("images/human.jpg")
            
            start_time = time.time()
            result = detector.predict(image)
            inference_time = time.time() - start_time
            
            print(f"推理时间: {inference_time*1000:.2f}ms")
            print(f"检测到 {len(result.boxes)} 个目标")
            
            # 打印详细结果
            for i, (box, score, class_name) in enumerate(zip(result.boxes, result.scores, result.class_names)):
                print(f"目标 {i+1}: {class_name} - {score:.3f} - {box}")
            
            # 绘制结果
            result_image = detector.draw_results(image, result)
            cv2.imwrite("result/detection_result.jpg", result_image)
            print("结果已保存到 detection_result.jpg")
        else:
            print("未找到测试图片 human.jpg")
            
    except Exception as e:
        print(f"检测测试失败: {e}")


def test_classification():
    """测试分类"""
    print("=" * 50)
    print("测试分类")
    print("=" * 50)
    
    try:
        # 检查分类模型是否存在
        model_path = "yolo11n-cls.onnx"
        if os.path.exists(model_path):
            print(f"找到分类模型: {model_path}")
            
            # 创建分类器
            classifier = create_classifier(
                model_path=model_path,
                device="auto"
            )
            
            print(f"分类器创建成功")
            print(f"模型输入尺寸: {classifier.input_shape}")
            print(f"使用设备: {classifier.device}")
            print(f"类别数量: {len(classifier.class_names)}")
            
            # 检查测试图片
            test_image = "human.jpg"
            if os.path.exists(test_image):
                print(f"使用测试图片: {test_image}")
                
                # 加载图片
                image = cv2.imread(test_image)
                if image is None:
                    print("无法加载图片")
                    return
                    
                print(f"图片尺寸: {image.shape}")
                
                # 执行推理
                start_time = time.time()
                result = classifier.predict(image)
                inference_time = time.time() - start_time
                
                print(f"推理时间: {inference_time*1000:.2f}ms")
                print(f"Top-{len(result.scores)} 分类结果:")
                
                # 打印详细结果 (按置信度排序)
                for i, (class_name, score) in enumerate(zip(result.class_names, result.scores)):
                    print(f"  {i+1}. {class_name}: {score:.4f}")
                
                # 打印最高置信度的预测
                if len(result.scores) > 0:
                    top_class = result.class_names[0]
                    top_score = result.scores[0]
                    print(f"\n最可能的类别: {top_class} (置信度: {top_score:.4f})")
                
                # 绘制分类结果
                result_image = classifier.draw_results(image, result)
                cv2.imwrite("classification_result.jpg", result_image)
                print("分类结果已保存到 classification_result.jpg")
                
            else:
                print(f"未找到测试图片 {test_image}")
                print("请确保有一张图片用于分类测试")
        else:
            print(f"未找到分类模型 {model_path}")
            print("请下载或准备一个YOLO格式的分类模型")
            print("\n分类器使用示例：")
            print("classifier = create_classifier('yolo11n-cls.onnx')")
            print("result = classifier.predict(image)")
            print("top5_results = [(name, score) for name, score in zip(result.class_names, result.scores)]")
            
    except Exception as e:
        print(f"分类测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_segmentation():
    """测试分割"""
    print("=" * 50)
    print("测试分割")
    print("=" * 50)
    
    try:
        # 检查分割模型是否存在
        model_path = "yolo11l-seg.onnx"
        if os.path.exists(model_path):
            print(f"找到分割模型: {model_path}")
            
            # 创建分割器
            segmentor = create_segmentor(
                model_path=model_path,
                device="auto",
                conf_threshold=0.25,
                iou_threshold=0.45
            )
            
            print(f"分割器创建成功")
            print(f"模型输入尺寸: {segmentor.input_shape}")
            print(f"使用设备: {segmentor.device}")
            
            # 检查测试图片
            test_image = "human.jpg"
            if os.path.exists(test_image):
                print(f"使用测试图片: {test_image}")
                
                # 加载图片
                image = cv2.imread(test_image)
                if image is None:
                    print("无法加载图片")
                    return
                    
                print(f"图片尺寸: {image.shape}")
                
                # 执行推理
                start_time = time.time()
                result = segmentor.predict(image)
                inference_time = time.time() - start_time
                
                print(f"推理时间: {inference_time*1000:.2f}ms")
                print(f"检测到 {len(result.boxes)} 个目标")
                print(f"生成的掩码数量: {len(result.masks)}")
                
                # 掩码系数信息
                if 'mask_coeffs' in result.extra_data:
                    print(f"掩码系数数量: {len(result.extra_data['mask_coeffs'])}")
                
                # 打印详细结果
                for i, (box, score, class_name) in enumerate(zip(result.boxes, result.scores, result.class_names)):
                    print(f"目标 {i+1}: {class_name} - 置信度: {score:.3f}")
                    print(f"  边界框: {box}")
                    if i < len(result.masks):
                        mask_area = result.masks[i].sum()
                        print(f"  掩码像素数: {mask_area}")
                
                # 绘制分割结果（包含掩码）
                result_image = segmentor.draw_results(image, result)
                cv2.imwrite("segmentation_result.jpg", result_image)
                print("分割结果已保存到 segmentation_result.jpg")
                
                # 创建所有掩码的组合图像
                # if result.masks:
                #     # 方法1: 创建彩色组合掩码
                #     combined_mask_image = segmentor.create_combined_mask_image(image, result)
                #     cv2.imwrite("combined_masks.jpg", combined_mask_image)
                #     print("组合掩码图像已保存到 combined_masks.jpg")
                    
                    # # 方法2: 创建所有掩码叠加效果
                    # overlay_image = segmentor.create_all_masks_overlay(image, result)
                    # cv2.imwrite("all_masks_overlay.jpg", overlay_image)
                    # print("所有掩码叠加图像已保存到 all_masks_overlay.jpg")
                    
                    # # 方法3: 创建纯掩码图像（所有掩码在一张图上，不同灰度值）
                    # all_masks_combined = np.zeros_like(image[:,:,0])
                    # for i, mask in enumerate(result.masks):
                    #     # 每个掩码使用不同的灰度值
                    #     mask_value = int(255 * (i + 1) / len(result.masks))
                    #     all_masks_combined[mask > 0] = mask_value
                    # cv2.imwrite("all_masks_pure.jpg", all_masks_combined)
                    # print("纯掩码组合图像已保存到 all_masks_pure.jpg")
                    
                    # # 方法4: 创建彩色的纯掩码图像
                    # all_masks_colored = np.zeros_like(image)
                    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    # for i, mask in enumerate(result.masks):
                    #     color = colors[i % len(colors)]
                    #     all_masks_colored[mask > 0] = color
                    # cv2.imwrite("all_masks_colored.jpg", all_masks_colored)
                    # print("彩色掩码组合图像已保存到 all_masks_colored.jpg")
                    
                    # # 保存单独的掩码图像（可选）
                    # for i, mask in enumerate(result.masks):
                    #     mask_image = mask * 255  # 转换为0-255范围
                    #     cv2.imwrite(f"mask_{i}.jpg", mask_image)
                    # print(f"单独的掩码图像已保存: mask_0.jpg ~ mask_{len(result.masks)-1}.jpg")

                
            else:
                print(f"未找到测试图片 {test_image}")
                print("请确保有一张图片用于分割测试")
        else:
            print(f"未找到分割模型 {model_path}")
            print("请下载或准备一个YOLO格式的分割模型")
            print("\n分割器使用示例：")
            print("segmentor = create_segmentor('yolo11l-seg.onnx')")
            print("result = segmentor.predict(image)")
            print("segmented_image = segmentor.draw_results(image, result)")
            
    except Exception as e:
        print(f"分割测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_pose():
    """测试姿态估计"""
    print("=" * 50)
    print("测试姿态估计")
    print("=" * 50)
    
    try:
        # 检查姿态估计模型是否存在
        model_path = "../yolo11l-pose.onnx"
        if os.path.exists(model_path):
            print(f"找到姿态估计模型: {model_path}")
            
            # 创建姿态估计器
            pose_estimator = create_pose_estimator(
                model_path=model_path,
                device="auto",
                conf_threshold=0.25
            )
            
            print(f"姿态估计器创建成功")
            print(f"模型输入尺寸: {pose_estimator.input_shape}")
            print(f"使用设备: {pose_estimator.device}")
            
            # 检查测试图片
            test_image = "../human.jpg"
            if os.path.exists(test_image):
                print(f"使用测试图片: {test_image}")
                
                # 加载图片
                image = cv2.imread(test_image)
                if image is None:
                    print("无法加载图片")
                    return
                    
                print(f"图片尺寸: {image.shape}")
                
                # 执行推理
                start_time = time.time()
                result = pose_estimator.predict(image)
                inference_time = time.time() - start_time
                
                print(f"推理时间: {inference_time*1000:.2f}ms")
                print(f"检测到 {len(result.keypoints)} 个人")
                
                # 打印详细结果
                for i, (box, score, class_name, keypoint) in enumerate(
                    zip(result.boxes, result.scores, result.class_names, result.keypoints)
                ):
                    print(f"人物 {i+1}: {class_name} - 置信度: {score:.3f}")
                    print(f"边界框: {box}")
                    print(f"关键点数量: {len(keypoint) // 3}")  # 每个关键点3个值(x,y,confidence)
                    
                    # 统计可见关键点
                    visible_points = sum(1 for j in range(2, len(keypoint), 3) if keypoint[j] > 0.5)
                    print(f"可见关键点: {visible_points}/17")
                
                # 绘制姿态结果
                result_image = pose_estimator.draw_pose_results(image, result)
                cv2.imwrite("pose_result.jpg", result_image)
                print("姿态结果已保存到 pose_result.jpg")
                
                # 获取姿态分析
                if len(result.keypoints) > 0:
                    analysis = pose_estimator.get_pose_analysis(result)
                    print("\n姿态分析:")
                    for person_idx, person_analysis in enumerate(analysis):
                        print(f"人物 {person_idx + 1}:")
                        print(f"  头部角度: {person_analysis.get('head_angle', 'N/A')}")
                        print(f"  身体姿态: {person_analysis.get('body_posture', 'N/A')}")
                        print(f"  动作状态: {person_analysis.get('action_state', 'N/A')}")
                        print(f"  关键点完整性: {person_analysis.get('keypoint_completeness', 'N/A')}")
                
            else:
                print(f"未找到测试图片 {test_image}")
                print("请确保有一张包含人物的图片用于测试")
        else:
            print(f"未找到姿态估计模型 {model_path}")
            print("请下载或准备一个YOLO格式的姿态估计模型")
            
    except Exception as e:
        print(f"姿态估计测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_rtsp_detection():
    """测试RTSP检测"""
    print("=" * 50)
    print("测试RTSP检测")
    print("=" * 50)
    
    try:
        # 创建检测器
        detector = create_detector(
            model_path="yolo11n.onnx",
            device="auto",
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        # 创建RTSP处理器
        rtsp_processor = RTSPProcessor(detector)
        
        # 这里只是示例，实际使用时需要真实的RTSP地址
        print("RTSP处理器创建成功")
        print("使用示例：")
        print("rtsp_processor.process_stream(")
        print("    rtsp_url='rtsp://your_camera_ip:port/stream',")
        print("    max_frames=1000,")
        print("    display=True,")
        print("    save_results=False")
        print(")")
        
    except Exception as e:
        print(f"RTSP测试失败: {e}")


def compare_models():
    """比较不同模型的性能"""
    print("=" * 50)
    print("模型性能比较")
    print("=" * 50)
    
    models = []
    
    # 检测模型
    if os.path.exists("yolo11n.onnx"):
        models.append(("检测模型", "yolo11n.onnx", create_detector))
    
    # 分割模型
    if os.path.exists("yolo11l-seg.onnx"):
        models.append(("分割模型", "yolo11l-seg.onnx", create_segmentor))
    
    if not models:
        print("未找到可用的模型文件")
        return
    
    if os.path.exists("human.jpg"):
        image = cv2.imread("human.jpg")
        
        for model_name, model_path, creator_func in models:
            try:
                print(f"\n测试 {model_name}:")
                model = creator_func(model_path, device="auto")
                
                # 预热
                for _ in range(3):
                    model.predict(image)
                
                # 性能测试
                times = []
                for _ in range(10):
                    start_time = time.time()
                    result = model.predict(image)
                    inference_time = time.time() - start_time
                    times.append(inference_time * 1000)
                
                avg_time = sum(times) / len(times)
                fps = 1000 / avg_time if avg_time > 0 else 0
                
                print(f"  平均推理时间: {avg_time:.2f}ms")
                print(f"  理论FPS: {fps:.2f}")
                print(f"  检测目标数: {len(result.boxes)}")
                
            except Exception as e:
                print(f"  {model_name} 测试失败: {e}")
    else:
        print("未找到测试图片 human.jpg")


if __name__ == "__main__":
    print("ONNX推理器重构版本测试")
    print("当前工作目录:", os.getcwd())
    
    # 运行各种测试
    test_detection()
    #test_classification()
    #test_segmentation()
    # test_pose()
    # test_rtsp_detection()
    # compare_models()
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
