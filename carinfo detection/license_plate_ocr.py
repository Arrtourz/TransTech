import cv2
import easyocr
import numpy as np
import glob
import os

class TexasPlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        
    def preprocess_image(self, img):
        """使用最优方法预处理图像"""
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 增大对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 增大图像尺寸
        scale_factor = 2
        enlarged = cv2.resize(enhanced, None, fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_CUBIC)
        
        # 使用最佳阈值进行二值化
        _, binary = cv2.threshold(enlarged, 120, 255, cv2.THRESH_BINARY)
        
        # 保存预处理结果
        cv2.imwrite('preprocessed.jpg', binary)
        
        return binary
    
    def format_plate_number(self, text):
        """格式化车牌号码"""
        # 移除所有空格和特殊字符
        text = ''.join(c for c in text if c.isalnum())
        
        # 转换为大写
        text = text.upper()
        
        # 如果长度为7，添加破折号
        if len(text) == 7:
            text = text[:3] + '-' + text[3:]
            
        return text
    
    def read_plate(self, img_path):
        """读取车牌图片并识别文字"""
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
            
        # 预处理图像
        processed_img = self.preprocess_image(img)
        
        # OCR识别
        results = self.reader.readtext(processed_img,
                                     allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                     paragraph=False,
                                     width_ths=1.0,
                                     height_ths=1.0)
        
        # 处理识别结果
        best_result = (None, 0)
        for box, text, confidence in results:
            # 格式化识别的文本
            formatted_text = self.format_plate_number(text)
            print(f"检测到: {formatted_text}, 置信度: {confidence}")
            
            # 更新最佳结果
            if confidence > best_result[1] and len(formatted_text) >= 7:
                best_result = (formatted_text, confidence)
        
        return best_result
    
def main():
    try:
        # 初始化识别器
        ocr = TexasPlateOCR()
        
        # 获取所有匹配的文件
        image_files = glob.glob('output/plate_roi_*.jpg')
        
        for img_path in image_files:
            print(f"处理文件: {img_path}")
            plate_text, confidence = ocr.read_plate(img_path)
            
            if plate_text:
                print("\n最终识别结果:")
                print(f"车牌号: {plate_text}")
                print(f"置信度: {confidence:.3f}")
                
                # 在原图上标注结果
                img = cv2.imread(img_path)
                height, width = img.shape[:2]
                # 添加文本背景
                cv2.rectangle(img, (0, 0), (width, 30), (0, 0, 0), -1)
                # 添加识别结果文本
                cv2.putText(img, f"{plate_text} ({confidence:.3f})", 
                           (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                result_path = os.path.join('output', f'result_{os.path.basename(img_path)}')
                cv2.imwrite(result_path, img)
                print(f"\n结果已保存至 {result_path}")
            else:
                print("未能识别出车牌号码")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()