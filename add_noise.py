import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置路径
input_folder = r'C:\Users\lwt\Desktop\exercise\iVPGSA\dataset\COULE_test'
output_folder = r'C:\Users\lwt\Desktop\exercise\iVPGSA\dataset\blurred_COULE_test'
comparison_folder = r'C:\Users\lwt\Desktop\exercise\iVPGSA\dataset\comparison'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(comparison_folder, exist_ok=True)
variance = 0.09

def add_noise_with_alpha(image_path, output_path):
    """处理带透明度的图片"""
    try:
        # 读取图片
        pil_img = Image.open(image_path)
        
        if pil_img.mode == 'RGBA':
            # 分离RGB和Alpha通道
            rgb_img = pil_img.convert('RGB')
            alpha_channel = pil_img.split()[-1]
            
            # 处理RGB部分
            rgb_array = np.array(rgb_img)
            rgb_float = rgb_array.astype(np.float64) / 255.0
            
            # 只对RGB通道添加噪声，保持Alpha通道不变
            noise = np.sqrt(variance) * np.random.randn(*rgb_float.shape)
            noisy_rgb = np.clip(rgb_float + noise, 0, 1)
            noisy_rgb_uint8 = (noisy_rgb * 255).astype(np.uint8)
            
            # 重新组合为RGBA
            noisy_rgb_img = Image.fromarray(noisy_rgb_uint8)
            noisy_rgba_img = Image.merge('RGBA', (*noisy_rgb_img.split(), alpha_channel))
            
            # 保存为PNG格式以保持透明度
            output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_noisy.png'
            output_filepath = os.path.join(output_path, output_filename)
            noisy_rgba_img.save(output_filepath, 'PNG')
            
            return True, output_filepath, pil_img, noisy_rgba_img
            
        else:
            # 处理普通图片
            img_array = np.array(pil_img)
            if len(img_array.shape) == 2:  # 灰度图
                img_array = np.stack([img_array, img_array, img_array], axis=2)
            
            img_float = img_array.astype(np.float64) / 255.0
            noise = np.sqrt(variance) * np.random.randn(*img_float.shape)
            noisy_img = np.clip(img_float + noise, 0, 1)
            noisy_img_uint8 = (noisy_img * 255).astype(np.uint8)
            
            output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_noisy.jpg'
            output_filepath = os.path.join(output_path, output_filename)
            Image.fromarray(noisy_img_uint8).save(output_filepath, 'JPEG')
            
            noisy_pil_img = Image.fromarray(noisy_img_uint8)
            return True, output_filepath, pil_img, noisy_pil_img
        
    except Exception as e:
        print(f"处理失败: {os.path.basename(image_path)} - {e}")
        return False, None, None, None

def create_simple_comparison(original_img, noisy_img, output_path, filename):
    """创建简单的对比图（解决中文显示问题）"""
    try:
        # 设置画布大小
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 显示原图
        ax1.imshow(np.array(original_img))
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 显示噪声图
        ax2.imshow(np.array(noisy_img))
        ax2.set_title('Noisy Image(gauss 0.09)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 添加整体标题（使用英文避免字体问题）
        plt.suptitle(f'Image Comparison: {os.path.splitext(filename)[0]}', 
                     fontsize=16, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存对比图
        comparison_filename = f"comparison_{os.path.splitext(filename)[0]}.png"
        comparison_filepath = os.path.join(output_path, comparison_filename)
        plt.savefig(comparison_filepath, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"已创建对比图: {comparison_filename}")
        return True
        
    except Exception as e:
        print(f"创建对比图失败: {filename} - {e}")
        return False

def create_comparison_with_borders(original_img, noisy_img, output_path, filename):
    """创建带边框的对比图"""
    try:
        # 设置画布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原图
        ax1.imshow(np.array(original_img))
        ax1.set_title('Original', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # 添加边框
        for spine in ax1.spines.values():
            spine.set_edgecolor('#2E86AB')
            spine.set_linewidth(2)
        
        # 噪声图
        ax2.imshow(np.array(noisy_img))
        ax2.set_title('Noisy', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        # 添加边框
        for spine in ax2.spines.values():
            spine.set_edgecolor('#A23B72')
            spine.set_linewidth(2)
        
        # 整体标题
        plt.suptitle(f'Image Noise Comparison: {os.path.splitext(filename)[0]}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存对比图
        comparison_filename = f"comparison_{os.path.splitext(filename)[0]}.png"
        comparison_filepath = os.path.join(output_path, comparison_filename)
        plt.savefig(comparison_filepath, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"已创建对比图: {comparison_filename}")
        return True
        
    except Exception as e:
        print(f"创建对比图失败: {filename} - {e}")
        return False

# 批量处理
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"找到 {len(image_files)} 张图片")
print("="*50)

success_count = 0
comparison_count = 0

for i, filename in enumerate(image_files, 1):
    input_path = os.path.join(input_folder, filename)
    print(f"\n[{i:02d}/{len(image_files)}] 处理: {filename}")
    
    # 处理图片并添加噪声
    success, noisy_path, original_img, noisy_img = add_noise_with_alpha(input_path, output_folder)
    
    if success:
        success_count += 1
        print(f"   噪声图片生成成功")
        
        # 创建对比图
        if original_img is not None and noisy_img is not None:
            # 使用简单的对比图函数（避免中文问题）
            if create_simple_comparison(original_img, noisy_img, comparison_folder, filename):
                comparison_count += 1
                print(f"   对比图生成成功")
            else:
                print(f"   对比图生成失败")
    else:
        print(f"   噪声图片生成失败")

print(f"\n" + "="*50)
print(f"处理完成!")
print(f"原图片数量: {len(image_files)} 张")
print(f"噪声图片: {success_count} 张")
print(f"对比图片: {comparison_count} 张")
print(f"噪声图片保存在: {output_folder}")
print(f"对比图片保存在: {comparison_folder}")
print("="*50)