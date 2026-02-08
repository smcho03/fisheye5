import os
from PIL import Image, ImageDraw, ImageFont

# 경로 설정 (중복 경로 수정)
base_path = "output/6ef16bfa-2"
output_dir = "grid_results"
os.makedirs(output_dir, exist_ok=True)

# 설정값 조정
iters = list(range(1000, 31000, 1000))
rows, cols = 2, 3
images_per_grid = rows * cols

# --- 수정한 부분 ---
font_size = 60          # 폰트 크기를 25에서 60으로 확대
caption_height = 80     # 텍스트 영역 높이를 40에서 80으로 확대
# ------------------

for grid_idx in range(5):
    sample_path = os.path.join(base_path, f"debug_iter_{iters[0]}", "fisheye_comparison.png")
    if not os.path.exists(sample_path):
        print(f"파일을 찾을 수 없습니다: {sample_path}")
        continue

    with Image.open(sample_path) as img:
        w, h = img.size
        unit_w, unit_h = w // 2, h
        
    # 그리드 배경 생성
    grid_img = Image.new('RGB', (unit_w * cols, (unit_h + caption_height) * rows), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    
    # 폰트 설정
    try:
        # 폰트 경로 (시스템에 따라 조정 필요)
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for i in range(images_per_grid):
        idx = grid_idx * images_per_grid + i
        current_iter = iters[idx]
        img_path = os.path.join(base_path, f"debug_iter_{current_iter}", "fisheye_comparison.png")
        
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                left_img = img.crop((0, 0, unit_w, unit_h))
                
                r, c = divmod(i, cols)
                x_offset = c * unit_w
                y_offset = r * (unit_h + caption_height)
                
                grid_img.paste(left_img, (x_offset, y_offset))
                
                # 캡션 추가 (가운데 정렬)
                text = f"Iter: {current_iter}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                
                # 텍스트 위치 계산 (이미지 바로 아래 중앙)
                text_x = x_offset + (unit_w - text_w) // 2
                text_y = y_offset + unit_h + (caption_height - text_h) // 2 - 5
                
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

    save_path = os.path.join(output_dir, f"grid_{grid_idx + 1}.png")
    grid_img.save(save_path)
    print(f"저장 완료: {save_path}")