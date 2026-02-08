import os
from PIL import Image, ImageDraw, ImageFont

# 1. 경로 설정 (현재 위치에 debug_fixed_view 폴더가 있다고 가정)
base_path = "output/f1dd7e9a-4/debug_fixed_view"
output_dir = "grid_results_fixed_view_117"
os.makedirs(output_dir, exist_ok=True)

# 2. 설정값 (30개 파일: 1000, 2000, ..., 30000)
# 파일명이 iter_001000.png 형식이므로 zfill(6) 사용
iter_numbers = list(range(1000, 31000, 1000))
rows, cols = 2, 3
images_per_grid = rows * cols

# 가독성을 위한 설정 (이전 피드백 반영: 크게 설정)
font_size = 80          # 폰트 크기 대폭 확대
caption_height = 100    # 텍스트 영역 높이 확대

# 3. 작업 수행 (총 5개 그리드 생성)
for grid_idx in range(5):
    # 첫 번째 유효한 이미지를 찾아 규격 확인
    sample_name = f"iter_{iter_numbers[0]:06d}.png"
    sample_path = os.path.join(base_path, sample_name)
    
    if not os.path.exists(sample_path):
        print(f"파일을 찾을 수 없어 건너뜁니다: {sample_path}")
        continue

    with Image.open(sample_path) as img:
        w, h = img.size
        unit_w, unit_h = w // 2, h  # 왼쪽 절반 크롭용
        
    # 그리드 캔버스 생성 (흰색 배경)
    grid_img = Image.new('RGB', (unit_w * cols, (unit_h + caption_height) * rows), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    
    # 폰트 로드
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 해당 그리드에 들어갈 6개의 이미지 처리
    for i in range(images_per_grid):
        idx = grid_idx * images_per_grid + i
        current_iter = iter_numbers[idx]
        file_name = f"iter_{current_iter:06d}.png"  # 001000 형식
        img_path = os.path.join(base_path, file_name)
        
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                # 왼쪽 이미지만 크롭
                left_img = img.crop((0, 0, unit_w, unit_h))
                
                # 배치 좌표
                r, c = divmod(i, cols)
                x_offset = c * unit_w
                y_offset = r * (unit_h + caption_height)
                
                # 이미지 붙이기
                grid_img.paste(left_img, (x_offset, y_offset))
                
                # 캡션 추가 (예: 1000, 2000...)
                text = str(current_iter)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # 텍스트 중앙 정렬 위치 계산
                text_x = x_offset + (unit_w - text_w) // 2
                text_y = y_offset + unit_h + (caption_height - text_h) // 2 - 10
                
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        else:
            print(f"경고: {file_name} 파일이 존재하지 않습니다.")

    # 결과 저장
    save_path = os.path.join(output_dir, f"fixed_view_grid_{grid_idx + 1}.png")
    grid_img.save(save_path)
    print(f"저장 완료: {save_path}")

print("\n모든 작업이 완료되었습니다.")