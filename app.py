import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import platform

# --- 0. 기본 설정 (파일명 지정) ---
# 같은 폴더에 이 파일들이 있어야 합니다.
IMAGE_FILE = 'images.png'
CSV_FILE = 'cancer.csv'
FONT_FILE = None  # 폰트 파일명 (예: 'NanumSquareB.ttf'). 없으면 None

# 페이지 설정
st.set_page_config(page_title="Cancer Visualization", layout="centered")
st.title("🩸 암 발병률 시각화")

# --- 1. 영상 생성 함수 (기존 로직과 동일) ---
# 캐싱(@st.cache_data)을 사용하여 새로고침해도 매번 다시 만들지 않고 빠르게 로딩되게 함
@st.cache_data(show_spinner=True)
def create_video_auto():
    # 파일 존재 확인
    if not os.path.exists(IMAGE_FILE) or not os.path.exists(CSV_FILE):
        return None, f"⚠️ 필수 파일이 없습니다. 폴더에 '{IMAGE_FILE}'와 '{CSV_FILE}'를 넣어주세요."

    # 데이터 로드
    try:
        df = pd.read_csv(CSV_FILE)
        df_filtered = df[(df['성별'] == '남녀전체') & (df['암종'] == '모든암') & (df['연령군'] == '연령전체')]
        df_filtered = df_filtered[df_filtered['발생연도'].astype(str).str.len() == 4].sort_values('발생연도')
        years_data = df_filtered['발생연도'].astype(int).tolist()
        rates_data = df_filtered['조발생률'].tolist()
    except Exception as e:
        return None, f"데이터 처리 오류: {e}"

    min_rate_baseline = rates_data[0]
    max_rate_peak = max(rates_data)
    if max_rate_peak == min_rate_baseline: max_rate_peak += 1 

    # 이미지 처리
    img_cv = cv2.imread(IMAGE_FILE)
    img_h, img_w = img_cv.shape[:2]
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    _, original_line_mask = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thinner_line_mask = cv2.erode(original_line_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(original_line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "이미지에서 실루엣을 찾을 수 없습니다."
    
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h_sil = cv2.boundingRect(main_contour)

    # 캔버스 설정
    canvas_w = int(img_w * 1.5)
    canvas_h = int(img_h * 1.5) 
    offset_x = (canvas_w - img_w) // 2
    offset_y = int(img_h * 0.25)

    canv_y_top = y + offset_y
    canv_y_bottom = y + h_sil + offset_y
    canv_feet_y = canv_y_bottom
    canv_x_center = (x + offset_x) + w // 2

    canv_body_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    shifted_contour = main_contour + [offset_x, offset_y]
    cv2.drawContours(canv_body_mask, [shifted_contour], -1, 255, -1)
    
    canv_line_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canv_line_mask[offset_y:offset_y+img_h, offset_x:offset_x+img_w] = thinner_line_mask

    # 비디오 설정
    fps = 48
    frames_per_month = 3
    
    # 임시 파일 생성
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = tfile.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    # 폰트 설정 (자동 감지)
    try:
        if FONT_FILE and os.path.exists(FONT_FILE):
            font_path = FONT_FILE
        elif platform.system() == 'Darwin': font_path = "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc"
        elif platform.system() == 'Windows': font_path = "C:/Windows/Fonts/malgun.ttf"
        else: font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        
        font_large = ImageFont.truetype(font_path, 60)
        font_num = ImageFont.truetype(font_path, 65)
        font_text = ImageFont.truetype(font_path, 50)
    except:
        font_large = font_num = font_text = ImageFont.load_default()

    # 프레임 생성 루프
    for i in range(len(years_data)):
        target_year = years_data[i]
        start_rate_segment = rates_data[i]
        end_rate_segment = rates_data[i+1] if i < len(years_data) - 1 else rates_data[i]
        
        for month in range(1, 13):
            for f in range(frames_per_month):
                total_steps = 12 * frames_per_month
                current_step = (month - 1) * frames_per_month + f
                alpha = current_step / total_steps
                
                interpolated_rate = start_rate_segment + (end_rate_segment - start_rate_segment) * alpha
                percentage_text = interpolated_rate / 1000 
                
                height_calc_rate = max(min_rate_baseline, interpolated_rate)
                fill_ratio = (height_calc_rate - min_rate_baseline) / (max_rate_peak - min_rate_baseline)
                
                frame = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
                
                fill_height_px = int(fill_ratio * h_sil)
                curr_y_fill = canv_y_bottom - fill_height_px
                liquid_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                liquid_mask[max(canv_y_top, curr_y_fill):canv_y_bottom, :] = 255
                final_fill_area = cv2.bitwise_and(canv_body_mask, liquid_mask)
                frame[final_fill_area > 0] = [70, 70, 230] 
                frame[canv_line_mask > 0] = [0, 0, 0]
                
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                
                head_text = f"{target_year}년 {month}월에는"
                bbox_head = draw.textbbox((0, 0), head_text, font=font_large)
                head_w = bbox_head[2] - bbox_head[0]
                head_h = bbox_head[3] - bbox_head[1]
                head_text_x = canv_x_center - head_w // 2
                head_text_y = canv_y_top - head_h - 30
                draw.text((head_text_x, head_text_y), head_text, fill=(0, 0, 0), font=font_large)
                
                num_str = f"{percentage_text:.2f}%"
                rest_str = "의 사람이 암에 걸렸습니다"
                
                bbox_num = draw.textbbox((0, 0), num_str, font=font_num)
                num_w = bbox_num[2] - bbox_num[0]
                bbox_rest = draw.textbbox((0, 0), rest_str, font=font_text)
                rest_w = bbox_rest[2] - bbox_rest[0]
                
                total_w = num_w + rest_w
                start_x = (canvas_w - total_w) // 2
                base_y = canv_feet_y + 40
                
                draw.text((start_x, base_y), num_str, fill=(200, 0, 0), font=font_num, stroke_width=2, stroke_fill=(200, 0, 0))
                text_y_adjust = base_y + (65 - 50) // 2 + 5 
                draw.text((start_x + num_w, text_y_adjust), rest_str, fill=(50, 50, 50), font=font_text)
                
                out.write(cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR))

    out.release()
    return output_path, None

# --- 2. 메인 실행 ---
# 자동으로 함수 실행하여 결과 받아옴
video_path, error_msg = create_video_auto()

if error_msg:
    st.error(error_msg)
else:
    # 영상 재생
    st.video(video_path)
    
    # 다운로드 버튼 제공
    with open(video_path, 'rb') as f:
        st.download_button("📥 영상 다운로드", f, "cancer_viz.mp4", "video/mp4")