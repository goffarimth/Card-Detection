import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
import streamlit.components.v1 as components

# โหลดโมเดล YOLOv8 ที่เทรนไว้
model = YOLO("C:\\Users\\goffa\\Documents\\ML\\train5\\best.pt")
model.to('cpu')

# เพิ่ม CSS สำหรับตกแต่งพื้นหลังด้วยอีโมจิไพ่เคลื่อนไหว และจัดกึ่งกลางหัวข้อ
st.markdown(
    """
    <style>
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .emoji {
        font-size: 50px;
        position: absolute;
        animation: float 3s infinite;
    }
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 10vh;
    }
    .title {
        font-size: 70px;
        font-weight: bold;
    }
    </style>
    <div class="emoji" style="top: 20%; left: 6%;">♥️</div>
    <div class="emoji" style="top: 20%; left: -2%;">♠️</div>
    <div class="emoji" style="top: 20%; left: 90%;">♦️</div>
    <div class="emoji" style="top: 20%; left: 81%;">♣️</div>
    <div class="title-container">
        <h1 class="title">Card Detection</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ตัวเลือกสำหรับการอัปโหลดไฟล์หรือเปิดกล้อง
option = st.radio("เลือกโหมด:", ("อัปโหลดไฟล์", "เปิดกล้องเรียลไทม์"))

if option == "อัปโหลดไฟล์":
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพหรือวิดีโอ", type=["jpg", "jpeg", "png", "mp4", "avi"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read() 
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension in ["jpg", "jpeg", "png"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            image = cv2.imread(tmp_file_path)
            if image is None:
                st.error("❌ ไม่สามารถเปิดรูปภาพได้!")
            else:
                results = model(image)
                for result in results:
                    image = result.plot()
                st.image(image, caption="ผลลัพธ์จาก YOLOv8", channels="RGB")
                os.remove(tmp_file_path)  # ลบไฟล์หลังใช้งาน
        
        elif file_extension in ["mp4", "avi"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            cap = cv2.VideoCapture(tmp_file_path)
            st_frame = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame)
                for result in results:
                    frame = result.plot()
                st_frame.image(frame, channels="RGB")
            
            cap.release()
            os.remove(tmp_file_path)  # ลบไฟล์หลังใช้งาน

elif option == "เปิดกล้องเรียลไทม์":
    st.warning("ฟีเจอร์กล้องต้องใช้ OpenCV และเข้าถึงเว็บแคม")
    run_camera = st.button("เริ่มกล้อง")
    
    if run_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ ไม่สามารถเปิดกล้องได้!")
        else:
            st_frame = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (640, 640))  # ปรับขนาดเฟรมให้ไม่เกิน 640x640
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame)
                for result in results:
                    frame = result.plot()
                st_frame.image(frame, channels="RGB")
                
            cap.release()
            cv2.destroyAllWindows()