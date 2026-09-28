import glob, os
import streamlit as st
st.title("ทดสอบ 2")

folder = os.path.dirname(os.path.abspath(__file__))
st.write("โฟลเดอร์:", folder)
st.write("ไฟล์ในโฟลเดอร์:", sorted(os.listdir(folder)))

files = glob.glob(os.path.join(folder, "*.pkcls"))
st.write("พบไฟล์โมเดล:", [os.path.basename(f) for f in files])

if files:
    try:
        import joblib, Orange
        model = joblib.load(files[0])
        st.write("โหลดโมเดลสำเร็จ:", type(model).__name__)
        domain = getattr(model, "original_domain", model.domain)
        st.write("ตัวแปรต้น:", [a.name for a in domain.attributes])
        st.write("ตัวแปรตาม:", domain.class_var.name if domain.class_var else None)
    except Exception as e:
        st.exception(e)
