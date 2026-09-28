# -*- coding: utf-8 -*-
"""
app.py - เว็บแอป Streamlit สำหรับใช้โมเดลที่ฝึกและบันทึกจาก Orange (*.pkcls)

วิธีรัน:
    pip install -r requirements.txt
    streamlit run app.py
"""

import glob
import os

import joblib
import streamlit as st
import Orange  # noqa: F401  (ต้อง import ไว้ เพราะไฟล์ .pkcls เป็นโมเดลของ Orange)
from Orange.classification import Model
from Orange.data import Domain, Table

# ------------------------------------------------------------------
# 1) การตั้งค่าเบื้องต้น
# ------------------------------------------------------------------
st.set_page_config(page_title="AI แนะนำอาชีพเสริม", page_icon="💼")

# ค่าคลาสที่ถือว่า "บรรลุเป้าหมาย" (ใช้เฉพาะกรณีโมเดลเป็น Classification)
POSITIVE_LABELS = {"1", "yes", "true", "บรรลุ", "บรรลุเป้าหมาย"}

# ค่าเริ่มต้นของช่องกรอกแต่ละ feature (เป็นค่าตัวอย่างเท่านั้น แก้ให้ตรงกับข้อมูลของคุณได้)
# feature ที่ไม่อยู่ในรายการนี้จะเริ่มที่ 0.0
DEFAULTS = {
    "Monthly_Income": 30000.0,
    "Monthly_Expenditure": 20000.0,
    "Market_Volatility_Index": 20.0,
    "Inflation_Rate": 2.0,
    "Investment_Amount": 50000.0,
    "Savings_Ratio": 0.2,
    "Credit_Score": 650.0,
    "Debt_to_Income_Ratio": 0.3,
    "Risk_Tolerance_Level": 3.0,
    "Economic_Sentiment_Score": 0.5,
    "Investor_Confidence": 0.5,
    "Financial_Stability_Index": 0.5,
}


# ชื่อภาษาไทยของแต่ละ feature (ถ้าไม่มีในรายการ จะแสดงชื่อคอลัมน์เดิม)
LABELS_TH = {
    "Monthly_Income": "รายได้ต่อเดือน",
    "Monthly_Expenditure": "ค่าใช้จ่ายต่อเดือน",
    "Market_Volatility_Index": "ดัชนีความผันผวนของตลาด",
    "Inflation_Rate": "อัตราเงินเฟ้อ",
    "Investment_Amount": "จำนวนเงินลงทุน",
    "Savings_Ratio": "สัดส่วนการออม",
    "Credit_Score": "คะแนนเครดิต",
    "Debt_to_Income_Ratio": "สัดส่วนหนี้ต่อรายได้",
    "Risk_Tolerance_Level": "ระดับการรับความเสี่ยง",
    "Economic_Sentiment_Score": "คะแนนความเชื่อมั่นทางเศรษฐกิจ",
    "Investor_Confidence": "ความมั่นใจของนักลงทุน",
    "Financial_Stability_Index": "ดัชนีความมั่นคงทางการเงิน",
    "Investment_Recommendation_Score": "คะแนนแนะนำการลงทุน",
}


def th(name: str) -> str:
    """คืนชื่อภาษาไทยของคอลัมน์ (ถ้าไม่มีให้คืนชื่อเดิม)"""
    return LABELS_TH.get(name, name)


# ------------------------------------------------------------------
# 2) ฟังก์ชันโหลดโมเดล (cache ไว้ จะได้ไม่โหลดซ้ำทุกครั้งที่กดปุ่ม)
# ------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    """โหลดไฟล์ .pkcls ด้วย joblib (ต้องติดตั้ง Orange3 ไว้ด้วย)"""
    return joblib.load(path)


# ------------------------------------------------------------------
# ฟังก์ชันแนะนำอาชีพเสริม (แบบกฎที่เขียนเอง ไม่ได้มาจากโมเดล ML)
# แก้เงื่อนไขและรายการอาชีพได้ตามต้องการ
# ------------------------------------------------------------------
def recommend_careers(v: dict, achieved: bool):
    """รับค่าที่ผู้ใช้กรอก (dict) และผลบรรลุ/ไม่บรรลุ แล้วคืนรายการ (อาชีพ, เหตุผล) ไม่เกิน 3 รายการ"""
    income = v.get("Monthly_Income", 0.0)
    spend = v.get("Monthly_Expenditure", 0.0)
    debt = v.get("Debt_to_Income_Ratio", 0.0)
    risk = v.get("Risk_Tolerance_Level", 3.0)   # สมมติสเกล 1-5 (แก้ตามข้อมูลฝึกของคุณ)
    credit = v.get("Credit_Score", 0.0)
    surplus = income - spend                     # เงินเหลือต่อเดือน

    recs = []

    # กลุ่ม 1: เงินตึงมือหรือมีหนี้สูง -> เน้นสร้างรายได้เพิ่มโดยใช้ทุนน้อย
    if surplus <= 0 or debt >= 0.4:
        recs += [
            ("รับงานฟรีแลนซ์ตามทักษะ (เขียน/แปล/ออกแบบ/พิมพ์งาน)",
             "ใช้ทุนน้อย เริ่มได้ทันที เหมาะกับช่วงที่เงินเหลือน้อยหรือมีภาระหนี้สูง"),
            ("งานพาร์ทไทม์/ส่งของ/ขับรถรับจ้างในเวลาว่าง",
             "ได้เงินสดต่อเนื่อง ช่วยลดภาระหนี้"),
        ]
    # กลุ่ม 2: ไม่ชอบความเสี่ยง -> งานรายได้มั่นคง
    elif risk <= 2:
        recs += [
            ("สอนพิเศษ/ติวออนไลน์",
             "รายได้ค่อนข้างสม่ำเสมอ ความเสี่ยงต่ำ ไม่ต้องลงทุนสต็อก"),
            ("รับงานฟรีแลนซ์ประจำ (เขียนบทความ/แปล/จัดการข้อมูล)",
             "เหมาะกับคนที่ต้องการความมั่นคงมากกว่าผลตอบแทนสูง"),
        ]
    # กลุ่ม 3: รับความเสี่ยงสูง และมีเงินเหลือพอ -> ธุรกิจที่ต้องลงทุน
    elif risk >= 4 and surplus > 0:
        recs += [
            ("เปิดร้านค้าออนไลน์แบบลงทุนสต็อกสินค้าเอง",
             "รับความเสี่ยงได้สูงและมีเงินเหลือ โอกาสได้กำไรมากกว่าการขายแบบไม่ลงทุน"),
            ("ทำธุรกิจเล็ก/แฟรนไชส์ขนาดย่อม",
             "ต้องใช้เงินทุนและรับความเสี่ยงได้ ควรศึกษาตลาดก่อน"),
        ]
    # กลุ่ม 4: ระดับกลาง
    else:
        recs += [
            ("ขายของออนไลน์แบบพรีออเดอร์/ดรอปชิป",
             "ลงทุนต่ำถึงปานกลาง ความเสี่ยงพอเหมาะ"),
            ("ทำคอนเทนต์/ขายงานดิจิทัล (สื่อ/ภาพ/เทมเพลต)",
             "ต่อยอดจากทักษะที่มี ทำควบคู่งานประจำได้"),
        ]

    # เสริมจากผลบรรลุเป้าหมายและเครดิต
    if achieved and surplus > 0 and credit >= 700:
        recs.append(("ขยายธุรกิจเล็ก ๆ โดยใช้สินเชื่อธุรกิจ SME",
                     "ผลบรรลุเป้าหมาย มีเงินเหลือ และเครดิตดี"))
    elif not achieved:
        recs.append(("เริ่มต้นเล็ก ๆ ทดลองตลาดก่อนขยาย",
                     "ผลยังไม่บรรลุเป้าหมาย ควรลดความเสี่ยงและสร้างรายได้เสริมก่อน"))

    return recs[:3]


def show_careers(v: dict, achieved: bool):
    """แสดงผลอาชีพเสริมที่แนะนำบนหน้าเว็บ"""
    st.subheader("อาชีพเสริมที่แนะนำ")
    for title, reason in recommend_careers(v, achieved):
        st.markdown(f"**{title}**  \n{reason}")
    st.caption(
        "หมายเหตุ: ส่วนนี้ใช้กฎที่เขียนขึ้นจากข้อมูลที่กรอก (เงินเหลือเก็บ หนี้ ความเสี่ยง เครดิต) "
        "ไม่ได้มาจากโมเดล AI ที่ฝึกไว้ จึงเป็นแนวทางเบื้องต้นเท่านั้น"
    )


# ------------------------------------------------------------------
# 3) หัวข้อแอป
# ------------------------------------------------------------------
st.title("โปรแกรม AI แนะนำอาชีพเสริมที่เหมาะสม")

# ------------------------------------------------------------------
# 4) ส่วนเลือกโมเดล: ค้นหาไฟล์ *.pkcls ทั้งหมดในโฟลเดอร์เดียวกับ app.py อัตโนมัติ
#    (เพิ่มไฟล์โมเดลใหม่ได้เลย ไม่ต้องแก้โค้ด)
# ------------------------------------------------------------------
folder = os.path.dirname(os.path.abspath(__file__))
available = {os.path.basename(f): f for f in sorted(glob.glob(os.path.join(folder, "*.pkcls")))}
if not available:
    st.error("ไม่พบไฟล์โมเดล (*.pkcls) กรุณาวางไฟล์ไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()

model_file = st.selectbox("เลือกโมเดลที่ต้องการใช้", list(available.keys()))
model = load_model(available[model_file])

# original_domain = คอลัมน์ "ดิบ" ก่อนผ่าน preprocess ซึ่งเป็นสิ่งที่ผู้ใช้กรอก
domain = model.original_domain if hasattr(model, "original_domain") else model.domain
class_var = domain.class_var          # คอลัมน์เป้าหมาย
is_classification = class_var is not None and class_var.is_discrete

# ------------------------------------------------------------------
# 5) สร้างช่องกรอกข้อมูลตามคอลัมน์ที่ใช้ฝึกโมเดลจริงโดยอัตโนมัติ
#    - คอลัมน์ตัวเลข   -> st.number_input
#    - คอลัมน์หมวดหมู่ -> st.selectbox (ตัวเลือกดึงจากโมเดลเอง)
# ------------------------------------------------------------------
st.subheader("กรอกข้อมูลของคุณ")

inputs = {}
left, right = st.columns(2)
for i, var in enumerate(domain.attributes):
    with (left if i % 2 == 0 else right):
        if var.is_discrete:
            inputs[var.name] = st.selectbox(th(var.name), list(var.values), help=var.name, key=f"in_{model_file}_{var.name}")
        else:
            inputs[var.name] = st.number_input(
                th(var.name),
                value=float(DEFAULTS.get(var.name, 0.0)),
                format="%.4f",
                help=var.name,  # ชื่อคอลัมน์เดิม (ภาษาอังกฤษ) แสดงเมื่อเอาเมาส์ชี้ไอคอน ?
                key=f"in_{model_file}_{var.name}",
            )

# โมเดล Regression ให้ "คะแนน" ไม่ใช่คลาส จึงต้องกำหนดเกณฑ์ว่าคะแนนเท่าไรถึงนับว่าบรรลุ
threshold = None
if not is_classification:
    threshold = st.number_input(
        f"เกณฑ์{th(class_var.name)}ที่ถือว่า 'บรรลุเป้าหมาย' (ตั้งแต่ค่านี้ขึ้นไป)",
        value=50.0,
        format="%.2f",
        help="ปรับให้เหมาะกับช่วงคะแนนในข้อมูลของคุณ",
    )

# ------------------------------------------------------------------
# 6) ปุ่มทำนาย
# ------------------------------------------------------------------
if st.button("ทำนายผล"):
    # 6.1 จัดข้อมูลเป็นแถวเดียว เรียงตามลำดับคอลัมน์ตอนฝึก
    row = [inputs[v.name] for v in domain.attributes]

    # 6.2 แปลงเป็น Orange Table (โมเดล Orange จัดการ one-hot/normalize ที่อยู่ใน workflow ให้เอง)
    data = Table.from_list(Domain(domain.attributes), [row])

    st.subheader("ผลการทำนาย")

    if is_classification:
        # ---------- Classification: ได้ทั้งคลาสและความน่าจะเป็น ----------
        values, probs = model(data, Model.ValueProbs)
        label = class_var.values[int(values[0])]

        achieved = str(label).strip().lower() in POSITIVE_LABELS
        if achieved:
            st.success(f"✅ ผลทำนาย: บรรลุเป้าหมายการออมเงิน ({label})")
        else:
            st.warning(f"⚠️ ผลทำนาย: ไม่บรรลุเป้าหมายการออมเงิน ({label})")

        st.write("ความน่าจะเป็น (Probability):")
        for cls_name, p in zip(class_var.values, probs[0]):
            st.write(f"- {cls_name}: {p * 100:.2f}%")
            st.progress(float(p))
        show_careers(inputs, achieved)
    else:
        # ---------- Regression: ได้คะแนนเป็นตัวเลข (ไม่มี Probability) ----------
        pred = float(model(data)[0])
        st.metric(th(class_var.name), f"{pred:,.2f}")

        achieved = pred >= threshold
        if achieved:
            st.success(f"✅ ผลทำนาย: บรรลุเป้าหมาย (คะแนน {pred:,.2f} ≥ เกณฑ์ {threshold:,.2f})")
        else:
            st.warning(f"⚠️ ผลทำนาย: ไม่บรรลุเป้าหมาย (คะแนน {pred:,.2f} < เกณฑ์ {threshold:,.2f})")

        st.info(
            "โมเดลนี้เป็น Linear Regression จึงให้เป็น 'คะแนน' ไม่ใช่ความน่าจะเป็น "
            "หากต้องการ Probability ต้องฝึกโมเดลแบบ Classification"
        )
        show_careers(inputs, achieved)
