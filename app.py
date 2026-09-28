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

# ค่าคลาสที่ถือว่า "พร้อมต่อยอด" (ใช้เฉพาะกรณีโมเดลเป็น Classification)
POSITIVE_LABELS = {"1", "yes", "true", "บรรลุ", "บรรลุเป้าหมาย", "พร้อมต่อยอด"}

# ------------------------------------------------------------------
# เกณฑ์ที่ใช้ในแอป (ผู้พัฒนาตั้งเอง ไม่ได้มาจากโมเดลหรือชุดข้อมูล)
# แก้ตรงนี้ที่เดียว ทั้งตัวกฎและคำอธิบายบนหน้าเว็บจะเปลี่ยนตามอัตโนมัติ
# ควรปรับให้ตรงกับการกระจายของข้อมูลจริง (ดูด้วย df.describe())
# ------------------------------------------------------------------
SCORE_THRESHOLD_DEFAULT = 30.0   # คะแนนแนะนำการลงทุนที่ถือว่า "พร้อมต่อยอด"
DEBT_HIGH = 0.4                  # สัดส่วนหนี้ต่อรายได้ที่ถือว่าสูง
RISK_LOW = 2                     # ระดับความเสี่ยงที่ถือว่าต่ำ (สมมติสเกล 1-5)
RISK_HIGH = 4                    # ระดับความเสี่ยงที่ถือว่าสูง (สมมติสเกล 1-5)
CREDIT_GOOD = 700                # คะแนนเครดิตที่ถือว่าดี

# ค่าเริ่มต้นของช่องกรอกแต่ละ feature (เป็นค่าตัวอย่างเท่านั้น แก้ให้ตรงกับข้อมูลของคุณได้)
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
# 3) ฟังก์ชันแนะนำอาชีพเสริม (แบบกฎที่เขียนเอง ไม่ได้มาจากโมเดล ML)
#    คืนรายการ (อาชีพ, คำอธิบายอาชีพ, เหตุผล, อ้างอิงจาก) ไม่เกิน 3 รายการ
# ------------------------------------------------------------------
def recommend_careers(v: dict, achieved: bool, score=None, threshold=None):
    income = v.get("Monthly_Income", 0.0)
    spend = v.get("Monthly_Expenditure", 0.0)
    debt = v.get("Debt_to_Income_Ratio", 0.0)
    risk = v.get("Risk_Tolerance_Level", 3.0)
    credit = v.get("Credit_Score", 0.0)
    surplus = income - spend                     # เงินเหลือต่อเดือน

    recs = []

    # กลุ่ม 1: เงินตึงมือหรือมีหนี้สูง
    if surplus <= 0 or debt >= DEBT_HIGH:
        why = []
        if surplus <= 0:
            why.append(f"รายจ่าย ({spend:,.0f} บาท) เท่ากับหรือมากกว่ารายได้ ({income:,.0f} บาท) ต่อเดือน")
        if debt >= DEBT_HIGH:
            why.append(f"สัดส่วนหนี้ต่อรายได้สูง ({debt:.0%}) ซึ่งถึงเกณฑ์ที่ควรเร่งหารายได้เพิ่ม")
        why = " และ ".join(why)
        basis = (f"เกณฑ์: เงินเหลือ ≤ 0 บาท หรือหนี้ต่อรายได้ ≥ {DEBT_HIGH:.0%} | "
                 f"ค่าของคุณ: เงินเหลือ {surplus:,.0f} บาท, หนี้ต่อรายได้ {debt:.0%}")
        recs += [
            ("รับงานฟรีแลนซ์ตามทักษะ (เขียน/แปล/ออกแบบ/พิมพ์งาน)",
             "ใช้ทุนน้อย เริ่มได้ทันที", why, basis),
            ("งานพาร์ทไทม์/ส่งของ/ขับรถรับจ้างในเวลาว่าง",
             "ได้เงินสดต่อเนื่อง ช่วยลดภาระหนี้", why, basis),
        ]
    # กลุ่ม 2: ไม่ชอบความเสี่ยง
    elif risk <= RISK_LOW:
        why = f"ระดับการรับความเสี่ยงของคุณต่ำ ({risk:g}) จึงเหมาะกับงานที่ไม่ต้องลงทุนและไม่เสี่ยงขาดทุน"
        basis = f"เกณฑ์: ความเสี่ยง ≤ {RISK_LOW} | ค่าของคุณ: ความเสี่ยง {risk:g}"
        recs += [
            ("สอนพิเศษ/ติวออนไลน์",
             "รายได้ค่อนข้างสม่ำเสมอ ไม่ต้องลงทุนสต็อก", why, basis),
            ("รับงานฟรีแลนซ์ประจำ (เขียนบทความ/แปล/จัดการข้อมูล)",
             "เหมาะกับคนที่ต้องการความมั่นคงมากกว่าผลตอบแทนสูง", why, basis),
        ]
    # กลุ่ม 3: รับความเสี่ยงสูงและมีเงินเหลือ
    elif risk >= RISK_HIGH and surplus > 0:
        why = (f"คุณรับความเสี่ยงได้สูง ({risk:g}) และมีเงินเหลือ {surplus:,.0f} บาทต่อเดือน "
               "จึงมีทุนและความพร้อมรับความเสี่ยงสำหรับธุรกิจที่ต้องลงทุน")
        basis = (f"เกณฑ์: ความเสี่ยง ≥ {RISK_HIGH} และเงินเหลือ > 0 บาท | "
                 f"ค่าของคุณ: ความเสี่ยง {risk:g}, เงินเหลือ {surplus:,.0f} บาท")
        recs += [
            ("เปิดร้านค้าออนไลน์แบบลงทุนสต็อกสินค้าเอง",
             "โอกาสได้กำไรมากกว่าการขายแบบไม่ลงทุน", why, basis),
            ("ทำธุรกิจเล็ก/แฟรนไชส์ขนาดย่อม",
             "ต้องใช้เงินทุนและรับความเสี่ยงได้ ควรศึกษาตลาดก่อน", why, basis),
        ]
    # กลุ่ม 4: ระดับกลาง
    else:
        why = (f"ความเสี่ยงระดับกลาง ({risk:g}) และมีเงินเหลือ {surplus:,.0f} บาทต่อเดือน "
               "เหมาะกับงานที่ลงทุนไม่สูงและทำควบคู่งานประจำได้")
        basis = (f"เกณฑ์: ความเสี่ยงอยู่ระหว่าง {RISK_LOW} ถึง {RISK_HIGH} และมีเงินเหลือ | "
                 f"ค่าของคุณ: ความเสี่ยง {risk:g}, เงินเหลือ {surplus:,.0f} บาท")
        recs += [
            ("ขายของออนไลน์แบบพรีออเดอร์/ดรอปชิป",
             "ลงทุนต่ำถึงปานกลาง ความเสี่ยงพอเหมาะ", why, basis),
            ("ทำคอนเทนต์/ขายงานดิจิทัล (สื่อ/ภาพ/เทมเพลต)",
             "ต่อยอดจากทักษะที่มี", why, basis),
        ]

    # เสริมจากผลคะแนนและเครดิต
    score_txt = f"คะแนน {score:,.2f}" if score is not None else "ผลทำนาย"
    thr_txt = f"{threshold:,.2f}" if threshold is not None else "เกณฑ์ที่ตั้งไว้"
    if achieved and surplus > 0 and credit >= CREDIT_GOOD:
        recs.append((
            "ขยายธุรกิจเล็ก ๆ โดยใช้สินเชื่อธุรกิจ SME",
            "ต่อยอดรายได้ด้วยเงินทุนจากสถาบันการเงิน",
            f"ผลทำนายพร้อมต่อยอด มีเงินเหลือ {surplus:,.0f} บาทต่อเดือน และเครดิตสกอร์ดี ({credit:,.0f})",
            f"เกณฑ์: คะแนน ≥ {thr_txt}, เงินเหลือ > 0 บาท, เครดิต ≥ {CREDIT_GOOD} | "
            f"ค่าของคุณ: {score_txt}, เงินเหลือ {surplus:,.0f} บาท, เครดิต {credit:,.0f}",
        ))
    elif not achieved:
        recs.append((
            "เริ่มต้นเล็ก ๆ ทดลองตลาดก่อนขยาย",
            "ลดความเสี่ยงระหว่างสร้างรายได้เสริม",
            "ผลทำนายชี้ว่าควรเสริมรายได้ก่อน จึงควรสร้างรายได้เสริมก่อนเพิ่มความเสี่ยง",
            f"เกณฑ์: คะแนน < {thr_txt} | ค่าของคุณ: {score_txt}",
        ))

    return recs[:3]


def show_criteria(threshold):
    """อธิบายว่าเกณฑ์ต่าง ๆ ในแอปมาจากไหน"""
    thr = f"{threshold:,.2f}" if threshold is not None else f"{SCORE_THRESHOLD_DEFAULT:,.2f}"
    with st.expander("เกณฑ์ที่แอปใช้มาจากไหน?"):
        st.markdown(
            "**1) คะแนนแนะนำการลงทุน**  \n"
            "คำนวณโดยโมเดล Linear Regression ที่ฝึกจากชุดข้อมูล Financial Planning Optimization "
            "(Kaggle) ใช้ข้อมูลที่คุณกรอกทั้งหมดเป็นตัวแปรนำเข้า\n\n"
            f"**2) เกณฑ์ 'พร้อมต่อยอด' = {thr}**  \n"
            "เป็นค่าที่ผู้พัฒนาตั้งเอง ไม่ได้มาจากโมเดล คะแนนถึงเกณฑ์ = พร้อมต่อยอด "
            "ต่ำกว่าเกณฑ์ = ควรเสริมรายได้ก่อน คุณปรับค่านี้ได้ในช่องด้านบน\n\n"
            "**3) เกณฑ์ที่ใช้เลือกอาชีพเสริม** (ผู้พัฒนาตั้งเอง)"
        )
        st.markdown(
            f"- เงินเหลือ ≤ 0 บาท หรือหนี้ต่อรายได้ ≥ {DEBT_HIGH:.0%} → งานที่ใช้ทุนน้อย  \n"
            f"- ความเสี่ยง ≤ {RISK_LOW} → งานรายได้มั่นคง  \n"
            f"- ความเสี่ยง ≥ {RISK_HIGH} และมีเงินเหลือ → ธุรกิจที่ต้องลงทุน  \n"
            f"- ความเสี่ยงอยู่ระหว่างนั้น → งานลงทุนต่ำถึงปานกลาง  \n"
            f"- พร้อมต่อยอด + มีเงินเหลือ + เครดิต ≥ {CREDIT_GOOD} → สินเชื่อ SME"
        )
        st.caption(
            "ตัวเลขเกณฑ์ในข้อ 2-3 เป็นค่าประมาณเพื่อเป็นแนวทางเบื้องต้น ยังไม่ได้ปรับตามการกระจายของข้อมูลจริง "
            "และไม่ใช่มาตรฐานทางการเงิน"
        )


def show_careers(v: dict, achieved: bool, score=None, threshold=None):
    """แสดงผลอาชีพเสริมที่แนะนำ พร้อมเหตุผลและสิ่งที่ใช้อ้างอิง"""
    st.subheader("อาชีพเสริมที่แนะนำ")

    if achieved:
        st.info("คะแนนของคุณถึงเกณฑ์ 'พร้อมต่อยอด' อาชีพเสริมด้านล่างช่วยเพิ่มรายได้หรือต่อยอดจากฐานการเงินที่มี")
    else:
        st.info("คะแนนของคุณต่ำกว่าเกณฑ์ จึงแนะนำให้เสริมรายได้ก่อน เพื่อเพิ่มความพร้อมทางการเงินในการลงทุนต่อไป")

    for title, desc, why, basis in recommend_careers(v, achieved, score, threshold):
        st.markdown(f"**{title}**  \n{desc}")
        st.caption(f"💡 ทำไมถึงแนะนำ: {why}")
        st.caption(f"📌 อ้างอิงจาก: {basis}")

    show_criteria(threshold)
    st.caption(
        "หมายเหตุ: ส่วนนี้ใช้กฎที่เขียนขึ้นจากข้อมูลที่กรอก (เงินเหลือเก็บ หนี้ ความเสี่ยง เครดิต) "
        "ไม่ได้มาจากโมเดล AI ที่ฝึกไว้ จึงเป็นแนวทางเบื้องต้นเท่านั้น"
    )


# ------------------------------------------------------------------
# 4) หัวข้อแอป
# ------------------------------------------------------------------
st.title("โปรแกรม AI แนะนำอาชีพเสริมที่เหมาะสม")

# ------------------------------------------------------------------
# 5) ส่วนเลือกโมเดล: ค้นหาไฟล์ *.pkcls ในโฟลเดอร์เดียวกับ app.py และโฟลเดอร์ models/
# ------------------------------------------------------------------
folder = os.path.dirname(os.path.abspath(__file__))
paths = sorted(
    glob.glob(os.path.join(folder, "*.pkcls"))
    + glob.glob(os.path.join(folder, "models", "*.pkcls"))
)
available = {os.path.basename(f): f for f in paths}
if not available:
    st.error("ไม่พบไฟล์โมเดล (*.pkcls) กรุณาวางไฟล์ไว้โฟลเดอร์เดียวกับ app.py หรือในโฟลเดอร์ models/")
    st.stop()

model_file = st.selectbox("เลือกโมเดลที่ต้องการใช้", list(available.keys()))

try:
    model = load_model(available[model_file])
except Exception as e:
    st.error("โหลดโมเดลไม่สำเร็จ")
    st.exception(e)
    st.stop()

# original_domain = คอลัมน์ "ดิบ" ก่อนผ่าน preprocess ซึ่งเป็นสิ่งที่ผู้ใช้กรอก
domain = model.original_domain if hasattr(model, "original_domain") else model.domain
class_var = domain.class_var          # คอลัมน์เป้าหมาย
is_classification = class_var is not None and class_var.is_discrete

# ------------------------------------------------------------------
# 6) สร้างช่องกรอกข้อมูลตามคอลัมน์ที่ใช้ฝึกโมเดลจริงโดยอัตโนมัติ
# ------------------------------------------------------------------
st.subheader("กรอกข้อมูลของคุณ")

inputs = {}
left, right = st.columns(2)
for i, var in enumerate(domain.attributes):
    with (left if i % 2 == 0 else right):
        if var.is_discrete:
            inputs[var.name] = st.selectbox(
                th(var.name), list(var.values), help=var.name,
                key=f"in_{model_file}_{var.name}",
            )
        else:
            inputs[var.name] = st.number_input(
                th(var.name),
                value=float(DEFAULTS.get(var.name, 0.0)),
                format="%.4f",
                help=var.name,  # ชื่อคอลัมน์เดิม (ภาษาอังกฤษ) แสดงเมื่อเอาเมาส์ชี้ไอคอน ?
                key=f"in_{model_file}_{var.name}",
            )

# โมเดล Regression ให้ "คะแนน" ไม่ใช่คลาส จึงต้องกำหนดเกณฑ์ว่าคะแนนเท่าไรถึงนับว่าพร้อมต่อยอด
threshold = None
if not is_classification:
    threshold = st.number_input(
        f"เกณฑ์{th(class_var.name)}ที่ถือว่า 'พร้อมต่อยอด' (ตั้งแต่ค่านี้ขึ้นไป)",
        value=SCORE_THRESHOLD_DEFAULT,
        format="%.2f",
        help="ค่านี้ผู้พัฒนาตั้งเอง ไม่ได้มาจากโมเดล ปรับให้เหมาะกับช่วงคะแนนในข้อมูลของคุณได้",
    )

# ------------------------------------------------------------------
# 7) ปุ่มทำนาย
# ------------------------------------------------------------------
if st.button("ทำนายผล"):
    # 7.1 จัดข้อมูลเป็นแถวเดียว เรียงตามลำดับคอลัมน์ตอนฝึก
    row = [inputs[v.name] for v in domain.attributes]

    # 7.2 แปลงเป็น Orange Table
    data = Table.from_list(Domain(domain.attributes), [row])

    st.subheader("ผลการทำนาย")

    if is_classification:
        # ---------- Classification: ได้ทั้งคลาสและความน่าจะเป็น ----------
        values, probs = model(data, Model.ValueProbs)
        label = class_var.values[int(values[0])]

        achieved = str(label).strip().lower() in POSITIVE_LABELS
        if achieved:
            st.success(f"✅ ผลทำนาย: พร้อมต่อยอด ({label})")
        else:
            st.warning(f"⚠️ ผลทำนาย: ควรเสริมรายได้ก่อน ({label})")

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
            st.success(f"✅ ผลทำนาย: พร้อมต่อยอด (คะแนน {pred:,.2f} ≥ เกณฑ์ {threshold:,.2f})")
        else:
            st.warning(f"⚠️ ผลทำนาย: ควรเสริมรายได้ก่อน (คะแนน {pred:,.2f} < เกณฑ์ {threshold:,.2f})")

        st.info(
            "โมเดลนี้เป็น Linear Regression จึงให้เป็น 'คะแนน' ไม่ใช่ความน่าจะเป็น "
            "หากต้องการ Probability ต้องฝึกโมเดลแบบ Classification"
        )
        show_careers(inputs, achieved, pred, threshold)
