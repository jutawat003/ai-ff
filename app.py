import glob, os
import streamlit as st
st.title("ทดสอบ 3")

try:
    import joblib, Orange
    from Orange.classification import Model
    from Orange.data import Domain, Table
    st.write("import Orange.classification / Orange.data สำเร็จ")

    folder = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(folder, "investment_linear_regression_model.pkcls"))
    domain = getattr(model, "original_domain", model.domain)

    row = [0.0] * len(domain.attributes)
    data = Table.from_list(Domain(domain.attributes), [row])
    st.write("สร้าง Table สำเร็จ", data.X.shape)

    pred = float(model(data)[0])
    st.write("ทำนายสำเร็จ:", pred)
except Exception as e:
    st.exception(e)
