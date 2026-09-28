import streamlit as st
st.title("ทดสอบ 1")
st.write("แอปเริ่มทำงาน")

import sklearn, numpy, pandas
st.write("sklearn", sklearn.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__)

try:
    import Orange
    st.write("Orange", Orange.__version__)
except Exception as e:
    st.exception(e)
