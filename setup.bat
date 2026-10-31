@echo off
py -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup done. Next: python -m src.train   then   streamlit run app.py
