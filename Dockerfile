FROM python:3.9
EXPOSE 8501
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "src/Home.py"]