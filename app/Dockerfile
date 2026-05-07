FROM python:3.12
WORKDIR /student
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python3.12 model.py
EXPOSE 5000
CMD ["python3.12", "app.py"]
