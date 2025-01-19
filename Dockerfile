FROM python:3.13

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Install PyCharm's debugger (pydevd)
#RUN pip install pydevd-pycharm

# Make port 5678 available for debugging
#EXPOSE 5678

#CMD ["python", "main.py"]