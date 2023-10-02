FROM python:3.9-slim
ENV REPO_URL=PersonIdentifier
COPY . $REPO_URL
WORKDIR $REPO_URL

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8080

ENTRYPOINT [ "streamlit", "run","main.py"]
