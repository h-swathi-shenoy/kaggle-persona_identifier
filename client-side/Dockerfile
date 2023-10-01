FROM public.ecr.aws/lambda/python:3.9

RUN yum -y install gcc-c++

WORKDIR ${LAMBDA_TASK_ROOT}

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

CMD ["main.main"]
