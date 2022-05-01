FROM ubuntu

# install packages
RUN apt update && \
    apt install python3 pip -y

COPY . /conformer-rnnt
WORKDIR /conformer-rnnt

# install project packages
RUN pip install -r requirements.txt

CMD python3 main.py