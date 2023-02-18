FROM python:3.8.0

LABEL NAME="ETH_Kalman_Filter"
LABEL VERSION=0.1

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY config/ /app/config/
COPY filters/ /app/filters/
COPY setup.py /app/setup.py
COPY README.md /app/README.md
COPY constants.py /app/constants.py


WORKDIR /app
CMD python -m filters --config=config/main.yml


