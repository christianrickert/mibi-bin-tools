FROM python:3.8

# system maintenance
RUN apt-get update && apt-get install -y gcc

WORKDIR /scripts

# copy over the requirements.txt, install dependencies, and README
COPY docker-requirements.txt /opt/mibi-bin-tools/
RUN pip install -r /opt/mibi-bin-tools/docker-requirements.txt

COPY requirements.txt /opt/mibi-bin-tools/
RUN pip install -r /opt/mibi-bin-tools/requirements.txt

# copy the scripts over
COPY setup.py /opt/mibi-bin-tools/
COPY mibi_bin_tools /opt/mibi-bin-tools/mibi_bin_tools

COPY README.md /opt/mibi-bin-tools/

# Install the package via setup.py
RUN pip install /opt/mibi-bin-tools

# jupyter lab
CMD jupyter lab --ip=0.0.0.0 --allow-root --no-browser