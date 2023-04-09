FROM python

WORKDIR /home/OrdoHereticus_bot

COPY . .

RUN apt update

RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install nltk
RUN pip install aiofiles
RUN pip install aiogram
RUN pip install googletrans==3.1.0a0
RUN pip install tensorflow
RUN pip install langdetect

CMD ["/bin/bash"]
