build:
	sudo docker build -t ejyou/anti_spam:bot .
run:
	sudo docker run -it --name spam_filter ejyou/anti_spam:bot
stop:
	sudo docker stop spam_filter
start:
	sudo docker start spam_filter
