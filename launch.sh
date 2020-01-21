# 1. installer docker (community edition)
# 2. créer un répertoire de travail
# 3. remplacer ??/change/this?? par le chemin vers le répertoire de travail choisi
# 4. lancer la commande ci-dessous ou le exécuter le script `bash launch.sh`

sudo docker run -it -p 8888:8888 \
	-v "/home/renan/Documents/ENSAI/Indexation Web/TP1":/root/sharedfolder \
	continuumio/anaconda3 /bin/bash \
	-c "pip install pycodestyle flake8 pycodestyle_magic &&
	jupyter lab --notebook-dir=/root/sharedfolder --ip='*' --port=8888 --no-browser --allow-root"
