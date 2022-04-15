TOKEN=
git clone https://$TOKEN@github.com/HareshNasit/CSC413-Project.git
cd CSC413-Project

python3 -m pip install pytorch_lightning

bash ./data_loader.sh horse2zebra

python3 train.py