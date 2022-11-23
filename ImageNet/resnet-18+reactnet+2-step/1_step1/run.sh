clear
mkdir log
python train.py --batch_size=512 --learning_rate=1e-3 --epochs=256 --weight_decay=1e-5 | tee -a log/training.txt
