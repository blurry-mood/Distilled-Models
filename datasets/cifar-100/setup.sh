wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvf cifar-100-python.tar.gz 
rm cifar-100-python.tar.gz

mv cifar-100-python dataset
cd dataset
mkdir Train
mkdir Test

python ../organize.py

rm train test meta file.txt~
mv Train train 
mv Test test