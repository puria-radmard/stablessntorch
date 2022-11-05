# Run for data!

mkdir data
mkdir data/greyscale_cifar

mkdir logs

mkdir save
mkdir save/network_growth

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O data/cifar-10-python.tar
tar -xvf data/cifar-10-python.tar

python -m utils.process_cifar
