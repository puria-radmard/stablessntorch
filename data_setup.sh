# Run for data!

mkdir data
mkdir data/greyscale_cifar

mkdir logs
mkdir save

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O data/cifar-10-python.tar
tar -xvf data/cifar-10-python.tar -C data

python -m utils.process_cifar
