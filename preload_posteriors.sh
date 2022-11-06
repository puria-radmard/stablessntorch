# Get posterior targets Wayne previously trained

experiment_name=$1

mkdir save/$experiment_name

gdown https://drive.google.com/uc?id=1mkCvK0xwvUrHAIrbtvorwqQ558CZQRZF -O save/$experiment_name/gsm_posterior_distribution_means.npy
gdown https://drive.google.com/uc?id=1Gcko5XsYrb0su_lZMUosdkQ_0xaWU-B- -O save/$experiment_name/gsm_posterior_distribution_covs.npy
gdown https://drive.google.com/uc?id=1fgjABI91PRK02-AI4I8BDne8Zy1nE3Pn -O save/$experiment_name/ssn_inputs.npy
