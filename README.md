First run `data_setup.sh`

Make sure you run `export CUDA_AVAILABLE_DEVICES=0` or `export CUDA_AVAILABLE_DEVICES=1` before anything

Run `python -m sampling_task_scripts.gsm_training <experiment name> train ` to train GSM and save parameters to `save/<experiment name>`

Then run `python -m sampling_task_script.ssn_initialisation <experiment name>` to initialise many stable 2x2 matrices, and select the best one (by inhibitory training error)

Finally, loop `python -m sampling_task_script.ssn_initialisation <experiment name> <n>` to take the saved network of size [2n x 2n] and train it on the sampling task, then grow it and save the [2(n+1) x 2(n+1)]
