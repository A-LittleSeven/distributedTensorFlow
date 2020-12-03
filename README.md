# distributedTensorFlow
A demo for distributed TensorFlow

how to run this project:
```shell script
python distributedTensorflow.py --job_name="ps" --task_index=0
python distributedTensorflow.py --job_name="worker" --task_index=0
python distributedTensorflow.py --job_name="worker" --task_index=1
python distributedTensorflow.py --job_name="worker" --task_index=2
```