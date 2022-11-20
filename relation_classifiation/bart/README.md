The base code is forked from [REBEL](https://github.com/Babelscape/rebel.git), which formats relation classification from free-text as a sequence to sequence problem and employs BART for classification. 


# Installation
```conda

conda install -c conda-forge pytorch-lightning
conda install -c conda-forge transformers==4.12.4

conda install -c conda-forge omegaconf

pip install datasets
pip install hydra-core
pip  neptune-client
pip  psutil
pip rouge-score
pip  sacrebleu
pip  streamlit
pip  pyDeprecate
pip  setuptools

pip install nltk
pip install wandb
conda install pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch
```

## Data 
The data used for training and test are under the folder of data/cnwae:
* training set: [train_wael_1000.json](data/cnwae/train_wael_1000.json)
* validation set: [dev_wael.json](data/cnwae/dev_wael.json)
* test set:  [test_wael.json](data/cnwae/test_wael.json)

## Train and test

You can run with `run_with_dev.sh` or put the following commands into your own shell file:

```
root='./data/cnwae/'
train_files=('train_wael_1000.json')
validation_files=('dev_wael.json')
test_files=('test_wael.json')
max_steps=1000
train_batch_sizes=8

echo "##### max_steps=${max_steps} ####"
for i in "${!train_files[@]}"; do
    train_file=$root${train_files[$i]}
    validation_file=$root${validation_files[$i]}
    rm -f data/cnwae/*.cache

    for j in "${!test_files[@]}"; do
        test_file=$root${test_files[$j]}
        echo 'train_file='${train_file}, 'validation_file='${validation_file} 'batch_size='${train_batch_size} 'test_file'=${test_file}  "learning_rate=0.00002"
        python3 -u src/train.py model=default_model data=cnwae_data train=cnwae_train train_file=${train_file} validation_file=${validation_file}  max_steps=${max_steps} train_batch_size=${train_batch_sizes} learning_rate=0.00002 do_predict=True  test_file=${test_file}
    done
done
```
