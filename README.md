# MLP example

train example
```
python mlp.py --train_path "./data/preprocessed_train.csv" --model_path "./model/weights" --meta_path "./model/meta_data.pickle" train
```

evaluate example 
```
python mlp.py --target_path "./data/preprocessed_test.csv" --out_path "./result/result_evaluate.csv" --model_path "./model/weights" --meta_path "./model/meta_data.pickle" evaluate
```

predict example
```
+ python mlp.py --target_path "./data/preprocessed_test_no_label.csv" --out_path "./result/result_predict.csv" --model_path "./model/weights" --meta_path "./model/meta_data.pickle" predict
```
