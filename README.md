### Line CNN

- LineCNNSimple: Reading multiple characters at once

```
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=emnist_lines.EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=line_cnn_simple.LineCNNSimple --window_width=28 --window_stride=28
```

```
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=emnist_lines.EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=line_cnn_simple.LineCNNSimple --window_width=28 --window_stride=20
```

```
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=emnist_lines.EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=line_cnn_simple.LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
```

```
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=emnist_lines.EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=line_cnn_simple.LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
```
- Loss Function

```
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=emnist_lines.EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=line_cnn.LineCNN --window_width=28 --window_stride=20 --loss=ctc
```
- CNN LSTM

```
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=emnist_lines.EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=line_cnn_lstm.LineCNNLSTM --window_width=28 --window_stride=18 --loss=ctc
```

