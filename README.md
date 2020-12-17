# spleeter-pytorch-mnn
convert spleeter pretrained model to pytorch and onnx, then convert to mnn

## Download pretrained
* [Original Spleeter](https://github.com/deezer/spleeter) - download [BaiduYun](https://pan.baidu.com/s/1i682L1AM9WAKjRMml9y1Ig): 1o40
* onnx -  download [BaiduYun](https://pan.baidu.com/s/1R62HIx6-PHqzqvPsiary4A): d0nk
* mnn -  download [BaiduYun](https://pan.baidu.com/s/1uFVY-AiyfDxXB1szLogVDg): gnwk

## Convert
TF model → ONNX(use pytorch) → MNN
* TF model → ONNX model(use pytorch)
```shell
python convert2onnx.py
```
* ONNX model → MNN model
```shell
# FP32
mnnconvert -f ONNX --modelFile vocals.onnx --MNNModel vocals.mnn
mnnconvert -f ONNX --modelFile accompaniment.onnx --MNNModel accompaniment.mnn

# FP16
mnnconvert -f ONNX --modelFile vocals.onnx --MNNModel vocals_fp16.mnn --fp16 FP16
mnnconvert -f ONNX --modelFile accompaniment.onnx --MNNModel accompaniment_fp16.mnn --fp16 FP16
```

## Usage
* run with TF model
```shell
python test_estimator.py
```
* run with MNN
```shell
python spleeter_mnn.py
```

## Performance
* Test on my laptop: Inter(R) Core(TM) i5-8300H CPU @ 2.30GHz 2.30GHz

|   -  |   TF model  |   MNN model  |
| --- | ---------   |  ---------   |
| CPU |    0.4666s  |  0.6690s     |

WORSE use MNN?

## Acknowledge
* [Spleeter](https://github.com/deezer/spleeter)
* [spleeter-pytorch](https://github.com/tuan3w/spleeter-pytorch) mainly inspired
* [MNN](https://www.yuque.com/mnn)
