# 2017HackNTU
2017 Hack NTU Project

Demo 網址 : http://weshare.jianlong.org/

- train.py : 從頭訓練的的cnn model
- pretrain.py : 用現成model下去串接訓練，但效果沒有比較好
- predict.py : 預測test data中資料，輸出準確率和預測結果
- predict_web.py : 網頁版用的python檔，輸入一張照片(不用限制size)，輸出1/0 (1是合格，0是不合格)
- model/cnn_model.h5 : 從頭訓練的cnn model

## 網頁版 predicor:

啟動 web server:

```
FLASK_APP=predict_flask.py flask run
```

用法:

```
curl http://127.0.0.1:5000?input=/path/to/image_file
```
