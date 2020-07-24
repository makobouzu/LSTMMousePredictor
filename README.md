# Build Flow
- Showする。
- 学習モデルを選択する。  
  (default_model_urlで他のプロジェクトで学習したモデルを指定可能)
- Trainモードで学習を行う。
- Testモードに遷移すると学習が始まる。
- 推論結果に反映される。
- エディト画面でmodelsに足したものが反映されない場合
  Tools -> Terminal
  ```
  refresh
  ```

# Control 
- Pull down Tabで既に学習したモデルを呼び出し。
- キーボード "1"でTestモード  
- キーボード "2"でTrainモード  
- キーボード "a"でMouseの位置に対する推論  
- キーボード "s"でmyfigureで選択した図形が描かれる。
- 「推論を表示」<->「推論を非表示」を切り替える。
- save messageで名前をつけて保存すると、modelsに保存される。  

# Save  
- 学習後に名前をつけて保存する。
- 名前を変更しない場合は上書き保存される。
  


# Reference
- [observable LSTM Mouse Movement Predictor](https://observablehq.com/@kiyu/tensorflow-js-lstm-mouse-movement-predictor)