# Build Flow
- 学習モデルを選択する。
  - default_model_urlで他のプロジェクトで学習したモデルを指定することも可能。  
- Trainモードで学習を行う。
- Testモードに遷移すると学習が始まる。
- 推論結果に反映される。
- エディト画面でmodelsに足したものが反映されない場合は、Tools -> Terminal -> "refresh"をタイプ

# Control 
- command + alt + iでインスペクターを出し、consoleを出すと様々な処理の様子が見れる。
- Pull down Tabで既に学習したモデルを呼び出し
- キーボード "1"でTestモード  
- キーボード "2"でTrainモード  
- キーボード "a"でMouseの位置に対する推論  
- キーボード "s"でmyfigureで選択した図形が描かれる。
- save messageで名前をつけて保存すると、modelsに保存される。  
  (名前を変更しない場合は上書き保存される。)   
  ※"1"や"2"を押すとモード変更がされてしまうので、使用しないこと。

# Reference
- [observable LSTM Mouse Movement Predictor](https://observablehq.com/@kiyu/tensorflow-js-lstm-mouse-movement-predictor)