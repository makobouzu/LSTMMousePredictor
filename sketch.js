const circleSize = 10; //circleの大きさ
let positionHistoryLength = 16; //学習に渡すmouseのpositionの長さ
let mousePos = ({x:0.5, y:0.5}); //mousePosの初期値
let prediction_source =[]; //学習に渡す固定長Array
let rawTrainingData = []; //train時のxy座標を格納する可変長Array
let keepTrain = []; //xy座標とlabelを格納するか変調Array
let predictAhead = 16; //推論される長さ 16手先まで推論
let model;
let batchSize = 32; //一回の学習で回す量
let flag = false; //推論されたかどうか
let mode = 1; //test mode - train mode
// let f = 0; //
let simulationFlag = false; //trainじに推論結果を返すかどうか
let predicted = false; 
const myfigure = myrect();

async function init() {
  prediction_source = Array(positionHistoryLength).fill(mousePos);
  tf.setBackend('cpu'); 
  console.log("Backend: ", tf.getBackend());
  console.log("loading model....");
  model = await tf.loadLayersModel("https://raw.githubusercontent.com/voodoohop/Mouse_tracking_predictor/master/tensorflowjs_model/model.json"); //v2.0.0
  model.predict;
  console.log("model has been loaded!");
  frameRate(30); 
}

async function setup() {
  frameRate(0); //block draw before init
  init();
  createCanvas(400, 400);
}

async function draw() {
  if(mode == 1){
    //test
    background(204, 255, 243);
    //最新のマウス位置からLSTM予測を実行
    let data = []
    if(!simulationFlag){
      data = await predict_result(mouseX, mouseY, prediction_source);
    }else{
      data = await predict_result(myfigure.next().value.x, myfigure.next().value.y, prediction_source);
    }
    //予測結果が取得できたら、予測を描画
    let prediction = data[0];
    prediction_source = data[1]
    //軌跡の描画
    fill(255, 138, 128);
    for(let i = 0; i < prediction_source.length; i++){
      ellipse(prediction_source[i].x * width, prediction_source[i].y * height, circleSize, circleSize);
    }
    //推論の描画
    fill(128, 149, 255);
    for(let i = 0; i < prediction.length; i++){
      ellipse(prediction[i].x * width, prediction[i].y * height, circleSize, circleSize);
    }
  }else if(mode == 2){  
    //train
    background(255,180,153);
    //最新のマウス位置からLSTM予測を実行
    let data = []
    if(!simulationFlag){
      data = await predict_result(mouseX, mouseY, prediction_source);
    }else{
      data = await predict_result(myfigure.next().value.x, myfigure.next().value.y, prediction_source);
    }
    //予測結果が取得できたら、予測を描画
    let prediction = data[0];
    prediction_source = data[1]
    //軌跡の描画
    fill(255, 138, 128);
    for(let i = 0; i < prediction_source.length; i++){
      ellipse(prediction_source[i].x * width, prediction_source[i].y * height, circleSize, circleSize);
    }
    //推論の描画
    fill(128, 149, 255);
    for(let i = 0; i < prediction.length; i++){
      ellipse(prediction[i].x * width, prediction[i].y * height, circleSize, circleSize);
    }
    keepTrain = inputData(prediction_source);
  }
}

function drawPoints(_x, _y){
  //軌跡の描画
  fill(255, 138, 128);
  for(let i = 0; i < array2.length; i++){
    ellipse(array2[i].x * width, array2[i].y * height, circleSize, circleSize);
  }
  //推論の描画
  fill(128, 149, 255);
  for(let i = 0; i < array.length; i++){
    ellipse(array[i].x * width, array[i].y * height, circleSize, circleSize);
  }
}

function pushPositionArray(_x, _y){
  
  
  return 
}


// function keyTyped(){
//   if(key === "1"){
//     //test mode
//     selectMode(0)
//     document.getElementById("test").checked = true;
//   }else if(key === "2"){
//     // train mode 
//     document.getElementById("train").checked = true;
//     selectMode(1)
//   }
//   if(key === "s"){
//     document.getElementById("prediction").checked = true;
//     selectDrawing(1);
//   }else if(key === "a"){
//     document.getElementById("prediction").checked = false;
//     selectDrawing(0);
//   }
//   if(key === "d"){
//     // if(!predicted == true){
//     //   document.getElementById("manual").checked = !predicted;
//     // }else{
//     //   document.getElementById("auto").checked = predicted;
//     // } 
//     selectPrediction(predicted)
//   }
// }


async function predict_result(mouse_x, mouse_y, _lastPositions) {
  mousePos = {x:mouse_x/width, y:mouse_y/height};  
  _lastPositions.push(mousePos);
  _lastPositions.shift();
  //予測データの取得
  let input = _lastPositions;
  let array2 = _lastPositions;
  for (let p=0; p < predictAhead; p++) {
    const predictionNow =  await predict(input);
    input  = [...input, predictionNow].slice(-positionHistoryLength);
  }
  let array = input.slice(-predictAhead); // predict(lastPositions)
  return [array, array2];  
}

async function predict_array(array) {
  return new Promise(resolve =>{
    for (let p=0; p < predictAhead; p++) {
      const predictionNow =  predict(array);
      array  = [...array, predictionNow].slice(-positionHistoryLength);
    }
    console.log(array)
    resolve(array)
  })
}
                     

// template
// async function getPose(_img) {
//   return new Promise(resolve =>{
//     let poses = net.estimateSinglePose(_img, {flipHorizontal: false});
//     resolve(poses);
//   })
// }

//train時のdata収集
function inputData(_prediction_source){
  if(rawTrainingData.lenght < 1){
    rawTrainingData = _prediction_source;
  }else{
    rawTrainingData = [...rawTrainingData, _prediction_source];
  }
  let trainingData = R.zipWith((data,rawLabel) => ({data, label: R.last(rawLabel)}), rawTrainingData, rawTrainingData.slice(1));
  return trainingData
}

//modelの設定
function compiledModel(){
  const LEARNING_RATE = 0.015;
  const optimizer = tf.train.sgd(LEARNING_RATE);
  model.compile({optimizer, loss: 'meanSquaredError'});
  return model;
}

let trainLoss = async (trainingData) => { 
  console.log("starting",trainingData.length);
  if (!trainLoss) return 1;
  while (true) {
    const trainRes = await trainBatch(pickTrainingBatch(trainingData));
    console.log("trained");
    console.log("Train Loss: ",R.last(trainRes.history.loss));
    return R.last(trainRes.history.loss);
  }
}

// function lossHistory(){
//   if(history.length < 1){
//     history = R.last(trainRes.history.loss);
//   }else{
//     history = [...history, R.last(trainRes.history.loss)];
//   }
  
//   const canvas = document.getElementById('canvas_lossHistory');
//   const context = canvas.context2d(width, 100);
//   context.beginPath();
//   context.moveTo(0, 0);
//   for (let x = 0; x < history.length; x++) context.lineTo(x*5, 100*(history[x]/R.max(...history)));
//   context.lineJoin = context.lineCap = "round";
//   context.strokeStyle = "blue";
//   context.stroke();
//   context.canvas;
// }



//detail---------------------------------------------------------------------------
//test_detail
const pointToArray= ({x,y}) => [x,y];
let formatInputTensor = positions => tf.tensor2d(positions.map(pointToArray)).reshape([2,positionHistoryLength]).expandDims();
let predict = async positions => { 
    const [x,y] = await model.predict(formatInputTensor(positions)).data();
    return {x,y}
}

//train_detail
let trainBatch =  async (trainData) => {
  const trainingData = formatTrainingData(trainData)
  return await model.fit(trainingData.data, trainingData.labels, {batchSize,epochs:2 })
}
let formatTrainingData = (inputData) => { 
  const data = tf.concat(inputData.map(R.prop("data")).map(formatInputTensor))
  const labels = tf.tensor(inputData.map(R.prop("label")).map(pointToArray))
  return {labels,data}
}
// let pickTrainingBatch = (data) => R.take(batchSize, myshuffle(data));
let pickTrainingBatch = (data) => R.take(batchSize, data);
function myshuffle(b) {
    const a = R.clone(b);
    var j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
    return a;
}

// select buttons
function selectMode(a){
  if(a == 0){
    // test mode
    document.getElementById("prediction").disabled = true; 
    console.log("test mode");
    if(keepTrain.length > 0){
      compiledModel();  
      trainLoss(keepTrain);
      // lossHistory();
      rawTrainingData = [];
      keepTrain = [];
    }
    mode = 1;
  }else{
    // train mode
     document.getElementById("prediction").disabled = false;
    console.log("train mode");
    mode = 2;
  }
  console.log(a);
}

function selectPrediction(s){
  predicted = s;
  
}

function selectDrawing(a){
  if(a == 0){
    simulationFlag = false;
  }else{
    simulationFlag = true;
  }
}





// example:1 draw rectangle
function* myrect(){
  let i = 0;
  let x = 100;
  let y = 100;
  let side = 200;
  
  while(true){
    let num = i%800;
    if(num/side <= 1){
      x = 100+num;
    }else if(num/side <= 2){
      y = 100 + (num-side*1);
    }else if(num/side <= 3){
      x = 300 - (num-side*2);
    }else if(num/side <= 4){
      y = 300 - (num-side*3);
    }
    yield {x:x, y:y}
    i+= 5;
  }
}









class hoge{
 constructor(a) {
    this.a = a;
  }
  
  get getA(){
    return this.a;
  }
  
  set setA(a){
    this.a = a;
  }
  
}





//movieファイルを再生する?
//レーシングのコースみたいな画像を表示する
//spaceボタンを押している時は、レコード始める・数秒・数点の記録
//classとして整理 run, predict, start record, stop record