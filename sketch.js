const circleSize = 10;
let positionHistoryLength = 16;
let mousePos = ({x:0.5, y:0.5});
let lastPositions =[];
let rawTrainingData = [];
let keepTrain = [];
let predictAhead = 16;
let model;
let batchSize = 32;//一回の学習で回す量
let flag = false;
let mode = 1;
let f = 0;
let simulationFlag = false;
let predicted = false;


function setup() {
  frameRate(30);
  init();
  createCanvas(400, 400);
}

function draw() {
  if(mode == 1){
    //testの処理
    background(204, 255, 243);
    // textSize(32);
    // fill(0);
    // text('test mode', 10, 30)

    //最新のマウス位置からLSTM予測を実行
    if(!simulationFlag){
      predict_result(mouseX, mouseY);
    }else{
      predict_result(myrect(f)[0], myrect(f)[1]);
    }

    //予測結果が取得できたら、予測を描画
    if(flag){
      //mouseの軌跡
      fill(255, 138, 128);
      for(let i = 0; i < array2.length; i++){
        ellipse(array2[i].x * width, array2[i].y * height, circleSize, circleSize);
      }
      //predict
      fill(128, 149, 255);
      for(let i = 0; i < array.length; i++){
        ellipse(array[i].x * width, array[i].y * height, circleSize, circleSize);
      }
    }
  }else if(mode == 2){
    //trainの処理
    background(255,180,153);
    // textSize(32);
    // fill(0);
    // text('train mode', 10, 30);

    if(!predicted){
      if(!simulationFlag){
        mousePos = {x:mouseX/width, y:mouseY/height};
      }else{
        mousePos = {x:myrect(f)[0]/width, y:myrect(f)[1]/height};
      }
    }else{
      if(!simulationFlag){
        mousePos = {x:mouseX/width, y:mouseY/height};
        predict_result(mouseX, mouseY);
      }else{
        mousePos = {x:myrect(f)[0]/width, y:myrect(f)[1]/height};
        predict_result(myrect(f)[0], myrect(f)[1]);
      }
    }
    lastPositions.push(mousePos);
    lastPositions.shift();
    fill(255, 138, 128);
    if(!predicted){
      fill(255, 138, 128);
      for(let i = 0; i < array2.length; i++){
        ellipse(array2[i].x * width, array2[i].y * height, circleSize, circleSize);
      }
    }else{
      if(flag){
        fill(255, 138, 128);
        for(let i = 0; i < array2.length; i++){
          ellipse(array2[i].x * width, array2[i].y * height, circleSize, circleSize);
        }

        fill(128, 149, 255);
        for(let i = 0; i < array.length; i++){
          ellipse(array[i].x * width, array[i].y * height, circleSize, circleSize);
        }
      }
    }
    keepTrain = inputData(lastPositions);
  }
  f+=5;
}

// example1 draw rectangle
let x = 100;
let y = 100;
function myrect(i){
  let num = i%800;
  let side = 200;
  if(num/side <= 1){
    x = 100+num;
  }else if(num/side <= 2){
    y = 100 + (num-side*1);
  }else if(num/side <= 3){
    x = 300 - (num-side*2);
  }else if(num/side <= 4){
    y = 300 - (num-side*3);
  }
  return [x, y];
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


async function init() {
  lastPositions = Array(positionHistoryLength).fill(mousePos);
  tf.setBackend('cpu');
  console.log("Backend: ", tf.getBackend());
  console.log("loading model....");
  // model = await tf.loadModel("https://raw.githubusercontent.com/voodoohop/Mouse_tracking_predictor/master/tensorflowjs_model/model.json"); //v0.9.0
  model = await tf.loadLayersModel("https://raw.githubusercontent.com/voodoohop/Mouse_tracking_predictor/master/tensorflowjs_model/model.json"); //v2.0.0
  model.predict;
  console.log("model has been loaded!");
}

//test時の予測
let predict_result = async (mouse_x, mouse_y)=>{
  mousePos = {x:mouse_x/width, y:mouse_y/height};
  lastPositions.push(mousePos);
  lastPositions.shift();
  // lastPositionsの流し込み
  /// 引数:positions, =>以下が関数formatInputTensorの内容
  /// await以下が返されるまで処理を待つ関数
  //予測データの取得
  let input = lastPositions;
  array2 = lastPositions;
  for (let p=0; p < predictAhead; p++) {
    const predictionNow =  await predict(input);
    input  = [...input, predictionNow].slice(-positionHistoryLength);
  }
  array = input.slice(-predictAhead); // predict(lastPositions)
  //本当は16点データを返したい（lastPositionsがawaitしてくれない）
  // array2 = input.slice(0, -predictAhead);
  flag = true;
}


//train時のdata収集
function inputData(positions){
  if(rawTrainingData.lenght < 1){
    rawTrainingData = lastPositions;
  }else{
    rawTrainingData = [...rawTrainingData, lastPositions];
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
let pickTrainingBatch = (data) => R.take(batchSize, myshuffle(data));
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
