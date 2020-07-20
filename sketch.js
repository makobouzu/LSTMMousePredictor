let   model;
const position_history_length = 16; //学習に渡すmouseのpositionの長さ
const predict_ahead           = 16; //推論される長さ 16手先まで推論
let   prediction_source       = []; //mouseの軌跡・figureの軌跡
let   raw_data                = []; //train時にxy座標を格納する可変長Array
let   keep_data               = []; //xy座標とlabelを格納するか変調Array
const batchSize               = 32; //一回の学習で回す数
const circle_size             = 10;
let   mode                    = 1; //test mode(1) - train mode(2)
let   simulation_flg          = false; //train時に推論結果を返すかどうか
const myfigure                = myrect();


//配列変換の便利機能
const pointToArray      = ({x,y})   => [x,y];
const arrayToPoint      = (_array)  => {return {x:_array[0], y:_array[1]};}
let   formatInputTensor = positions => tf.tensor2d(positions.map(pointToArray)).reshape([2, position_history_length]).expandDims();


async function init() {
  prediction_source = Array(position_history_length).fill({x:1, y:1});
  tf.setBackend('cpu'); 
  console.log("Backend", tf.getBackend());
  console.log("loading model....");
  model = await tf.loadLayersModel("https://raw.githubusercontent.com/voodoohop/Mouse_tracking_predictor/master/tensorflowjs_model/model.json"); //v2.0.0
  model.predict;
  console.log("model has been loaded!");
  frameRate(30); 
}

function setup() {
  frameRate(0); //block draw before init
  init();
  createCanvas(400, 400);
}

async function draw() {
  if(mode == 1){
    //test
    background(204, 255, 243);
    
    let data = [];
    if(!simulation_flg){
      data = await predict(mouseX, mouseY, prediction_source);
    }else{
      data = await predict(myfigure.next().value.x, myfigure.next().value.y, prediction_source);
    }
    let prediction = data[0];
    prediction_source = data[1];
    
    drawPoints(prediction_source, prediction);
    
  }else if(mode == 2){  
    //train
    background(255,180,153);
    
    let data = [];
    if(!simulation_flg){
      data = await predict(mouseX, mouseY, prediction_source);
    }else{
      data = await predict(myfigure.next().value.x, myfigure.next().value.y, prediction_source);
    }
    let prediction = data[0];
    prediction_source = data[1];
    
    drawPoints(prediction_source, prediction);
    
    keep_data = formatData(prediction_source);//データの引き渡し
  }
}

//描画
function drawPoints(_prediction_source, _prediction){
  //軌跡の描画
  fill(255, 138, 128);
  for(let i = 0; i < _prediction_source.length; i++){
    ellipse(_prediction_source[i].x * width, _prediction_source[i].y * height, circle_size, circle_size);
  }
  //推論の描画
  fill(128, 149, 255);
  for(let i = 0; i < _prediction.length; i++){
    ellipse(_prediction[i].x * width, _prediction[i].y * height, circle_size, circle_size);
  }
}

//推論
async function predict(_x, _y, _prediction_source) {
  let mousePos = {x:_x/width, y:_y/height};  
  _prediction_source.push(mousePos);
  _prediction_source.shift();
  
  //データの取得
  let input = _prediction_source;
  let prediction_source = _prediction_source;
  for (let p=0; p < predict_ahead; p++) {
    const predictionNow =  arrayToPoint(await model.predict(formatInputTensor(input)).data());
    input  = [...input, predictionNow].slice(-position_history_length);
  }
  let prediction = input.slice(-predict_ahead);
  return [prediction, prediction_source];  
}

//data収集・整形
function formatData(_prediction_source){
  if(raw_data.lenght < 1){
    raw_data = _prediction_source;
  }else{
    raw_data = [...raw_data, _prediction_source];
  }
  let training_data = R.zipWith((data,rawLabel) => ({data, label: R.last(rawLabel)}), raw_data, raw_data.slice(1));
  return training_data;
}

//train
async function trainLoss(_training_data) { 
  console.log("starting", _training_data.length);
  
  //モデルの設定
  const LEARNING_RATE = 0.015;
  const optimizer = tf.train.sgd(LEARNING_RATE);
  model.compile({optimizer, loss: 'meanSquaredError'});
  
  while (true) {
    const pick_batch = R.take(batchSize, _training_data); //batchsizeずつ取得
    const train_res = await trainBatch(pick_batch); // 
    console.log("trained");
    console.log("Train Loss: ",R.last(train_res.history.loss));
    return R.last(train_res.history.loss);
  }
}
//use in trainLoss, モデルの学習
async function trainBatch(_train_data){
  const data = tf.concat(_train_data.map(R.prop("data")).map(formatInputTensor));
  const labels = tf.tensor(_train_data.map(R.prop("label")).map(pointToArray));
  return await model.fit(data, labels, {batchSize,epochs:2 });
}


//-----------------------------------------------------------------
// select buttons
function selectMode(a){
  if(a == 0){
    // test mode
    document.getElementById("prediction").disabled = true; 
    console.log("test mode");
    if(keep_data.length > 0){
      trainLoss(keep_data);
      raw_data  = [];
      keep_data = [];
    }
    mode = 1;
  }else{
    // train mode
     document.getElementById("prediction").disabled = false;
    console.log("train mode");
    mode = 2;
  }
}

function selectPrediction(s){
  predicted = s;
}

function selectDrawing(a){
  if(a == 0){
    simulation_flg = false;
  }else{
    simulation_flg = true;
  }
}

// keyTyped
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


//-----------------------------------------------------------------
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