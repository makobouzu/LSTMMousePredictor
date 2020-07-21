let   model;
const position_history_length = 16; //学習に渡すmouseのpositionの長さ
const predict_ahead           = 16; //推論される長さ 16手先まで推論
let   prediction_source       = []; //mouseの軌跡・figureの軌跡
let   raw_data                = []; //train時にxy座標を格納する可変長Array
let   keep_data               = []; //xy座標とlabelを格納するか変調Array
const batch_size              = 32; //一回の学習で回す数
const circle_size             = 10;
let   mode                    = 1; //test mode(1) - train mode(2)
let   simulation_flg          = false; //train時に推論結果を返すかどうか
const myfigure                = myCircle();

//配列変換の便利機能
const pointToArray      = ({x,y})   => [x,y];
const arrayToPoint      = (_array)  => {return {x:_array[0], y:_array[1]};}
let   formatInputTensor = positions => tf.tensor2d(positions.map(pointToArray)).reshape([2, position_history_length]).expandDims();


function setup() {
  frameRate(0); //block draw before init
  init();
  createCanvas(400, 400);
}

async function draw() {
  if(mode == 1){
    //testモード
    background(204, 255, 243);
    //推論結果の取得
    let data = [];
    if(!simulation_flg){
      data = await predict(mouseX, mouseY, prediction_source);
    }else{
      data = await predict(myfigure.next().value.x, myfigure.next().value.y, prediction_source);
    }
    let prediction = data[0];
    prediction_source = data[1];
    //点の描画
    drawPoints(prediction_source, prediction);
    
  }else if(mode == 2){  
    //trainモード
    background(255,180,153);
    //推論結果の取得
    let data = [];
    if(!simulation_flg){
      data = await predict(mouseX, mouseY, prediction_source);
    }else{
      data = await predict(myfigure.next().value.x, myfigure.next().value.y, prediction_source);
    }
    let prediction = data[0];
    prediction_source = data[1];
    //点の描画
    drawPoints(prediction_source, prediction);
    //軌跡のxyデータを保存
    if(raw_data.lengh < 1){
      raw_data = prediction_source.slice();
    }else{
      raw_data = [...raw_data, prediction_source.slice()];
    }
  }
}

/* draw function ---------------------------------- */
function drawPoints(_prediction_source, _prediction){
  //軌跡の描画
  fill(255, 138, 128);
  for(let i = 0; i < _prediction_source.length; i++){
    fill(255, 138, 128);
    ellipse(_prediction_source[i].x * width, _prediction_source[i].y * height, circle_size, circle_size);
  }
  //推論の描画
  fill(128, 149, 255);
  for(let i = 0; i < _prediction.length; i++){
    ellipse(_prediction[i].x * width, _prediction[i].y * height, circle_size, circle_size);
  }
}

/* initialize functinos ----------------------------- */
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

/*  predict ------------------------------------- */
//推論
async function predict(_x, _y, _prediction_source) {
  let mousePos = {x:_x/width, y:_y/height};  
  _prediction_source.push(mousePos);
  _prediction_source.shift();
  
  //データの取得
  let input = _prediction_source.slice();
  let prediction_source = _prediction_source.slice();
  for (let p=0; p < predict_ahead; p++) {
    const predictionNow =  arrayToPoint(await model.predict(formatInputTensor(input)).data());
    input  = [...input, predictionNow].slice(-position_history_length);
  }
  let prediction = input.slice(-predict_ahead);
  return [prediction, prediction_source];  
}

/*  train ------------------------------------- */
//[data:{[ ]}, label:{}]に整形
function formatData(_raw_data){
  const hold_data = _raw_data.slice();
  let train_data = [];
  for(let i = 0; i < hold_data.length - 1; i++){
    let joint_data = {};
    joint_data.data  = hold_data[i]; //dataに格納する軌跡のデータ
    joint_data.label = hold_data[i+1][position_history_length-1]; //16組のdataの最後(その時点の現在位置)
    train_data.push(joint_data);
  }
  return train_data;
}

//train
async function trainLoss(_train_data){ 
  console.log("start training", _train_data.length);
  
  //モデルの設定
  const LEARNING_RATE = 4e-3;
  const optimizer = tf.train.sgd(LEARNING_RATE);
  model.compile({
    optimizer: optimizer, 
    loss: 'meanSquaredError',
    metrics: ['acc']
  });
  
  let train_res = await trainBatch(_train_data);
  console.log("trained", R.last(train_res.history.loss));
}

//use in trainLoss, データの変換とモデルの学習
async function trainBatch(_train_data){
  const data = tf.concat(_train_data.map(R.prop("data")).map(formatInputTensor),0);
  const labels = tf.tensor(_train_data.map(R.prop("label")).map(pointToArray));
  return await model.fit(data, labels, {
    epochs: 30,
    batchSize: batch_size,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        document.getElementById('train-epoch').innerHTML = epoch + 1;
        document.getElementById('train-loss').innerHTML  = logs.loss;
      }
    }
  });
}


/* events ------------------------------------- */
// select buttons
function selectMode(a){
  if(a == 0){
    // test モード
    console.log("test mode");
    
    //保存したデータを元に学習を行う
    if(raw_data.length > 0){
      keep_data = formatData(raw_data).slice();
      trainLoss(keep_data);
      //学習ごとでデータを初期化をする場合
      raw_data  = []; 
      keep_data = [];
    }
    mode = 1;
  }else{
    // train モード
    console.log("train mode");
    mode = 2;
  }
}

function selectDrawing(a){
  if(a == 0){
    simulation_flg = false;
  }else{
    simulation_flg = true;
  }
}

// keyboard Pressed
function keyTyped(){
  if(key === "1"){
    //testモード
    selectMode(0)
    document.getElementById("test").checked = true;
  }else if(key === "2"){
    // trainモード
    document.getElementById("train").checked = true;
    selectMode(1)
  }
  if(key === "s"){
    // 図形の描画
    document.getElementById("auto").checked = true;
    selectDrawing(1);
  }else if(key === "a"){
    // mouseの座標に描画
    document.getElementById("manual").checked = true;
    selectDrawing(0);
  }
}

/* eigure example ------------------------------------- */
// example:1 draw rectangle
function* myRect(){
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

//example:2 draw circle
function* myCircle(){
  let angle = 0;
  let r = 100;
  let x = 0;
  let y = 0;
  let d = -1;
  while(true){
    x = width/2+sin(radians(d*angle)) * r;
    y = height/2+cos(radians(d*angle)) * r;
    
    yield {x: x, y: y}
    angle += 3;
  }
}