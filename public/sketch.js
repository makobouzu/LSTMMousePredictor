let   model;  //現在使用してるmodel
let   models; //使用可能なmodel
let   default_model_url       = "https://raw.githubusercontent.com/voodoohop/Mouse_tracking_predictor/master/tensorflowjs_model/model.json" //default model, you can change value to pretrained model
const position_history_length = 16; //学習に渡すmouseのpositionの長さ
const predict_ahead           = 16; //推論される長さ 16手先まで推論
let   prediction_source       = []; //mouseの軌跡・figureの軌跡
let   raw_data                = []; //train時にxy座標を格納する可変長Array
let   keep_data               = []; //xy座標とlabelを格納するか変調Array
const batch_size              = 32; //一回の学習で回す数
const circle_size             = 10;
let   mode                    = 1; //test mode(1) - train mode(2)
let   simulation_flg          = false; //train時に推論結果を返すかどうか
const figure                  = myCircle();


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
    
    //推論データの格納
    let data = [];
    if(!simulation_flg){
      data = await predict(mouseX, mouseY, prediction_source);
    }else{
      data = await predict(figure.next().value.x, figure.next().value.y, prediction_source);
    }
    let prediction = data[0];
    prediction_source = data[1];

    //円の描画
    drawPoints(prediction_source, prediction);

  }else if(mode == 2){
    //trainモード
    background(255,180,153);
    
    //推論データの格納
    let data = [];
    if(!simulation_flg){
      data = await predict(mouseX, mouseY, prediction_source);
    }else{
      data = await predict(figure.next().value.x, figure.next().value.y, prediction_source);
    }
    let prediction = data[0];
    prediction_source = data[1];
    
    //円の描画
    drawPoints(prediction_source, prediction);

    //軌跡のxyデータを保存
    if(raw_data.lengh < 1){
      raw_data = prediction_source.slice();
    }else{
      raw_data = [...raw_data, prediction_source.slice()];
    }
  }
}

/* initialize functinos ----------------------------- */
async function init() {
  prediction_source = Array(position_history_length).fill({x:1, y:1});
  tf.setBackend('cpu');
  console.log("Backend", tf.getBackend());
  console.log("loading model....");
  model = await tf.loadLayersModel(default_model_url); //v2.0.0
  model.predict;
  console.log("model has been loaded!");
  await modelList(); // add
  frameRate(30);
}

/* model select, save--------------------------------------------*/
async function modelList(){
  models = await doGet("/models/all");
  for(let i = 0; i < models.length ; i++){
    let option = document.createElement("option");
        option.setAttribute("id", models[i]);
        option.setAttribute("value", models[i]);//"/models/"+models[i]
        option.text = models[i];
        document.getElementById("model_select").appendChild(option);
  }
}

async function modelSelect(_model){
  // save fieldに名前を反映
  let model_path = "";
  if(_model == "default_model"){
    document.getElementById("model_name").value = "";
    model_path = default_model_url;
  }else{
    document.getElementById("model_name").value = _model
    model_path = "/models/"+_model+"/model.json";
  }
  // modelの読み込み
  console.log("loading model...", model_path);
  model = await tf.loadLayersModel(model_path); //v2.0.0
  model.predict;
  console.log("model has been loaded!");
}

async function modelSave(_model){
  console.log("saving...")
  let save_name = document.getElementById("model_name").value;
  let save_url = location.href + "models/"+save_name;
  let saveResult = await model.save(save_url);
  console.log(saveResult.responses[0].status);
  if(saveResult.responses[0].status == 200){console.log("saved!")}

  if(models.indexOf(save_name) == -1){
    let option = document.createElement("option");
        option.setAttribute("value", save_name);
        option.setAttribute("id", save_name);
        option.text = save_name;
        option.selected = true;
        document.getElementById("model_select").appendChild(option);
  } else {
    document.getElementById(save_name).selected = true;
  }
}

/* draw function ---------------------------------- */
//円の描画
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
async function trainLoss(_training_data) {
  console.log("start training", _training_data.length);

  //モデルの設定
  const LEARNING_RATE = 0.015;
  const optimizer = tf.train.sgd(LEARNING_RATE);
  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    metrics: ['acc']
  });

  let train_res = await trainBatch(_training_data); //
  console.log("trained");
}

//use in trainLoss, モデルの学習
async function trainBatch(_train_data){
  const data = tf.concat(_train_data.map(R.prop("data")).map(formatInputTensor),0);
  const labels = tf.tensor(_train_data.map(R.prop("label")).map(pointToArray));
  return await model.fit(data, labels, {
    epochs: 30,
    batch_size: batch_size,
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
    if(raw_data.length > 0){
      keep_data = formatData(raw_data).slice();
      trainLoss(keep_data);
      //データの初期化
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
    //test モード
    selectMode(0)
    document.getElementById("test").checked = true;
  }else if(key === "2"){
    // train モード
    document.getElementById("train").checked = true;
    selectMode(1)
  }
  if(key === "s"){
    document.getElementById("auto").checked = true;
    selectDrawing(1);
  }else if(key === "a"){
    document.getElementById("manual").checked = true;
    selectDrawing(0);
  }
}


/* figure example ------------------------------------- */
// example:1 draw rectangle
function* myRect(){
  let i    = 0;
  let x    = 100;
  let y    = 100;
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
  let r     = 100;
  let x     = 0;
  let y     = 0;
  let d     = -1;
  while(true){
    x = width/2+sin(radians(d*angle)) * r;
    y = height/2+cos(radians(d*angle)) * r;

    yield {x: x, y: y}
    angle += 2;
  }
}


/*  dataset (json, get, post) ------------------------------------- */
async function doGet(_url) {
  return new Promise(resolve =>{
    var xhr = new XMLHttpRequest();
    xhr.open("GET", _url);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onload = () => {
      resolve(JSON.parse(xhr.response));
    };
    xhr.onerror = () => {
     console.log(xhr.status);
     console.log("error!");
    };
    xhr.send();
  })
}

async function doPost(_url, _data) {
  return new Promise(resolve =>{
    var xhr = new XMLHttpRequest();
    var jsonText = JSON.stringify(_data);
    xhr.open("POST", _url);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onload = () => {
     resolve("saved");
    };
    xhr.onerror = () => {
     resolve(xhr.status);
     console.log("error!");
    };
    xhr.send(jsonText);
  })
}
