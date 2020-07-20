// p5js関数で書き換え
size = ({width, height:500 })
circles = lastPositions.map((pos,element) => svg`<circle cx=${pos.x*size.width} cy=${pos.y*size.height}  r=${circleSize(element)} style='opacity: ${circleOpacity(element)}' />`)
predictionCircles =  predictions.map((pos,element) => svg`<circle cx=${pos.x*size.width} cy=${pos.y*size.height} r=10 fill='red' style='opacity:0.6'/>`)
circleSize = elementNo => 7*(elementNo / positionHistoryLength)+7
circleOpacity = elementNo => 0.5*(elementNo / positionHistoryLength)
view= {
  const res = svg`<svg width=${size.width} height=${size.height} style="border: 1px solid black">${[...circles, ...predictionCircles]}</svg>`
 
  res.addEventListener("mousemove",(e) => { 
    mutable mousePos = {x:e.layerX/size.width,y:e.layerY/size.height};
  })
  return res;
}
mutable mousePos = ({x:0.5, y:0.5})

// そのまま使いたい 
positionHistoryLength = 16
lastPositions = this ? [...this, mousePos].slice(-positionHistoryLength) : new Array(positionHistoryLength).fill(mousePos)
formatInputTensor = positions => tf.tensor2d(positions.map(pointToArray)).reshape([2,positionHistoryLength]).expandDims()
predict = async positions => { 
  const [x,y] = await model.predict(formatInputTensor(positions)).data();
  return {x,y}
}
predictAhead = 8

/// predictionの取得
/// この構文何 !?
predictions =  {
  
  let predictions = [];
  let input = lastPositions;
  let abort = false;
  for (let p=0; p < predictAhead && !abort; p++) {
    invalidation.then(() => abort = true);
    const predictionNow = await predict(input)
    input = [...input, predictionNow].slice(-positionHistoryLength)
    // yield input.slice(0,predictAhead)
  }
  yield input.slice(-predictAhead)  // predict(lastPositions)
}

// script引っ張ってくる
/// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js"></script>
/// tf.___でアクセス可能
tf = require('@tensorflow/tfjs@0.9.0')
corsFetchProxy = (url) => `https://runkit.io/voodoohop/5aec471760d3f3001256a6a3/branches/master?url=${encodeURI(url)}`
model = tf.loadModel("https://raw.githubusercontent.com/voodoohop/Mouse_tracking_predictor/master/tensorflowjs_model/model.json")
model.predict







// script引っ張ってくる
/// <script src="//cdnjs.cloudflare.com/ajax/libs/ramda/0.25.0/ramda.min.js"></script>
/// R.___でアクセス可能
R = require("ramda")
batchSize=32
rawTrainingData = this ? [...this, lastPositions] : [lastPositions] //if文 
//zipwith 二つの配列の値を一つにまとめる
trainingData = R.zipWith((data,rawLabel) => ({data, label: R.last(rawLabel)}), rawTrainingData, rawTrainingData.slice(1))


function shuffle(b) {
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


pickTrainingBatch = (data) => R.take(batchSize, shuffle(data))
pointToArray= ({x,y}) => [x,y]

transpose = a => {
  return R.map(c => {
    return R.map(r => {
      return r[c];
    }, a);
  }, R.keys(a[0]));
}

formatTrainingData = (inputData) => { 
  const data = tf.concat(inputData.map(R.prop("data")).map(formatInputTensor))
  const labels = tf.tensor(inputData.map(R.prop("label")).map(pointToArray))
  return {labels,data}
}

/// この構文何 !? functionじゃないよね？
compiledModel = {
  
  const LEARNING_RATE = 0.015;
const optimizer = tf.train.sgd(LEARNING_RATE);

model.compile({optimizer, loss: 'meanSquaredError'});
return model
}



//trainDataをmodelに投げる
trainBatch =  async (trainData) => {
  const trainingData = formatTrainingData(trainData)
  return await model.fit(trainingData.data, trainingData.labels, {batchSize,epochs:2 })
}




/// この構文何 !? functionじゃないよね？
trainLoss = { 
  console.log("starting",trainingData.length);
  if (!this) yield 1;
             //disabled
  while (false) {
    await Promises.delay(2000);
    if (trainingData.length > batchSize) {
      const trainRes = await trainBatch(pickTrainingBatch(trainingData));
      yield R.last(trainRes.history.loss);
      console.log("trained")
    }
  }
}

//何してんだ？
lossHistory = this ? [...this, trainLoss] : [trainLoss]
{
  const context = DOM.context2d(width, 100);
  context.beginPath();
  context.moveTo(0, 0);
  for (let x = 0; x < lossHistory.length; x++) context.lineTo(x*5, 100*(lossHistory[x]/R.max(...lossHistory)));
  context.lineJoin = context.lineCap = "round";
  context.strokeStyle = "blue";
  context.stroke();
  return context.canvas;
}