// server.js
// where your node app starts

// we've started you off with Express (https://expressjs.com/)
// but feel free to use whatever libraries or frameworks you'd like through `package.json`.
const express = require("express");
const cors = require('cors');
const fs = require('fs');
var multer  = require('multer');

var storage = multer.diskStorage({
    destination: function (req, file, cb) {
      let dir = __dirname+"/public"+req.path
      if (!fs.existsSync(dir)){
        fs.mkdirSync(dir);
      }
      cb(null, __dirname+"/public"+req.path)
    },
    filename: function (req, file, cb) {
    cb(null, file.originalname)
  }
})

var upload = multer({ storage: storage })
const app = express();

app.use(cors());


// make all the files in 'public' available
// https://expressjs.com/en/starter/static-files.html
app.use(express.static("public"));

// https://expressjs.com/en/starter/basic-routing.html
app.get("/", (request, response) => {
  response.sendFile(__dirname + "/views/index.html");
});

app.get("/models/all", function (request, response) {
  fs.readdir(__dirname + '/public/models', (err, files) => {
    response.send(files)
  });
})

app.post('/models/*', upload.any(), function(req, res) { //upload.single('model')
  res.send('success!');
});


// listen for requests :)
const listener = app.listen(process.env.PORT, () => {
  console.log("Your app is listening on port " + listener.address().port);
});
