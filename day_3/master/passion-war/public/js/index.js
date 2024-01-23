//globals
let socket = io()

//generate random room name
if (!location.hash) {
  location.hash = Math.floor(Math.random() * 0xFFFFFF).toString(16)
}
const roomHash = location.hash.substring(1)
const webcam = new Webcam(document.getElementById('webcam'))
let emotions = ["angry","disgust","scared","happy","sad","surprised","neutral"]
let model


//socket events
socket.on('start rtc', function(isOfferer){ startWebRTC(isOfferer) })
socket.on('respond offer', function(message){ respondOffer(message)})
socket.on('add candidate', function(message){ addCandidate(message) })
//TODO improve the player disconnection
socket.on('player left', function(){ location.reload() })
socket.on('room ready', function(){ displayStep02() })
socket.on('next level', function(){ nextEmotion() })
socket.on('player score', function(score){ updateOpponentScore(score) })
socket.on('set random', function(emotionIndices, intervals){setRandomIndexList(emotionIndices, intervals)})

//on click events
$('.bu-ready').click(function(){
  $(this).addClass('ready')
  socket.emit('player ready')
})

$('.bu-play-again').click(function(){
  location.reload()
})


init()

async function init() {
  try {
    await webcam.setup()
  } catch (e) {
    console.log(e)
  }

  model = await tf.loadModel('models/model.json')

  displayStep01()

  socket.emit('room join', roomHash)

  isPredicting = true
  predict()
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      //capture image from webcam
      const img = webcam.capture()
      //predict
      return model.predict(img)
      //return predictions.as1D().argMax()
    })

    const predictions = await predictedClass.data()

    if (g_isGameStarted) {
      let index = emotions.indexOf(g_emotionsList[g_emotionIndex])
      g_emotionLevel.css('width', Math.round(predictions[index] * 100))

      g_myScore = g_myScore + (predictions[index] * 0.2)
      //console.log("~~~")
      //console.log(predictions)
      //console.log("~~~")        
      g_myScoreLevel.css('width', (g_myScore*0.5)+'%')
      g_myScoreLabel.html(Math.round(g_myScore))

      //send score to server
      socket.emit('player score', g_myScore)
    }

    predictedClass.dispose()

    await tf.nextFrame()
  }
}

let g_emotionsList = ["happy","sad","angry","scared","disgust", "surprised"]
let g_emotionIndex = 0
let emotion_count = 0
let g_myScore = 0
let preEmotionIndex = -1
let g_isGameStarted = false
let emotionIndicesList = []
let emotionIndicesIdx = 0
let intervalsList = []
let currentIntervalsIdx = 0

let g_emotionIcon = $('.current-emotion .icon')
let g_emotionLevel = $('.feed-info .jauge .level')
let g_myScoreLevel = $('.scores .me .level')
let g_myScoreLabel = $('.scores .me .label')
let g_oppScoreLevel = $('.scores .opponent .level')
let g_oppScoreLabel = $('.scores .opponent .label')

let countdown = {
  interval: null,
  duration : 61,
  startTime: null,
  nextEmotionChange: 0,    
  Start : function() {
    g_isGameStarted = true;
    this.startTime = Date.now();
    this.setNextEmotionChange();      

    this.interval = setInterval(() => {
      this.updateTimer();     
    }, 100);
  },
  updateTimer: function() {
    let elapsedTime = Date.now() - this.startTime;
    let remainingTime = Math.max(this.duration * 1000 - elapsedTime, 0);
    let seconds = Math.floor(remainingTime / 1000);
    $('.timer .seconds').html(seconds);
  
    // 感情をランダムな時間で切り替える
    if (elapsedTime >= this.nextEmotionChange) {
      this.setNextEmotionChange(elapsedTime);
      nextEmotion(); // 次の感情に切り替える
    }
  
    if (remainingTime <= 0) {
      this.Stop();
    }
  },  
  setNextEmotionChange: function(currentTime = 0) {
    let randomInterval = intervalsList[currentIntervalsIdx]
    currentIntervalsIdx++
    this.nextEmotionChange = currentTime + randomInterval;
  },    
  Stop: function(){
    clearInterval(this.interval);
    this.interval = null;
    g_isGameStarted = false;

    //g_emotionIndex++;

    console.log('index', g_emotionIndex);

    //if (g_emotionIndex <= g_emotionsList.length - 1) {
    //  socket.emit('player ready')
    //}else{
    displayFinalScore()
    //}
  }
}

function setRandomIndexList(emotionIndices, intervals){
    emotionIndicesList = emotionIndices
    intervalsList = intervals
    console.log(emotionIndicesList)
    console.log(intervalsList)    
}

function displayStep01(){
  $('.step-01').show()
  $('.step-01 .room-link .link').html(location.href)
}
function displayStep02(){
  $('.step-01').hide()
  $('.step-02').show()
}
function displayFinalScore(){
  let me = Number(g_myScoreLabel.html())
  let opponent = Number(g_oppScoreLabel.html())

  $('.timer').hide()
  $('.video-feed').addClass('timeout')
  $('.feed-info').hide()
  $('.current-emotion').hide()

  if (me >= opponent) { //I won
    $('.scores-result img').attr('src', '/images/win-message.png')
  }
  $('.scores-result').show()
}

function nextEmotion(){
  if (emotion_count == 0){
      countdown.Start();
      emotion_count++;
  }

  // ランダムに感情のインデックスを選択
  g_emotionIndex = emotionIndicesList[emotionIndicesIdx]
  emotionIndicesIdx++

  // 感情のアイコンを更新
  g_emotionIcon.attr('src', '/images/block-' + g_emotionsList[g_emotionIndex] + '.png');

  //hide ready button and display timer
  $('.bu-ready').hide()
  $('.timer').show()
}

function updateOpponentScore(score){
  g_oppScoreLevel.css('width', (score*0.5)+'%')
  g_oppScoreLabel.html(Math.round(score))
}


//fix the zoom of the camera
navigator.mediaDevices.getUserMedia({video: true})
.then(async mediaStream => {
  document.querySelector('video').srcObject = mediaStream
  await sleep(1000)

  const track = mediaStream.getVideoTracks()[0]
  const capabilities = track.getCapabilities()
  const settings = track.getSettings()

  track.applyConstraints({advanced: [ {zoom: 2} ]})
}).catch(error => console.log(error))

function sleep(ms = 0) {
  return new Promise(r => setTimeout(r, ms))
}
