// -*- coding: utf-8 -*-
import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';
import { CANVAS_WIDTH, CANVAS_HEIGHT } from './game/constants';
import { Runner } from './game';

const START_BUTTON = document.getElementById('start');
const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];

let runner = null;
// initial setup for the game the  setup function is called when the dom gets loaded

let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
const trainingDataInputs = [];
const trainingDataOutputs = [];
const examplesCount = [];
let predict = false;
let mobilenet;

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

// Call the function immediately to start loading.
loadMobileNetFeatureModel();

const dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i+=1) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));

model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  // Adam changes the learning rate over time which is useful.
  optimizer: 'adam',
  // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
  // Else categoricalCrossentropy is used if more than 2 classes.
  loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy',
  // As this is a classification problem you can record accuracy in the logs too!
  metrics: ['accuracy']
});

/**
 * Loads the MobileNet model and warms it up so ready for use.
 * */
async function loadMobileNetFeatureModel() {
  const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  STATUS.innerText = 'AIの読み込みに成功しました';

  // Warm up the model by passing zeros through it once.
  tf.tidy(() => {
    const answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
  if (hasGetUserMedia()) {
    // getUsermedia parameters.
    const constraints = {
      video: true,
      width: 640,
      height: 480
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      VIDEO.srcObject = stream;
      VIDEO.addEventListener('loadeddata', () => {
        videoPlaying = true;
        ENABLE_CAM_BUTTON.classList.add('removed');
      });
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}

async function trainAndPredict() {
  STATUS.innerText = 'AIの学習中...';
  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  const outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  const oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  const inputsAsTensor = tf.stack(trainingDataInputs);

  const results = await model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epochs: 10,
      callbacks: {onEpochEnd: logProgress} });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  predict = true;
  predictLoop();
}

function logProgress(epoch, logs) {
  console.log(`Data for epoch ${epoch}, ${logs}`);
}

function predictLoop() {
  if (predict) {
    tf.tidy(() => {
      const videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
      const resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);

      const imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
      const prediction = model.predict(imageFeatures).squeeze();
      const highestIndex = prediction.argMax().arraySync();
      const predictionArray = prediction.arraySync();

      STATUS.innerText = `予測: ${CLASS_NAMES[highestIndex]} 確率: ${Math.floor(predictionArray[highestIndex] * 100)}%`;

    });

    window.requestAnimationFrame(predictLoop);
  }
}

/**
 * Purge data and start over. Note this does not dispose of the loaded
 * MobileNet model and MLP head tensors as you will need to reuse
 * them to train a new model.
 * */
function reset() {
  predict = false;
  examplesCount.length = 0;
  for (let i = 0; i < trainingDataInputs.length; i+=1) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.length = 0;
  trainingDataOutputs.length = 0;
  STATUS.innerText = 'リセットしました';

  console.log(`Tensors in memory: ${tf.memory().numTensors}`);
}

/**
 * Handle Data Gather for button mouseup/mousedown.
 * */
function gatherDataForClass() {
  const classNumber = parseInt(this.getAttribute('data-1hot'), 10);
  gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

function dataGatherLoop() {
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    const imageFeatures = tf.tidy(() => {
      const videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
      const resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);
      const normalizedTensorFrame = resizedTensorFrame.div(255);
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);

    // Intialize array index element if currently undefined.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    examplesCount[gatherDataState]+=1;

    STATUS.innerText = '';
    for (let n = 0; n < CLASS_NAMES.length; n += 1) {
      STATUS.innerText += `${CLASS_NAMES[n]}時のデータ ${examplesCount[n]} 枚  `;
    }
    window.requestAnimationFrame(dataGatherLoop);
  }
}

function setup() {
  // Initialize the game Runner.
  runner = new Runner('.game', {
    DINO_COUNT: 1,
    onReset: handleReset,
    onCrash: handleCrash,
    onRunning: handleRunning
  });
  // Set runner as a global variable if you need runtime debugging.
  window.runner = runner;
  // Initialize everything in the game and start the game.
  runner.init();
  
  // Disable the start button after the first click
  START_BUTTON.disabled = true;
}


function handleReset(dinos) {
  // running this for single dino at a time
  // console.log(dinos);
  
  const dino = dinos[0]; 
  // if the game is being started for the first time initiate 
  // the model and compile it to make it ready for training and predicting
}

/**
 * documentation
 * @param {object} dino
 * @param {object} state
 * returns a promise resolved with an action
 */

function handleRunning( dino, state ) {
  return new Promise((resolve) => {
    if (!dino.jumping) {
      // whenever the dino is not jumping decide whether it needs to jump or not
      let action = 0;// variable for action 1 for jump 0 for not
    
      // predictionArrayを関数の外側で宣言
      let predictionArray;
        
      tf.tidy(() => {
        const videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
        const resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);

        const imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
        const prediction = model.predict(imageFeatures).squeeze();
        const highestIndex = prediction.argMax().arraySync();

        // tf.tidyブロック内でpredictionArrayを更新
        predictionArray = prediction.arraySync();

        STATUS.innerText = `Prediction: ${CLASS_NAMES[highestIndex]} with ${Math.floor(predictionArray[highestIndex] * 100)}% confidence`;
      });
      
      if (predictionArray[1] > predictionArray[0]) {
        // we want to jump
        action = 1;
        // set last jumping state to current state
        dino.lastJumpingState = state;
      } else {
        // set running state to current state
        dino.lastRunningState = state;
      }
      resolve(action);
    } else {
      resolve(0);
    }
  });
}

/**
 * 
 * @param {object} dino 
 * handles the crash of a dino before restarting the game
 * 
 */
function handleCrash( dino ) {
  const input = null;
  const label = null;
}

/**
 * 
 * @param {object} state
 * returns an array 
 * converts state to a feature scaled array
 */
function convertStateToVector(state) {
  if (state) {
    return [
      state.obstacleX / CANVAS_WIDTH,
      state.obstacleWidth / CANVAS_WIDTH,
      state.speed / 100
    ];
  }
  return [0, 0, 0];
}

// call setup on loading content
// document.addEventListener('DOMContentLoaded', setup);
START_BUTTON.addEventListener('click', setup);
