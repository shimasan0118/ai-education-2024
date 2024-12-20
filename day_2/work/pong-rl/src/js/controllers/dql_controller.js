// A simple deep Q-learning controller that learns while playing.

import * as tf from '@tensorflow/tfjs';

import BaseController from '../base_controller';
import ReplayMemory from './replay_memory';
import DenseDQN from './dense_dqn';

import _ from 'lodash';

export default class DQLController extends BaseController {
  constructor(leftOrRight, options) {
    options = {
      gamma: 0.99,
      trainingSetMinSize: 40,
      trainingSetMaxSize: 400,
      trainingEpochs: 1,
      trainingIterations: 4,
      lr: 0.001,
      lrDecay: 0.995,
      epsilonInit: 0.5,
      epsilonDecay: 0.98,
      verbose: false,
      ...(options || {}),
    };
    options.modelOptions = {
      nInputs: 6,
      nHiddenLayers: 3,
      nHiddenUnits: 100,
      dropout: 0.1,
      ...(options.dqnOptions || {}),
    };
    options.memoryOptions = {
      capacity: 6000,
      ...(options.memoryOptions || {}),
    };
    super(leftOrRight, options);

    this.replayMemory = this.replayMemory || new ReplayMemory(options.memoryOptions);
    this.model = this.model || new DenseDQN(options.modelOptions);

    this.previousState = null;
    this.previousAction = null;
    this.epsilon = this.epsilonInit;
  }

  // Create a mirrored controller of this controller for self-play.
  // Shares the underlying replay memory and model.
  mirrorController(options) {
    let leftOrRight = 'right';
    if (this.leftOrRight === 'right') this.leftOrRight = 'left';
    options = {
      ...this.options,
      replayMemory: this.replayMemory,
      model: this.model,
      trainingIterations: 0,
      ...(options || {}),
    };
    return new this.constructor(leftOrRight, options);
  }

  // Return the reward for the given state. Simple: +1 when we win, -1 when we lose.
  getReward(state) {
    if (state.winner === this.leftOrRight) return 1;
    else if (state.winner != null) return -1;
    else return 0;
  }

  // Convert a state to an array to be used as input to the DQN model.
  // Contains the ball position and force, as well as both paddle's y positions.
  stateToArray(s, side) {
    side = side || this.leftOrRight;

    const ownPaddle = side === 'left' ? 'leftPaddle' : 'rightPaddle';
    const otherPaddle = side === 'left' ? 'rightPaddle' : 'leftPaddle';
    const ballY = s.ball.y * 2 - 1;
    let ballForceX = s.ball.forceX;
    const ballForceY = s.ball.forceY;

    let ballX = s.ball.x * 2 - 1;
    const ownY = s[ownPaddle].y * 2 - 1;
    const otherY = s[otherPaddle].y * 2 - 1;

    if (side === 'right') {
      // Mirror x-based features
      ballX = -ballX;
      ballForceX = -ballForceX;
    }

    return [ballX, ballY, ballForceX, ballForceY, ownY, otherY];
  }

  // Given a batch of transitions, converts them to an x tensor.
  async transitionsToX(transitions) {
    const x = transitions.map(t => this.stateToArray(t.state, t.side));
    return tf.tensor(x);
  }

  // Given a batch of transitions, returns a y tensor of target values.
  async transitionsToY(transitions) {
    // Get the expected reward for each transition
    const expectedStateActionValues = Array(transitions.length);

    // Pre-fill with "NaNs". We use -10 as a NaN value, which will be filtered out in the loss function.
    const stateExpectationsTensor = tf.mul(tf.ones([transitions.length, 3]), -10);

    // Estimate Q values for resulting states:
    const newStateExpectationsTensor = tf.tidy(() => {
      const newStates = tf.tensor(transitions.map(t => this.stateToArray(t.newState, t.side)));
      return this.model.predict(newStates);
    });

    // Wait for the computations to be done:
    const [stateExpectations, newStateExpectations] = await Promise.all([
      stateExpectationsTensor.array(),
      newStateExpectationsTensor.array(),
    ]);

    tf.dispose([stateExpectationsTensor, newStateExpectationsTensor]);

    for (let i = 0; i < transitions.length; i++) {
      const transition = transitions[i];

      // Bootstrap the target Q values
      const directReward = transition.reward;
      const winner = transition.newState && transition.newState.winner;
      expectedStateActionValues[i] = stateExpectations[i];
      const actionIndex = [-1, 0, 1].indexOf(transition.action);
      const nextStateQ = winner ? 0 : Math.max(...newStateExpectations[i]);
      const target = directReward + this.gamma * nextStateQ;
      expectedStateActionValues[i][actionIndex] = Math.max(-1, Math.min(target, 1));
    }

    return tf.tensor(expectedStateActionValues);
  }

  // Select action given state
  async selectAction(state) {
    const reward = this.getReward(state);
    console.log("===")
    console.log(reward)      
    console.log("===")
    if (this.previousState) {
      // Remember this transition so we can learn from it:
      this.replayMemory.push(
        this.leftOrRight,
        this.previousState,
        this.previousAction,
        state,
        reward,
      );
    }

    // Let the model pick the next action
    let action = 0;

    if (Math.random() < this.epsilon) {
      // Random action:
      if (Math.random() < 0.5) action = -1;
      else action = 1;
    } else {
      // Sample from model predictions:
      const temperature = 0.1 + 2 * this.epsilon;
      action = await this.model.sampleAction(this.stateToArray(state), temperature);
    }

    this.previousState = state;
    this.previousAction = action;

    return action;
  }

  // Train the model
  async trainModel() {
    // Training set should not be bigger than our replay memory:
    const trainingSetSize = Math.round(
      Math.min(this.replayMemory.memory.length, this.trainingSetMaxSize),
    );

    // Let's not train if we didn't collect enough examples yet:
    if (trainingSetSize < this.trainingSetMinSize) return;

    // Train the model
    return new Promise((resolve, reject) => {
      const trainingSet = this.replayMemory.sample(trainingSetSize);

      Promise.all([this.transitionsToX(trainingSet), this.transitionsToY(trainingSet)]).then(
        ([x, y]) => {
          if (this.verbose) {
            const average = data => data.reduce((sum, value) => sum + value) / data.length;
            const standardDeviation = values =>
              Math.sqrt(average(values.map(value => (value - average(values)) ** 2)));

            const p = this.model.predict(x, true);
            const e = tf.abs(tf.sub(y, p));

            const describe = x => {
              return {
                min: tf.min(x).arraySync(),
                max: tf.max(x).arraySync(),
                mean: tf.mean(x).arraySync(),
                std: standardDeviation(_.flatten(x.arraySync())),
              };
            };

            console.table({
              y: describe(y),
              p: describe(p),
              e: describe(e),
            });
          }

          this.model
            .fit(x, y, { epochs: this.trainingEpochs })
            .then(resolve)
            .catch(reject)
            .finally(() => {
              // Clear tensors from memory:
              tf.dispose([x, y]);
            });
        },
      );
    });
  }

  async onMatchEnd(won) {
    this.previousState = null;
    this.previousAction = null;

    // Train model a few times since the default values get updated in each step
    this.model.setLearningRate(this.lr);
    for (let i = 0; i < this.trainingIterations; i++) await this.trainModel();

    // Decay learning rate and epsilon:
    this.lr *= this.lrDecay;
    this.epsilon *= this.epsilonDecay;
  }
}
