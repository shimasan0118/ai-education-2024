// -*- coding: utf-8 -*-
// Represents a single match. Updates and keeps the game state. Draws to a canvas.

import { sleep } from './util';

export default class Match {
  constructor(options) {
    options = {
      paddleHeight: 0.25,
      canvasId: 'gameCanvas',

      // If false, doesn't draw and goes as fast as possible
      live: true,

      // How often the game should be updated / redrawn
      updateFrequency: 40, // = 25 FPS
      // Ask controllers every X frames for an updated action:
      controllerFrameInterval: 5, // 25 FPS / 5 = 5 updates per second

      // How fast the paddles and the ball can move
      paddleSpeed: 1,
      ballSpeed: 0.8,
      ballSpeedIncrease: 1.01,
      ballSpeedMax: 2,

      // How strongly the image is downscaled for visual controllers
      visualDownscalingFactor: 10,     

      ...options,
    };

    Object.assign(this, options);

    // Find the canvas and get its drawing context.
    this.canvas = document.getElementById(this.canvasId);
    this.ctx = this.canvas.getContext('2d');
      
    // プレイヤー名を初期化
    this.playerAName = 'CPU';
    this.playerBName = '';      
      
    // オプションから試合数を取得
    this.matchCount = options.matchCount || 0;
      
    // AIとのバトルが開始されたかどうかを追跡するフラグ
    this.startedAIBattle = false;            

    // How much time has passed at each update. Fixed so we get same results
    // on every machine.
    this.timeFactor = this.updateFrequency / 1000;

    // Keep track of the ball and two paddles.
    this.leftPaddle = {
      x: 0.02,
      y: 0.5,
      height: this.paddleHeight,
      width: 0.0375,
      forceY: 0,
      previousAction: null,
      speed: this.paddleSpeed,
    };
    this.rightPaddle = {
      x: 0.98,
      y: 0.5,
      height: this.paddleHeight,
      width: 0.0375,
      forceY: 0,
      previousAction: null,
      speed: this.paddleSpeed,
    };
    this.ball = {
      x: 0.5,
      y: 0.5,
      height: 0.05,
      width: 0.0375,
      forceX: 0,
      forceY: 0,
      speed: this.ballSpeed,
    };

    // If any of the controllers require a visual state, this whole class needs to keep track of it
    this.isVisual = this.leftController.isVisual || this.rightController.isVisual;

    // Start the ball in a random direction.
    const forceX = 0.5 + Math.random() * 0.25;
    const forceY = 0.9 + Math.random() * 0.25;
    const norm = Math.sqrt(Math.pow(forceX, 2) + Math.pow(forceY, 2));
    this.ball.forceX = ((Math.random() > 0.5 ? 1 : -1) * forceX) / norm;
    this.ball.forceY = ((Math.random() > 0.5 ? 1 : -1) * forceY) / norm;

    // Keep track of the last two game states
    this.currentState = this.getState();
    this.previousState = null;
    this.currentFrame = 0;
    this.winner = null;
      
    this.battleMode = false

    // コンストラクタ内で得点を初期化
    this.leftScore = 0;
    this.rightScore = 0;
    this.scoreLimit = 5; // 5点先取

    this.live && this.draw();
  }

  // Given a difficulty level from 1 to 3, generates an options object to instantiate
  // a new Match object with.
  static createOptions(difficulty) {
    let q = 1;
    if (difficulty === 2) q = 1.2;
    if (difficulty === 3) q = 1.7;
    const nq = 1 / q;

    let ballSpeedIncrease = 1.05;
    if (difficulty === 2) ballSpeedIncrease = 1.1;
    if (difficulty === 3) ballSpeedIncrease = 1.2;

    return {
      paddleHeight: 0.26 * nq,
      paddleSpeed: 1.5 * nq,
      ballSpeed: 0.7 * q,
      ballSpeedMax: 1.25 * q,
      ballSpeedIncrease,
    };
  }

  // Takes a snapshot of the game canvas.
  // Returns a 2D array of values in [0, 1] describing the current visual state of the game.
  // Optionally pass a factor by which it should be scaled down, e.g. 2 to halve the size.
  getImageData(downscalingFactor) {
    downscalingFactor = downscalingFactor || this.visualDownscalingFactor;
    const downscaledCubeSize = downscalingFactor * downscalingFactor;
    const width = this.canvas.width;
    const height = this.canvas.height;

    // This gets the raw RGB data. Still needs to be converted to grayscale and scaled.
    const rawData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);

    // Create an array full of zeros for the output
    const data = new Array(width / downscalingFactor);
    for (let i = 0; i < width / downscalingFactor; i++) {
      data[i] = new Array(height / downscalingFactor);
      data[i].fill(0);
    }

    // Go through each pixel in the raw data and set the corresponding pixel in the output data
    let x = 0;
    let y = 0;
    for (let i = 0; i < rawData.data.length; i += 4) {
      const r = rawData.data[i];
      const g = rawData.data[i + 1];
      const b = rawData.data[i + 2];
      const a = rawData.data[i + 3];
      const targetX = Math.floor(x / downscalingFactor);
      const targetY = Math.floor(y / downscalingFactor);
      const value = (Math.max(r, g, b) / 255) * (a / 255);

      data[targetX][targetY] += value / downscaledCubeSize;

      x += 1;
      if (x >= width) {
        x = 0;
        y += 1;
      }
    }

    return data;
  }
    
  startAIBattle() {
    this.startedAIBattle = true;
  }  
    
  updateEnemyName() {         
    if (this.matchCount <= 50) {
      this.playerBName = 'スライム';
    } else if (this.matchCount <= 100) {
      this.playerBName = 'スカル';
    } else if (this.matchCount <= 150) {
      this.playerBName = 'ゴブリン';
    } else if (this.matchCount <= 200) {
      this.playerBName = 'ドラゴン';        
    } else {
      this.playerBName = 'デビル';
    }
  }
    
  // 現在の敵キャラクターの名前を取得するメソッド
  getCurrentEnemyName() {
    if (this.matchCount <= 50) {
      return 'スライム';
    } else if (this.matchCount <= 100) {
      return 'スカル';
    } else if (this.matchCount <= 150) {
      return 'ゴブリン';
    } else if (this.matchCount <= 200) {
      return 'ドラゴン';        
    } else {
      return 'デビル';
    }
  }    

  // バトル終了時のコールバック設定
  onEnd(callback) {
    this.endCallback = callback;
  }

  // Return the current state of the game.
  getState() {
    const state = {
      ball: {
        x: this.ball.x,
        y: this.ball.y,
        forceX: this.ball.forceX * this.ball.speed,
        forceY: this.ball.forceY * this.ball.speed,
      },
      leftPaddle: {
        x: this.leftPaddle.x,
        y: this.leftPaddle.y,
      },
      rightPaddle: {
        x: this.rightPaddle.x,
        y: this.rightPaddle.y,
      },
      winner: this.getWinner(),
      timePassed: this.currentFrame * this.timeFactor,
    };

    if (this.isVisual) {
      // Add image data to state.
      state.imageData = this.getImageData();
      state.previousImageData = this.previousState && this.previousState.imageData;
    }

    return state;
  }
    
  setBattleMode() {
      this.battleMode = true
  }

  // ボールをリセットするメソッド
  resetBall() {
    this.ball.x = 0.5;
    this.ball.y = 0.5;
    const forceX = 0.5 + Math.random() * 0.25;
    const forceY = 0.9 + Math.random() * 0.25;
    const norm = Math.sqrt(Math.pow(forceX, 2) + Math.pow(forceY, 2));
    this.ball.forceX = ((Math.random() > 0.5 ? 1 : -1) * forceX) / norm;
    this.ball.forceY = ((Math.random() > 0.5 ? 1 : -1) * forceY) / norm;
  }    
    
  // Check if the given side's paddle is colliding with the ball.
  // Pass 'left' or 'right' (since the logic is slightly different)
  checkCollision(leftOrRight) {
    const paddle = leftOrRight === 'left' ? this.leftPaddle : this.rightPaddle;
    const ball = this.ball;

    const paddleWidth = paddle.width;
    const paddleHeight = paddle.height + 0.01;
    const ballWidth = ball.width;
    const ballHeight = ball.height;

    // First, check on the x dimension if a collision is possible:
    if (leftOrRight === 'left' && ball.x - ballWidth / 2 > paddle.x + paddleWidth / 2) {
      // It's too far from the left paddle
      return false;
    }
    if (leftOrRight === 'right' && ball.x + ballWidth / 2 < paddle.x - paddleWidth / 2) {
      // It's too far from the right paddle
      return false;
    }

    // Now check on the y dimension:
    if (ball.y - ballHeight / 2 > paddle.y + paddleHeight / 2) {
      // The top of the ball is below the bottom of the paddle
      return false;
    }
    if (ball.y + ballHeight / 2 < paddle.y - paddleHeight / 2) {
      // The bottom of the ball is above the top of the paddle
      return false;
    }

    // Check if its too far behind the paddle
    if (leftOrRight === 'left' && ball.x - ballWidth / 2 < paddle.x - paddleWidth / 2) {
      // It's past the left paddle
      return false;
    }
    if (leftOrRight === 'right' && ball.x + ballWidth / 2 > paddle.x + paddleWidth / 2) {
      // It's past the right paddle
      return false;
    }

    return true;
  }

  // Move the given object by its force, checking for collisions and potentially
  // updating the force values. If the ball, returns whether it was hit by a paddle.
  moveObject(obj, timeFactor, isBall) {
    const radiusY = obj.height / 2;
    const minY = radiusY;
    const maxY = 1 - radiusY;
    let wasHit = false;

    // If a paddle is already touching the wall, forceY should set to zero:
    if (!isBall && obj.forceY) {
      if ((obj.y === minY && obj.forceY < 0) || (obj.y === maxY && obj.forceY > 0)) {
        obj.forceY = 0;
      }
    }

    if (obj.forceX) {
      obj.x += obj.forceX * obj.speed * timeFactor;

      // A ball should bounce off paddles
      const sideToCheck = obj.forceX > 0 ? 'right' : 'left';
      if (isBall && this.checkCollision(sideToCheck)) {
        obj.forceX = -obj.forceX;
        wasHit = true;

        // Add a spin to it:
        const paddle = this[`${sideToCheck}Paddle`];
        if (paddle.forceY !== 0) {
          obj.forceY = (obj.forceY + paddle.forceY) / 2;
          // Make mean spins a little harder:
          if (Math.abs(obj.forceY) < 0.33) obj.forceY *= 2;
          // Re-normalize it:
          const norm = Math.sqrt(Math.pow(obj.forceX, 2) + Math.pow(obj.forceY, 2));
          obj.forceX /= norm;
          obj.forceY /= norm;
        }
      }
    }

    if (obj.forceY) {
      obj.y += obj.forceY * obj.speed * timeFactor;

      // When hitting a wall, a paddle stops, a ball bounces back:
      if (!isBall) {
        obj.y = Math.max(minY, Math.min(maxY, obj.y));
      } else if ((obj.forceY < 0 && obj.y < radiusY) || (obj.forceY > 0 && obj.y > 1 - radiusY)) {
        obj.forceY = -obj.forceY;
      }
    }

    return wasHit;
  }

  // getWinner メソッド内で得点を更新せず、単に勝者を確認する
  getWinner() {
    const ballWidth = this.ball.width / 2;
    const paddleWidth = this.leftPaddle.width / 2;

    if (this.ball.forceX < 0 && this.ball.x - ballWidth < this.leftPaddle.x - paddleWidth) {
      return 'right';
    }
    if (this.ball.forceX > 0 && this.ball.x + ballWidth > this.rightPaddle.x + paddleWidth) {
      return 'left';
    }
  }


  // Moves objects, checks for collisions, etc.
  async update() {
    this.previousState = this.currentState;
    this.currentState = this.getState();

    // Check if match ended:
    const winner = this.currentState.winner;     
    if (winner) this.winner = winner;

    // Ask controllers for action based on current state.
    // Either every few frames or if there's a winner (to give them a chance to register the win)
    let leftAction = this.leftPaddle.lastAction || 0;
    let rightAction = this.rightPaddle.lastAction || 0;

    if (this.currentState.winner || this.currentFrame % this.controllerFrameInterval === 0) {
      if (this.leftController)
        leftAction = await this.leftController.selectAction(this.currentState);
      if (this.rightController)
        rightAction = await this.rightController.selectAction(this.currentState);
    }

    this.leftPaddle.forceY = leftAction;
    this.rightPaddle.forceY = rightAction;

    this.leftPaddle.lastAction = leftAction;
    this.rightPaddle.lastAction = rightAction;

    // Update each object:
    this.moveObject(this.leftPaddle, this.timeFactor);
    this.moveObject(this.rightPaddle, this.timeFactor);
    const ballWasHit = this.moveObject(this.ball, this.timeFactor, true);

    if (ballWasHit) {
      // Increase ball speed
      this.ballSpeed = Math.min(this.ballSpeedMax, this.ballSpeed * this.ballSpeedIncrease);
      this.ball.speed = this.ballSpeed;
    }

    //this.currentFrame += 1;

    // 勝者を確認し、得点を加算
    //const gameWinner = this.getWinner();
    if (this.winner) {
      if (this.winner === 'left') {
        this.leftScore++;
      } else {
        this.rightScore++;
      }
      this.ballSpeed = 0.8
      this.ball.speed = this.ballSpeed;        
      // ボールをリセット
      this.resetBall();
        
      // スコアリミットに到達してなければマッチを終了させないために、winnerをnullにする
      if (this.leftScore < this.scoreLimit && this.rightScore < this.scoreLimit) { 
        if (this.battleMode){
          // 0.3秒待機
           await this.waitForSecond();            
          this.winner = null;
        }
      }
      else {
        // 勝者の表示を保証するために、drawメソッドを呼び出す
        await this.draw();

        // バトル終了時のコールバックを呼び出し
        if (this.endCallback) {
          this.endCallback();
        }          
      }
    }
    this.currentFrame += 1;
  }

  waitForSecond() {
    return new Promise(resolve => {
      setTimeout(() => {
        resolve();
      }, 300);
    });
  }    

  // Given an object with coordinates and size, draw it to the canvas
  drawObject(obj) {
    const width = obj.width * this.canvas.width;
    const height = obj.height * this.canvas.height;
    const x = obj.x * this.canvas.width - width / 2;
    const y = obj.y * this.canvas.height - height / 2;
    this.ctx.fillRect(x, y, width, height);
  }

  // Matchクラス内に追加
  checkCompletion() {
    const allDefeated = Array.from(document.querySelectorAll('#defeatList li span.defeated')).length === 5;

    if (allDefeated) {
      const popup = document.getElementById('popup');
      popup.classList.add('active');

      const closeButton = document.getElementById('popupCloseButton');
      closeButton.addEventListener('click', () => {
        popup.classList.remove('active');
      });
    }
  }
    
    
  // 討伐リストを更新するメソッド
  updateDefeatList() {
    let enemyId;
    if (this.matchCount <= 50) {
      enemyId = 'slime';        
    } else if (this.matchCount <= 100) {
      enemyId = 'skeleton';
    } else if (this.matchCount <= 150) {
      enemyId = 'goblin';
    } else if (this.matchCount <= 200) {
      enemyId = 'dragon';        
    } else {
      enemyId = 'devil';
    }

    const enemyElement = document.getElementById(enemyId);
    if (enemyElement) {
      const enemyName = enemyElement.textContent.split(':')[0];
      const content = enemyElement.innerHTML;
      enemyElement.innerHTML = content.replace('未討伐', '<span class="defeated">討伐！</span>');
      enemyElement.classList.add('defeated');
      
      // 討伐状況をlocalStorageに保存
      const savedStatus = JSON.parse(localStorage.getItem('defeatStatus')) || {};
      savedStatus[enemyId] = '討伐！';
      localStorage.setItem('defeatStatus', JSON.stringify(savedStatus));        
    }
    // 討伐リスト更新後にクリアチェックを実行
    this.checkCompletion();      
  } 

  // Redraw the game based on the current state
  async draw() {
    this.ctx.fillStyle = '#e5e5e6';
    this.ctx.strokeStyle = '#e5e5e6';
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.strokeRect(0, 0, this.canvas.width, this.canvas.height);

    // ボールとパドルの描画
    this.drawObject(this.ball);
    this.drawObject(this.leftPaddle);
    this.drawObject(this.rightPaddle);

    // 得点の表示
    this.ctx.font = '16px Arial';
    const scoreAText = `${this.playerAName}: ${this.leftScore}`;
    const scoreBText = `${this.playerBName}: ${this.rightScore}`;      

    // テキストの幅を計算
    const scoreATextWidth = this.ctx.measureText(scoreAText).width;
    const scoreBTextWidth = this.ctx.measureText(scoreBText).width;

    // キャンバスの左半分の中央にプレイヤーAの得点を配置
    const xA = (this.canvas.width / 4) - (scoreATextWidth / 2);
    this.ctx.fillText(scoreAText, xA, 20);

    // キャンバスの右半分の中央にプレイヤーBの得点を配置
    const xB = (3 * this.canvas.width / 4) - (scoreBTextWidth / 2);
    this.ctx.fillText(scoreBText, xB, 20);

    // 勝者の表示
    if (this.winner) {     
      const winText = `${this.winner === 'left' ? this.playerAName : this.playerBName} の勝利!`;
      this.ctx.font = '24px Arial';

      // テキストの幅を計算
      const textWidth = this.ctx.measureText(winText).width;

      // キャンバスの中央にテキストを配置
      const x = (this.canvas.width - textWidth) / 2;
      const y = this.canvas.height / 2;
      this.ctx.fillStyle = '#f06543';
      this.ctx.fillText(winText, x, y);
      this.ctx.fillStyle = '#e5e5e6';
    }

    // アニメーションフレームをリクエスト
    return new Promise(resolve => {
      window.requestAnimationFrame(resolve);
    });
  }

  // Call periodically. Will update the state and draw every few frames
  async updateAndDraw() {
    await this.update();
    // Only draw or update stats when live or once in a while:
    if (this.live || Math.random() < 0.01) {
      await Promise.all([
        this.draw(),
        this.stats && this.stats.onFrame(this.currentFrame * this.updateFrequency),
      ]);
    }
  }    

  // Starts the game and runs until completion.
  async run() {
    let updateInProgress = false;
      
    const updateFrequency = this.live ? this.updateFrequency : 1;

    return new Promise((resolve, reject) => {
      this.leftController && this.leftController.onMatchStart();
      this.rightController && this.rightController.onMatchStart();

      const updateInterval = setInterval(() => {
        if (updateInProgress) return;

        let error = null;
        updateInProgress = true;

        this.updateAndDraw()
          .then(() => {
            updateInProgress = false;
          })
          .catch(e => {
            error = e;
            console.error(e);
          })
          .finally(() => {
            // Check if the match is finished or there was an error
            if (error) {
              clearInterval(updateInterval);
              reject(error);
            } else if (this.winner) {
              clearInterval(updateInterval);
              Promise.all([
                this.stats &&
                  this.stats.onMatchEnd(this.winner, this.currentFrame * this.updateFrequency),
                this.live && sleep(250),
                this.leftController && this.leftController.onMatchEnd(this.winner === 'left'),
                this.rightController && this.rightController.onMatchEnd(this.winner === 'right'),
              ]).then(() => {
                // プレイヤーが勝利し、AIとのバトルが開始されていた場合に討伐リストを更新
                if (this.winner === 'left' && this.startedAIBattle) {
                  this.updateDefeatList();
                }
              resolve(this.winner);});
            }
          });
      }, updateFrequency);
    });
  }
}
