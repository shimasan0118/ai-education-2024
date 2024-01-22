import '../scss/style.scss';

import Menu from './menu';
import Match from './match';
import Stats from './stats';

import { sleep } from './util';

import KeyController from './controllers/key_controller';
import DumbController from './controllers/dumb_controller';
import DQLController from './controllers/dql_controller';
import VisualDQLController from './controllers/visual_dql_controller';

const controllers = {
  DQLController,
  KeyController,
};

let matchCount = 0;  // 試合数を追跡する変数

$(document).ready(async () => {
  Menu.init(controllers);
  const matchOptions = await Menu.run();
  let liveMode = true;

  $('#liveDropdown').change(event => {
    liveMode = event.currentTarget.value === 'Yes';
  });

  // 手動バトル開始ボタンのイベントリスナーを追加
  $('#startBattleButton').click(() => {
    startBattle();
  });

  await sleep(500);
  $('#menu').remove();
  $('#game').addClass('active');
  $('#defeatList').addClass('active');

  matchOptions.stats = new Stats();

  for (;;) {
    matchCount++;      
    window.currentMatch = new Match({
      ...matchOptions,
      live: liveMode,
      matchCount: matchCount,
    });
      
    // 敵キャラクターの名前を更新
    window.currentMatch.updateEnemyName();
      
    // ボタンのテキストを更新
    const enemyName = window.currentMatch.getCurrentEnemyName();
    const battleButton = document.getElementById('startBattleButton');
    battleButton.textContent = `${enemyName}とバトル`;      
      
    await window.currentMatch.run();
      
  }
});

function startBattle() {
  // 現在のMatchインスタンスを取得
  const currentMatch = window.currentMatch;

  if (currentMatch) {
    // 現在の左側のコントローラーを保存
    const originalLeftController = currentMatch.leftController;

    // Player AをKeyControllerに切り替え
    currentMatch.leftController = new KeyController('left');
      
    currentMatch.startAIBattle();      

    // プレイヤー名を更新
    currentMatch.playerAName = '勇者';
    // currentMatch.playerBName = 'AI';
      
    // 点数リセット
    currentMatch.leftScore = 0;
    currentMatch.rightScore = 0;
      
    // ボールリセット
    currentMatch.resetBall();
      
    // バトルモードを設定
    currentMatch.setBattleMode();

    // バトル終了後の処理を設定
    currentMatch.onEnd(() => {
      currentMatch.leftController = originalLeftController;
      currentMatch.BattleMode = false;
    });
  }
}
