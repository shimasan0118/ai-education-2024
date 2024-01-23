// -*- coding: utf-8 -*-
const apiUrl = 'http://localhost:3025';
const boardElement = document.getElementById('board');
const playerVsPlayerButton = document.getElementById('playerVsPlayerButton');
const dungeonButton = document.getElementById('dungeonButton');
const cpuSelection = document.getElementById('cpuSelection');
const vsSlimeButton = document.getElementById('vsSlimeButton');
const vsDragonButton = document.getElementById('vsDragonButton');
const vsDemonKingButton = document.getElementById('vsDemonKingButton');
const menu = document.getElementById('menu');
const gameContainer = document.getElementById('gameContainer');
const backToMenuButton = document.getElementById('backToMenuButton');
const secretAISwitch = document.getElementById('secretAISwitch');


playerVsPlayerButton.addEventListener('click', () => startGame());
dungeonButton.addEventListener('click', showCpuSelection);
vsSlimeButton.addEventListener('click', () => startGame('slime'));
vsDragonButton.addEventListener('click', () => startGame('dragon'));
vsMaouButton.addEventListener('click', () => startGame('maou'));
backToMenuButton.addEventListener('click', () => {
    gameContainer.style.display = 'none';
    menu.style.display = 'block';
});

// enemyImage要素を取得
const enemyImage = document.getElementById('enemyImage');
// yusyaImage要素を取得
const yusyaImage = document.getElementById('yusyaImage');

let secretInput = ''; // キー入力を追跡するための変数
const secretCode = '3941'; // 隠しコマンド
const secretToggle = document.getElementById('secretToggle');

let currentBoard = '' // 現在表示されているボード状態

let gameIsOver = false; // ゲーム終了フラグ
let alertThisTurn = false //パスのアラートを1回だけ出すためのフラグ

let cpu = ''; // どの敵か

document.addEventListener('keydown', (e) => {
    secretInput += e.key; // キー入力を追加
    if (secretInput.length > secretCode.length) {
        // 入力がシークレットコードより長くなったら、最初の文字を削除
        secretInput = secretInput.substr(1);
    }

    if (secretInput === secretCode) {
        alert('封印されていたAIが解放されました');
        secretToggle.style.display = 'block'; // トグルスイッチを表示
        secretInput = ''; // 入力をリセット
    }
});

function showCpuSelection() {
    cpuSelection.style.display = 'block';
}

// startButton.addEventListener('click', startGame);

function startGame(cpuStrategy = '') {
    // 画像の傾きをリセットする
    yusyaImage.classList.remove('rotate-315deg');
    enemyImage.classList.remove('rotate-45deg');    
    // 敵の種類に基づいてenemyImageを切り替える
    if (cpuStrategy === '') {
        enemyImage.src = '';
        enemyImage.style.display = 'none'; // 要素自体を非表示にする
    } else if (cpuStrategy === 'slime') {
        enemyImage.src = '/static/img/slime.png';
        enemyImage.style.display = 'block'; // 要素を表示する
        enemyImage.classList.add('slime');
    } else if (cpuStrategy === 'dragon') {
        enemyImage.src = '/static/img/dragon.png';
        enemyImage.style.display = 'block'; // 要素を表示する
        enemyImage.classList.add('dragon');
    } else if (cpuStrategy === 'maou') {
        enemyImage.src = '/static/img/maou.png';
        enemyImage.style.display = 'block'; // 要素を表示する
        enemyImage.classList.add('maou');
    }    
    cpu = cpuStrategy;
    gameIsOver = false;
    menu.style.display = 'none';
    secretToggle.style.display = 'none';
    gameContainer.style.display = 'block';
    fetch(`${apiUrl}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cpuStrategy })
    })
    .then(response => response.json())
    .then(data => {          
        updateBoard(data.board);     
        if (data.current_player === 'black' && secretAISwitch.checked) {
            getRecommendedMove();
        } 
        // cpuMove(); もしCPUを先手にしたい場合は、getValidMovesではなくcpuMoveを呼び出す必要がある
    })
    .catch(error => console.error('Error:', error));
}

function updateBoard(boardString, finish=false, checkPass=true) {
    boardElement.innerHTML = '';
    currentBoard = boardString
    for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.addEventListener('click', () => makeMove(x, y));

            const cellValue = boardString[y * 8 + x];
            if (cellValue === 'B' || cellValue === 'W') {
                const disc = document.createElement('div');
                disc.className = 'disc ' + (cellValue === 'B' ? 'black' : 'white');
                cell.appendChild(disc);
            }

            boardElement.appendChild(cell);
        }
    }
    if (lastCpuMove) {
        highlightCpuMove(lastCpuMove); // 盤面更新後にCPUの手を再度ハイライト
    }
    if (!finish) {
        getValidMoves(checkPass);
    }
}

function makeMove(x, y) {
    const isValidMove = validMoves.some(move => move[0] === x && move[1] === y);

    if (!isValidMove) {
        console.log("Invalid move");
        return; // 有効な手でない場合は何もしない
    }

    fetch(`${apiUrl}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x, y })
    })
    .then(response => response.json())
    .then(data => {
        alertThisTurn = false        
        updateBoard(data.board);       
        if (data.current_player === 'black' && secretAISwitch.checked) {
            getRecommendedMove(); // 黒番プレイヤーの場合にレコメンド機能を呼び出す
        } else if (data.current_player === 'white' && data.is_cpu_player) {
            cpuMove(); // 白番CPUの場合にCPUの手を実行
        }
    })
    .catch(error => console.error('Error:', error));
}

function getRecommendedMove() {
    fetch(`${apiUrl}/recommend`)
        .then(response => response.json())
        .then(data => {
            if (data.move && data.board === currentBoard){
                highlightMove(data.move);
            }
        })
        .catch(error => console.error('Error:', error));
}

// レコメンドされた手をハイライトする関数
function highlightMove(move) {
    // レコメンドされた手をボード上でハイライトする
    const [x, y] = move;
    const cellIndex = y * 8 + x;
    const cell = boardElement.children[cellIndex];
    cell.classList.add('highlighted-move');
}

let lastCpuMove = [];  // CPUの最後の手の位置を保持する変数

function cpuMove() {
    fetch(`${apiUrl}/cpu_move`)
        .then(response => response.json())
        .then(data => {
            alertThisTurn = false        
            lastCpuMove = data.cpu_move;
            if (data.cpu_move.length > 0) {
                updateBoard(data.board);
            }
            if (data.current_player === 'black' && secretAISwitch.checked) {
                getRecommendedMove(); // 黒番プレイヤーの場合にレコメンド機能を呼び出す
            }
        })
        .catch(error => console.error('Error:', error));
}

function getValidMoves(checkPass=true) {
    fetch(`${apiUrl}/status`)
        .then(response => response.json())
        .then(data => {
            if (data.game_over){
                updateBoard(data.board, finish=true); // 最終的な盤面を更新                
                fetch(`${apiUrl}/result`)
                    .then(response => response.json())
                    .then(resultData => {
                        if (!gameIsOver){
                            if (resultData.result === "Black WIN!"){
                                alert(`${resultData.diff}石差で勇者の勝利です！`);
                            }
                            else if (resultData.result === "White WIN!"){
                                alert(`${resultData.diff}石差で敗北です...`);
                            }
                            else{
                                alert(`引き分けです`);
                            }
                            gameIsOver= true;
                            if (resultData.result === "Black WIN!" && cpu !== '') {
                                localStorage.setItem(`stageCleared_${cpu}`, true);
                                updateStageButtons();
                                // 敵の画像を傾ける
                                enemyImage.classList.add('rotate-45deg');                           
                            } else if (resultData.result === "White WIN!") {
                                // 勇者の画像を傾ける
                                yusyaImage.classList.add('rotate-315deg');
                            }
                        }
                    })
                    .catch(error => console.error('Error:', error));
                return;
            }

            else if (data.valid_moves && Array.isArray(data.valid_moves)) {
                updateValidMoves(data.valid_moves);
                highlightValidMoves(data.valid_moves);
                // 有効な手がない場合はパス
                if (data.valid_moves.length === 0) {
                    if (!alertThisTurn){
                        alert(`${data.current_player === 'black' ? '黒' : '白'}番には有効な手がありません。`);
                    }
                    alertThisTurn = true
                    if (checkPass){
                        passTurn();
                    }
                }                
            } else {
                console.error('Valid moves are not available:', data);
            }
        })
        .catch(error => console.error('Error:', error));
}

function passTurn() {
    // パスをサーバーに通知するための処理
    fetch(`${apiUrl}/pass`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {       
            updateBoard(data.board, checkPass=false);
            if (data.current_player === 'white' && data.is_cpu_player) {
                // 白番プレイヤーがCPUの場合、CPUの手を実行
                cpuMove();
            }
            if (data.current_player === 'black' && secretAISwitch.checked) {
                getRecommendedMove(); // 黒番プレイヤーの場合にレコメンド機能を呼び出す
            }
        })
        .catch(error => console.error('Error:', error));
}

let validMoves = [];

function updateValidMoves(newValidMoves) {
    validMoves = newValidMoves;
}

function highlightCpuMove(move) {
    if (move.length === 0)  return;  // ハイライトする手がない場合は何もしない
    
    const [x, y] = move;
    const cellIndex = y * 8 + x;
    if (cellIndex >= boardElement.children.length) {
        return;  // インデックスが範囲外の場合は何もしない
    }
    const cell = boardElement.children[cellIndex];
    const disc = cell.querySelector('.disc');
    if (disc) {
        disc.classList.add('cpu-highlight');
    }
}

function highlightValidMoves(newValidMoves) {
    updateValidMoves(newValidMoves); 
    document.querySelectorAll('.valid-move').forEach(cell => cell.classList.remove('valid-move'));

    newValidMoves.forEach(move => {
        const x = move[0];
        const y = move[1];
        const cellIndex = y * 8 + x;
        if (cellIndex < boardElement.children.length) {
            const cell = boardElement.children[cellIndex];
            cell.classList.add('valid-move');
        }
    });
}

function updateStageButtons() {
    ['slime', 'dragon', 'maou'].forEach(stage => {
        const isCleared = localStorage.getItem(`stageCleared_${stage}`);
        const button = document.getElementById(`vs${stage.charAt(0).toUpperCase() + stage.slice(1)}Button`);
        const clearBadge = button.querySelector('.clear-badge');
        if (isCleared) {
            clearBadge.style.display = 'block'; // CLEARバッジを表示
            button.classList.add('button-cleared'); // ボタンの色を変更
        } else {
            clearBadge.style.display = 'none';
            button.classList.remove('button-cleared');
        }
    });
}

// ページ読み込み時にボタンの状態を更新
document.addEventListener('DOMContentLoaded', updateStageButtons);


// 文字列の最初の文字を大文字にするヘルパー関数
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
