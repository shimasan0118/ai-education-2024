//packages
const express = require('express')
const app = express()
const http = require('http')
const fs = require('fs')

// 環境変数からポートを取得、またはデフォルト値として3010を使用
const port = process.env.PORT || 3010;

let g_emotionsList = ["happy","sad","angry","scared","disgust", "suprised"]

//server init
const server = http.createServer(app).listen(port, () => {
  console.log('server running at ' + port)
})
//socket init
const io = require('socket.io')(server)

//config
app.use(express.static('public'))

//routes
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/views/index.html')
})

io.on('connection', function(socket){
  socket.on('room join', function(id){
    //check if the room is full
    if (io.sockets.adapter.rooms[id] != undefined) {
      if (io.sockets.adapter.rooms[id].length >= 2) {
        console.log('the room is full')
        //TODO handle error client side
        return
      }
    }

    //ルームに入る
    socket.join(id)
    socket._room = id
    socket._isReady = false

    let isOfferer = io.sockets.adapter.rooms[id].length == 1 ? false : true

    socket.emit('start rtc', isOfferer)

    console.log('joined', id)
    console.log('number of client:', io.sockets.adapter.rooms[id].length)

    //if two players are connected game can start
    if (io.sockets.adapter.rooms[id].length == 2) {
      io.to(id).emit('room ready')
    }
  })

  socket.on('rtc message', function(roomHash, message){
    if (message.sdp) {
      socket.to(roomHash).emit('respond offer', message.sdp)
    } else if (message.candidate) {
      socket.to(roomHash).emit('add candidate', message.candidate)
    }
  })

  //check if players are ready to move on to next level
  socket.on('player ready', function() {
    socket._isReady = true

    let playerReadyCount = 0
    io.in(socket._room).clients((err , players) => {
      for (player of players) {
        if (io.sockets.connected[player]._isReady) playerReadyCount++
      }

      if (playerReadyCount == 2) {
        let emotionIndices = [];
        let intervals = [];          
        console.log('all players are ready !')
        for (let i = 0; i < 500; i++) {
          let newIndex;
          do {
              newIndex = Math.floor(Math.random() * g_emotionsList.length);
          } while (emotionIndices.length > 0 && newIndex === emotionIndices[emotionIndices.length - 1]);
          emotionIndices.push(newIndex);
          intervals.push(Math.floor(Math.random() * (8000 - 3000 + 1)) + 3000); // 3～8秒のランダムな間隔
        }          
        io.to(socket._room).emit('set random', emotionIndices, intervals)
        io.to(socket._room).emit('next level')
        //reset players ready state
        resetPlayerReadyState(socket)
      }
    })
  })

  socket.on('player score', function(score) {
    socket.to(socket._room).emit('player score', score)
  })

  socket.on('disconnect', function() {
    socket.to(socket._room).emit('player left')
  })
}) //io connection event end

function resetPlayerReadyState(socket){
  io.in(socket._room).clients((err , players) => {
    for (player of players) {
      io.sockets.connected[player]._isReady = false
    }
  })
}
