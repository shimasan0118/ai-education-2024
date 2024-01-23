import reversi

def Slime():
    return reversi.strategies.MonteCarlo(count=1)

def Dragon():
    return reversi.strategies.MonteCarlo(count=100)

def Maou():
    return reversi.strategies.MonteCarlo_EndGame(count=10000, end=14)
