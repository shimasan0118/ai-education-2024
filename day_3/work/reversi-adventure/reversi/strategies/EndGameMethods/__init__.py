import os
#import pyximport
#pyximport.install()


ENDGAME_SIZE8_64BIT_ERROR = True

try:
    if 'FORCE_ENDGAMEMETHODS_IMPORT_ERROR' in os.environ:
        if os.environ['FORCE_ENDGAMEMETHODS_IMPORT_ERROR'] == 'RAISE':
            raise ImportError

    from .NextMoveSize8_64bit import next_move, get_best_move
    ENDGAME_SIZE8_64BIT_ERROR = False
except ImportError:
    pass


__all__ = [
    'next_move',
    'get_best_move',
]
