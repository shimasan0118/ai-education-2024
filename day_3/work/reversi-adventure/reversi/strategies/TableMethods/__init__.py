import os
#import pyximport
#pyximport.install()


SLOW_MODE = True

try:
    if 'FORCE_TABLEMETHODS_IMPORT_ERROR' in os.environ:
        if os.environ['FORCE_TABLEMETHODS_IMPORT_ERROR'] == 'RAISE':
            raise ImportError

    from .GetScoreFast import get_score
    SLOW_MODE = False
except ImportError:
    from GetScore import get_score


__all__ = [
    'get_score',
]
