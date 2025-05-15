from .STFRunner import STFRunner
from .LTSFRunner import LTSFRunner
from .MHACRNRunner import MHACRNRUNNER 

def runner_select(name):
    name = name.upper()

    if name in ("STF", "BASIC", "DEFAULT"):
        return STFRunner
    elif name in ("LTSF", "LONG", "LONGTERM"):
        return LTSFRunner
    elif name in ('MHACRNRUNNER', 'MHACRN'):
        return MHACRNRUNNER
    else:
        raise NotImplementedError
