from .model import MHACRN 

def model_select(name):
    name = name.upper()

    if name == 'MHACRN':
        return MHACRN 
    else:
        raise NotImplementedError
