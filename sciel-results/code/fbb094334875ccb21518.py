_P = 1009
_M = 5

def _build_tables():
    is_qr = [False] * _P
    root = [0] * _P
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0:
            is_qr[r] = True
            if root[r] == 0:
                root[r] = x
    return is_qr, root

def construct():
    is_qr, root = _build_tables()
    
    for s in range(1, _P - _M + 1):
        if is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]:
            return {'start': s, 'roots': [root[s], root[s+1], root[s+2], root[s+3], root[s+4]]}
            
    raise RuntimeError("No solution found")
