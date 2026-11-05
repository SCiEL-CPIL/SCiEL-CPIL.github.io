import time

def _gen_smooth(limit):
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(vals)

# Precompute smooth numbers up to 128
_SMOOTH = _gen_smooth(128)
_SMOOTH_SET = set(_SMOOTH)
_N = len(_SMOOTH)
_IDX = {v: i for i, v in enumerate(_SMOOTH)}

# Precompute incompatibility matrix (a|b or b|a)
_INCOMP = [[False] * _N for _ in range(_N)]
for i in range(_N):
    a = _SMOOTH[i]
    for j in range(i + 1, _N):
        b = _SMOOTH[j]
        if a % b == 0 or b % a == 0:
            _INCOMP[i][j] = True
            _INCOMP[j][i] = True

def _incompatible_val(a, b):
    if a == b:
        return True
    return (a % b == 0) or (b % a == 0)

def _find_representation(n):
    # k=1
    if n in _SMOOTH_SET:
        return [n]
    
    # k=2
    # We need a+b=n, a!=b, a,b smooth, a !| b
    # Iterate a in _SMOOTH
    for a in _SMOOTH:
        if a >= n:
            break
        b = n - a
        if b <= 0 or b == a:
            continue
        if b not in _SMOOTH_SET:
            continue
        if _incompatible_val(a, b):
            continue
        return [a, b]
        
    # k=3
    # a+b+c=n, a<b<c, smooth, pairwise incompatible
    for i in range(_N):
        a = _SMOOTH[i]
        if a >= n:
            break
        for j in range(i + 1, _N):
            b = _SMOOTH[j]
            if a + b >= n:
                break
            if _INCOMP[i][j]:
                continue
            c = n - a - b
            if c <= b:
                continue
            if c not in _SMOOTH_SET:
                continue
            k = _IDX[c]
            if _INCOMP[i][k] or _INCOMP[j][k]:
                continue
            return [a, b, c]
            
    # k=4
    # a+b+c+d=n
    for i in range(_N):
        a = _SMOOTH[i]
        if a >= n:
            break
        for j in range(i + 1, _N):
            b = _SMOOTH[j]
            if a + b >= n:
                break
            if _INCOMP[i][j]:
                continue
            for k in range(j + 1, _N):
                c = _SMOOTH[k]
                if a + b + c >= n:
                    break
                if _INCOMP[i][k] or _INCOMP[j][k]:
                    continue
                d = n - a - b - c
                if d <= c:
                    continue
                if d not in _SMOOTH_SET:
                    continue
                l = _IDX[d]
                if _INCOMP[i][l] or _INCOMP[j][l] or _INCOMP[k][l]:
                    continue
                return [a, b, c, d]
                
    # Fallback: should not happen for 1..128
    # But just in case, return a valid representation if possible, else [n] (invalid if n not smooth)
    # Actually, the problem states "All targets require valid representations".
    # If we reach here, we failed.
    return [n]

def construct():
    start = time.time()
    DEADLINE = 80.0
    
    answers = [None] * 129
    for n in range(1, 129):
        if time.time() - start > DEADLINE:
            break
        answers[n] = _find_representation(n)
        
    for n in range(1, 129):
        if answers[n] is None:
            answers[n] = _find_representation(n)
            
    return answers[1:129]
