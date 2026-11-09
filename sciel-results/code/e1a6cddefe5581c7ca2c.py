import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def get_paley(p):
        squares = {x*x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                if ((j - i) % p) in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    def count_bad_fast(orient, n):
        in_m = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orient[idx] == 0:
                    in_m[j] |= (1 << i)
                else:
                    in_m[i] |= (1 << j)
                idx += 1
        
        bad = 0
        for a in range(n):
            ia = in_m[a]
            for b in range(a+1, n):
                iab = ia & in_m[b]
                if iab == 0:
                    rem = n - b - 1
                    if rem >= 2:
                        bad += rem * (rem - 1) // 2
                    continue
                for c in range(b+1, n):
                    iabc = iab & in_m[c]
                    if iabc == 0:
                        bad += n - c - 1
                        continue
                    for d in range(c+1, n):
                        if (iabc & in_m[d]) == 0:
                            bad += 1
        return bad

    # Check smaller Paley tournaments
    # Primes p = 3 mod 4
    candidates = [11, 19, 23, 31, 43, 47, 59]
    
    for p in candidates:
        if time.time() > deadline - 10:
            break
        res = get_paley(p)
        bad = count_bad_fast(res['orientation'], p)
        if bad == 0:
            return res

    # Fallback to Paley 67
    return get_paley(67)
