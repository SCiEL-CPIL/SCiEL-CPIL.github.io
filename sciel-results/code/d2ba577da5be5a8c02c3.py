import math
import time

def construct():
    deadline = time.time() + 110
    
    def is_valid(selected, m, r):
        for (m2, r2) in selected:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    selected = []
    for m in range(20, 201):
        if time.time() > deadline:
            break
        best_r = None
        best_cs = None
        for r in range(m):
            if is_valid(selected, m, r):
                # Tiebreaker: minimize "closeness" to existing residues mod gcd
                cs = 0
                for (m2, r2) in selected:
                    g = math.gcd(m, m2)
                    d = abs(r % g - r2 % g)
                    d = min(d, g - d)
                    cs += d
                if best_cs is None or cs > best_cs:
                    best_cs = cs
                    best_r = r
                if cs == 0:
                    break
        if best_r is not None:
            selected.append((m, best_r))
    
    return [[m, r] for m, r in selected]
