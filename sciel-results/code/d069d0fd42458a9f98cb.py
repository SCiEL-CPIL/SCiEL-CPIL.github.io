"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Try to find L2 orthogonal to L1
    # Fix first row of L2 to 0..9
    L2 = [[None for _ in range(n)] for _ in range(n)]
    for j in range(n):
        L2[0][j] = j
        
    row_used = [0] * n
    col_used = [0] * n
    l1_sym_used = [0] * n
    
    for j in range(n):
        val = L2[0][j]
        row_used[0] |= (1 << val)
        col_used[j] |= (1 << val)
        l1_sym = (0 + j) % n
        l1_sym_used[l1_sym] |= (1 << val)
        
    def backtrack(i):
        if time.time() > deadline:
            return False
        if i == n:
            return True
            
        candidates = []
        for j in range(n):
            if L2[i][j] is not None:
                continue
            mask = row_used[i] | col_used[j] | l1_sym_used[(i+j)%n]
            avail = 0
            for v in range(n):
                if not (mask & (1 << v)):
                    avail |= (1 << v)
            if avail:
                cnt = bin(avail).count('1')
                candidates.append((cnt, j, avail))
                
        if not candidates:
            return False
            
        candidates.sort()
        
        for cnt, j, avail in candidates:
            v = avail
            while v:
                lb = v & -v
                val = lb.bit_length() - 1
                v -= lb
                
                L2[i][j] = val
                row_used[i] |= lb
                col_used[j] |= lb
                l1_sym = (i + j) % n
                l1_sym_used[l1_sym] |= lb
                
                if backtrack(i + 1):
                    return True
                    
                L2[i][j] = None
                row_used[i] ^= lb
                col_used[j] ^= lb
                l1_sym_used[l1_sym] ^= lb
                
        return False
        
    found = backtrack(1)
    
    if found:
        # Verify
        for i in range(n):
            if sorted(L2[i]) != list(range(n)):
                found = False
                break
        if found:
            for j in range(n):
                col = [L2[i][j] for i in range(n)]
                if sorted(col) != list(range(n)):
                    found = False
                    break
        if found:
            pairs = set()
            for i in range(n):
                for j in range(n):
                    p = (L1[i][j], L2[i][j])
                    if p in pairs:
                        found = False
                        break
                    pairs.add(p)
                if not found:
                    break
                    
    if found:
        return [L1, L2]
    else:
        return [L1]
# EVOLVE-BLOCK-END
