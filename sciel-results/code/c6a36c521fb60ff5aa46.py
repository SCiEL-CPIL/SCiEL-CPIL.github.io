import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Primes p ≡ 3 mod 4 in range
    # 31, 43, 47, 59, 67, 71, 79, 83
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    best_paley_n = 83
    best_paley_orientation = None
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Construct Paley
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # Build out_masks
        out_masks = [0] * p
        for i in range(p):
            mask = 0
            for j in range(p):
                if i == j: continue
                if (j - i) % p in squares:
                    mask |= (1 << j)
            out_masks[i] = mask
            
        # Check validity
        valid = True
        for combo in itertools.combinations(range(p), 4):
            m = out_masks[combo[0]] & out_masks[combo[1]] & out_masks[combo[2]] & out_masks[combo[3]]
            if m == 0:
                valid = False
                break
                
        if valid:
            best_paley_n = p
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            best_paley_orientation = orientation
            
    # Try random tournaments for n < best_paley_n
    for n in range(best_paley_n - 1, 30, -1):
        if time.time() > deadline:
            break
            
        trials = 5
        for _ in range(trials):
            if time.time() > deadline:
                break
                
            bits = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            
            out_masks = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    b = bits[idx]
                    idx += 1
                    if b == 0:
                        out_masks[i] |= (1 << j)
                    else:
                        out_masks[j] |= (1 << i)
                        
            valid = True
            for combo in itertools.combinations(range(n), 4):
                m = out_masks[combo[0]] & out_masks[combo[1]] & out_masks[combo[2]] & out_masks[combo[3]]
                if m == 0:
                    valid = False
                    break
                    
            if valid:
                return {'n': n, 'orientation': bits}
                
    if best_paley_orientation is not None:
        return {'n': best_paley_n, 'orientation': best_paley_orientation}
        
    # Fallback to 83
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}
