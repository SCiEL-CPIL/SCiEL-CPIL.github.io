"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe under 90s limit
    N = 512
    n = N
    
    # Precompute compatibility masks
    # mask[i] is a bitmask where bit j is 1 if lcm(i+1, j+1) <= N
    masks = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(n):
            b = j + 1
            g = gcd(a, b)
            if a * b // g <= N:
                masks[i] |= (1 << j)
    
    best_size = 0
    best_set = set()
    best_mask = 0
    
    def try_improve(C_set, C_mask):
        nonlocal best_size, best_set, best_mask
        C_list = list(C_set)
        # Try to add elements
        while True:
            added = False
            # To speed up, iterate over all v not in C
            # We can precompute a list of candidates? No, C changes.
            for v in range(n):
                if v not in C_set:
                    if (masks[v] & C_mask) == C_mask:
                        C_set.add(v)
                        C_mask |= (1 << v)
                        C_list.append(v)
                        added = True
                        if len(C_set) > best_size:
                            best_size = len(C_set)
                            best_set = set(C_set)
                            best_mask = C_mask
                        break
            if not added:
                break
        
        # If we couldn't add, do random swaps to escape local minimum
        # We do this in a loop controlled by the main caller
        return C_set, C_mask

    # Initializations
    inits = []
    
    # 1. Natural order 1..22
    s1 = set(range(22))
    m1 = 0
    for i in s1: m1 |= (1 << i)
    inits.append((s1, m1))
    
    # 2. Divisors of 512
    divs = [i+1 for i in range(n) if 512 % (i+1) == 0]
    s2 = set([d-1 for d in divs])
    m2 = 0
    for i in s2: m2 |= (1 << i)
    inits.append((s2, m2))
    
    # 3. Powers of 2
    pows = set()
    p = 1
    while p <= 512:
        pows.add(p-1)
        p *= 2
    m3 = 0
    for i in pows: m3 |= (1 << i)
    inits.append((pows, m3))
    
    # 4. Divisors of 360 (known good seed)
    divs360 = [i+1 for i in range(n) if 360 % (i+1) == 0]
    s4 = set([d-1 for d in divs360])
    m4 = 0
    for i in s4: m4 |= (1 << i)
    inits.append((s4, m4))
    
    # 5. Divisors of 480
    divs480 = [i+1 for i in range(n) if 480 % (i+1) == 0]
    s5 = set([d-1 for d in divs480])
    m5 = 0
    for i in s5: m5 |= (1 << i)
    inits.append((s5, m5))
    
    # 6. Random
    random.seed(42)
    for _ in range(10):
        s6 = set(random.sample(range(n), 20))
        m6 = 0
        for i in s6: m6 |= (1 << i)
        inits.append((s6, m6))

    for s, m in inits:
        if time.time() > deadline:
            break
        s, m = try_improve(s, m)
        
        # Local search with swaps
        stagnation = 0
        while time.time() < deadline:
            # Try add
            added = False
            for v in range(n):
                if v not in s:
                    if (masks[v] & m) == m:
                        s.add(v)
                        m |= (1 << v)
                        added = True
                        if len(s) > best_size:
                            best_size = len(s)
                            best_set = set(s)
                            best_mask = m
                        break
            if added:
                stagnation = 0
                continue
            
            # No add possible, try a random valid swap
            # Pick random u in s
            u = random.choice(list(s))
            u_bit = 1 << u
            m_wo_u = m ^ u_bit
            # Find random v not in s compatible with m_wo_u
            # Filter candidates
            candidates = []
            for v in range(n):
                if v not in s:
                    if (masks[v] & m_wo_u) == m_wo_u:
                        candidates.append(v)
            
            if candidates:
                v = random.choice(candidates)
                s.remove(u)
                s.add(v)
                m = m_wo_u | (1 << v)
                stagnation = 0
            else:
                # No valid swap found, stuck?
                # Try a different u
                # If we keep failing, break and try next init
                stagnation += 1
                if stagnation > 50:
                    break
    
    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
