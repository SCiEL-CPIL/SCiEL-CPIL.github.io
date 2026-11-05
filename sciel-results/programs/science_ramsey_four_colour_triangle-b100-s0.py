"""Model-and-optimize: construct large triangle-free 4-colorings via bitmask local search."""
import random
import time
import math

# EVOLVE-BLOCK-START

def _baseline32():
    """Greenwood-Gleason K16 doubled -> K32."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}

def _solve_n(n, rng, deadline):
    """
    Attempt to find a valid 4-coloring of K_n such that no monochromatic triangle exists.
    Returns the colors list if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    if m == 0:
        return []
    
    # Precompute edge indices
    # idx[u][v] = index in colors list
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1

    # Initialize random coloring
    colors = [rng.randint(0, 3) for _ in range(m)]
    
    # Initialize masks
    # masks[u][c] is a bitmask of vertices v such that edge (u,v) has color c
    masks = [[0] * 4 for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = colors[idx[u][v]]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)

    # Calculate initial triangle count
    tri_count = 0
    for u in range(n):
        for c in range(4):
            # Count pairs (v, w) in masks[u][c] such that (v,w) is also color c
            # Sum over v in masks[u][c] of popcount(masks[v][c] & masks[u][c])
            # Each triangle is counted 3 times (once for each vertex)
            # Actually, for a fixed u and c, the number of triangles (u,v,w) with color c is:
            # (1/2) * sum_{v in masks[u][c]} popcount(masks[v][c] & masks[u][c])
            # Because for each v, we count w. The pair (v,w) is counted once at u.
            # Wait, if we sum over all u, each triangle (u,v,w) is counted 3 times:
            # at u: v and w are in mask.
            # at v: u and w are in mask.
            # at w: u and v are in mask.
            # So total triangles = (1/3) * sum_{u} sum_{c} (1/2) * sum_{v in masks[u][c]} popcount(masks[v][c] & masks[u][c])
            
            # This O(N^3) initialization is fast enough for N<=62 (125k * 4 / 3 ~ 150k ops)
            pass
            
    # Simpler O(N^3) init for correctness and speed (Python loops are slow, but N is small)
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                c1 = colors[idx[u][v]]
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    tri_count += 1

    if tri_count == 0:
        return colors

    # Simulated Annealing / Local Search
    T = 1.0
    cooling = 0.9995
    steps = 0
    max_steps = 500000 # Safety cap, time is the real limit
    
    # We want to minimize tri_count
    while T > 0.001 and time.time() < deadline - 2.0:
        # Pick random edge
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        i_idx = idx[u][v]
        old_c = colors[i_idx]
        new_c = rng.randint(0, 3)
        if old_c == new_c:
            continue
        
        # Calculate delta in triangle count
        # lost: triangles containing (u,v) with color old_c
        # gained: triangles containing (u,v) with color new_c
        
        lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
        gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
        
        delta = gained - lost
        
        # Accept if improvement or with probability exp(-delta/T)
        accept = False
        if delta < 0:
            accept = True
        else:
            # Avoid overflow in exp
            if delta > 50 * T:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta / T)
        
        if accept:
            colors[i_idx] = new_c
            tri_count += delta
            
            # Update masks
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][new_c] |= (1 << v)
            masks[v][new_c] |= (1 << u)
            
            if tri_count == 0:
                return colors
            
            T *= cooling
            steps += 1
            
            # Optional: if stuck for a long time, restart with new random init?
            # For now, just keep cooling.
            
    return None

def construct():
    start = time.time()
    # Deadline with safety margin. 120s limit.
    deadline = start + 115.0 
    rng = random.Random(42)

    # Start with baseline
    best = _baseline32()
    best_n = 32

    # Try to increase n
    # We try n from 33 up to 62.
    # We allocate time to each n. If we fail, we try the next one with remaining time.
    # If we succeed, we move to the next n.
    
    # To be efficient, we can try larger n first? No, smaller n is easier.
    # But if we can't do 33, we won't do 34.
    # So we go 33, 34, ...
    
    # However, random restarts for a fixed n might be better than trying many n's briefly.
    # Let's give each n a decent chunk of time.
    
    # Strategy:
    # 1. Try n=33 with 10s.
    # 2. If success, try n=34 with 10s.
    # 3. If failure at n, try n again with more time? Or move on?
    #    Usually, if you can't solve n quickly, you might not solve n+1.
    #    But sometimes a lucky random init helps.
    
    # Let's try n=33 to 62.
    # If time < 5s, break.
    
    current_n = 33
    while current_n <= 62:
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time for this n
        # If we have lots of time, give more.
        # Simple heuristic: 5 seconds per attempt?
        # Or scale with remaining time?
        
        # Let's try to solve current_n
        sub_deadline = time.time() + min(10.0, deadline - time.time() - 2.0)
        
        # Multiple restarts within sub_deadline
        res = None
        while time.time() < sub_deadline:
            res = _solve_n(current_n, rng, sub_deadline)
            if res is not None:
                break
            # If we didn't find it, we loop and try again with a new random init
            # _solve_n uses rng, so it will be different.
        
        if res is not None:
            # Valid coloring found
            best = {'n': current_n, 'colors': res}
            best_n = current_n
            current_n += 1
        else:
            # Failed to find coloring for current_n
            # Should we try larger n? Unlikely to succeed if smaller failed.
            # But maybe a specific structure?
            # For now, stop trying larger n if we failed at current_n?
            # Or try one more time with more time?
            
            # If we have plenty of time, try again with a longer budget?
            # If time.time() < deadline - 20:
            #    try again with 20s
            # else:
            #    break
            
            # Let's break if we failed, as it's a hard constraint.
            # R(3,3,3,3) > 50 is known, but finding it is hard.
            # If we can't do 33, we are stuck at 32.
            break

    # Final validation
    if time.time() < deadline - 1.0:
        n = best['n']
        colors = best['colors']
        if len(colors) != n * (n - 1) // 2:
            best = _baseline32()
        else:
            # Quick validation
            idx = [[0] * n for _ in range(n)]
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    idx[i][j] = k
                    idx[j][i] = k
                    k += 1
            
            valid = True
            for u in range(n):
                for v in range(u + 1, n):
                    for w in range(v + 1, n):
                        c1 = colors[idx[u][v]]
                        if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                best = _baseline32()

    return best

# EVOLVE-BLOCK-END
