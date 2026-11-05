"""Model-and-optimize: construct large triangle-free 4-colorings via Chung's construction + bitmask local search."""
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

def _build_chung_50():
    """
    Chung's 50-vertex 4-colouring of K_50 with no monochromatic triangle.
    Based on the Paley graph of order 51 (Z_51), using QR mod 17.
    """
    qr_17 = set()
    for k in range(17):
        qr_17.add((k * k) % 17)
    # Non-zero QR mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    
    def get_color(i, j):
        d = (j - i) % 51
        d3 = d % 3
        d17 = d % 17
        if d17 == 0:
            in_qr = True
        else:
            in_qr = (d17 in qr_17)
        
        if d3 == 1:
            return 0 if in_qr else 1
        else:  # d3 == 2
            return 2 if in_qr else 3
    
    n = 50
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            vi = i + 1
            vj = j + 1
            colors.append(get_color(vi, vj))
    
    return {'n': n, 'colors': colors}

def _validate_simple(colors, n):
    """O(n^3) validation that no monochromatic triangle exists."""
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    for u in range(n):
        for v in range(u + 1, n):
            c1 = colors[idx[u][v]]
            for w in range(v + 1, n):
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    return False
    return True

def _solve_n(n, rng, deadline, init_colors=None):
    """
    Attempt to find a valid 4-coloring of K_n with no monochromatic triangle.
    Uses bitmask local search (simulated annealing) to minimize triangle count.
    """
    m = n * (n - 1) // 2
    if m == 0:
        return []
    
    # Precompute edge indices
    idx = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1

    # Initialize coloring
    if init_colors is not None:
        colors = init_colors[:]
    else:
        # Greedy initialization: for each edge (i,j), pick color minimizing new triangles
        colors = [0] * m
        for i in range(n):
            for j in range(i + 1, n):
                best_c = 0
                best_tri = float('inf')
                for c in range(4):
                    tri = 0
                    for kk in range(n):
                        if kk == i or kk == j:
                            continue
                        idx_ik = idx[i][kk]
                        idx_jk = idx[j][kk]
                        if colors[idx_ik] == c and colors[idx_jk] == c:
                            tri += 1
                    if tri < best_tri:
                        best_tri = tri
                        best_c = c
                colors[idx[i][j]] = best_c
    
    # Initialize masks
    masks = [[0] * 4 for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = colors[idx[u][v]]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)

    # Calculate initial triangle count
    tri_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                c1 = colors[idx[u][v]]
                if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                    tri_count += 1

    if tri_count == 0:
        return colors

    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    
    while T > min_T and time.time() < deadline - 1.0:
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
        
        lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
        gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
        delta = gained - lost
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if delta > 50 * T:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta / T)
        
        if accept:
            colors[i_idx] = new_c
            tri_count += delta
            
            masks[u][old_c] &= ~(1 << v)
            masks[v][old_c] &= ~(1 << u)
            masks[u][new_c] |= (1 << v)
            masks[v][new_c] |= (1 << u)
            
            if tri_count == 0:
                return colors
            
            T *= cooling

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # First: try the deterministic Chung 50-vertex construction
    try:
        chung = _build_chung_50()
        if _validate_simple(chung['colors'], chung['n']):
            best = chung
            best_n = 50
        else:
            best = _baseline32()
            best_n = 32
    except Exception:
        best = _baseline32()
        best_n = 32
    
    # Now try to extend beyond best_n using local search
    rng = random.Random(42)
    current_n = best_n + 1
    
    while current_n <= 62:
        if time.time() > deadline - 5.0:
            break
        
        # Time budget for this n: scale with remaining time
        remaining = deadline - time.time() - 3.0
        if remaining < 3.0:
            break
        
        # Give up to 8 seconds per attempt, with multiple restarts
        sub_deadline = time.time() + min(8.0, remaining)
        
        res = None
        # Try a few restarts
        for attempt in range(3):
            if time.time() >= sub_deadline:
                break
            res = _solve_n(current_n, rng, sub_deadline, init_colors=None)
            if res is not None:
                break
        
        if res is not None:
            # Validate before accepting
            if _validate_simple(res, current_n):
                best = {'n': current_n, 'colors': res}
                best_n = current_n
                current_n += 1
            else:
                break
        else:
            # Failed to find coloring for current_n
            # Try one more time with longer budget if we have time
            remaining = deadline - time.time() - 3.0
            if remaining > 15.0:
                sub_deadline2 = time.time() + min(20.0, remaining)
                res2 = _solve_n(current_n, rng, sub_deadline2, init_colors=None)
                if res2 is not None and _validate_simple(res2, current_n):
                    best = {'n': current_n, 'colors': res2}
                    best_n = current_n
                    current_n += 1
                else:
                    break
            else:
                break
    
    # Final validation
    if time.time() < deadline - 1.0:
        n = best['n']
        colors = best['colors']
        if len(colors) != n * (n - 1) // 2:
            best = _baseline32()
        else:
            if not _validate_simple(colors, n):
                best = _baseline32()
    
    return best

# EVOLVE-BLOCK-END
