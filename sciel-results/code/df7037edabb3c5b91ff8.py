"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _check_k4_fast(colors):
    """Check every K4 has at least 5 distinct colours. Returns True if valid."""
    n = 16
    # Precompute edge colors in a 2D array for fast access
    edge = np.zeros((n, n), dtype=np.int32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge[i, j] = colors[idx]
            edge[j, i] = colors[idx]
            idx += 1
    
    # Check all K4s
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # 6 edges in K4
                    cs = [edge[a,b], edge[a,c], edge[a,d], edge[b,c], edge[b,d], edge[c,d]]
                    # Count distinct colors without set overhead
                    cs.sort()
                    distinct = 1
                    for k in range(1, 6):
                        if cs[k] != cs[k-1]:
                            distinct += 1
                    if distinct < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_colors = None
    best_count = 17  # initial solution uses 17 colours
    
    # The base solution: (i + j) % 17
    base_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    if _check_k4_fast(base_colors):
        best_colors = base_colors
        best_count = _num_colours(base_colors)
    
    # Search over (a*i + b*j) % p for various a, b, p
    # Focus on primes that give good coverage
    primes = [13, 17, 19, 23, 29, 31, 37]
    
    for p in primes:
        if time.time() > deadline:
            break
        # For each prime, try different linear coefficients
        for a in range(1, p):
            for b in range(1, p):
                if time.time() > deadline:
                    break
                colors = []
                for i in range(16):
                    for j in range(i + 1, 16):
                        colors.append((a * i + b * j) % p)
                cnt = _num_colours(colors)
                if cnt < best_count:
                    if _check_k4_fast(colors):
                        best_colors = colors
                        best_count = cnt
                        if best_count <= 13:
                            return colors
    
    # Try more general forms: (a*i + b*j + c*i*j) % p
    for p in [17, 19, 23]:
        if time.time() > deadline:
            break
        for a in range(1, min(p, 5)):
            for b in range(1, min(p, 5)):
                for c in range(0, min(p, 3)):
                    if time.time() > deadline:
                        break
                    colors = []
                    for i in range(16):
                        for j in range(i + 1, 16):
                            colors.append((a * i + b * j + c * i * j) % p)
                    cnt = _num_colours(colors)
                    if cnt < best_count:
                        if _check_k4_fast(colors):
                            best_colors = colors
                            best_count = cnt
                            if best_count <= 13:
                                return colors
    
    # Try (i * j) % p forms
    for p in [17, 19, 23]:
        if time.time() > deadline:
            break
        colors = [(i * j) % p for i in range(16) for j in range(i + 1, 16)]
        cnt = _num_colours(colors)
        if cnt < best_count and _check_k4_fast(colors):
            best_colors = colors
            best_count = cnt
    
    # Try (i^2 + j) % p and (i + j^2) % p
    for p in [17, 19, 23]:
        if time.time() > deadline:
            break
        for form in ['i2j', 'ij2']:
            colors = []
            for i in range(16):
                for j in range(i + 1, 16):
                    if form == 'i2j':
                        colors.append((i * i + j) % p)
                    else:
                        colors.append((i + j * j) % p)
            cnt = _num_colours(colors)
            if cnt < best_count and _check_k4_fast(colors):
                best_colors = colors
                best_count = cnt
    
    # Try combinations: (i + j) % p + offset
    for p in [13, 17, 19]:
        if time.time() > deadline:
            break
        for offset in range(p):
            colors = [((i + j) + offset) % p for i in range(16) for j in range(i + 1, 16)]
            cnt = _num_colours(colors)
            if cnt < best_count and _check_k4_fast(colors):
                best_colors = colors
                best_count = cnt
    
    # Local improvement: try swapping colors on individual edges
    if best_colors is not None and time.time() < deadline - 30:
        current = list(best_colors)
        current_count = best_count
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try changing each edge's color to a lower unused color
            for idx in range(120):
                if time.time() > deadline - 5:
                    break
                orig = current[idx]
                # Try replacing with each color from 0 to current_count-1
                for new_c in range(current_count):
                    if new_c == orig:
                        continue
                    current[idx] = new_c
                    if _check_k4_fast(current):
                        # Check if this reduces color count
                        new_count = _num_colours(current)
                        if new_count < current_count:
                            current_count = new_count
                            best_count = current_count
                            best_colors = current[:]
                            improved = True
                            if best_count <= 13:
                                return current
                            break
                current[idx] = orig
    
    if best_colors is None:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_colors
# EVOLVE-BLOCK-END
