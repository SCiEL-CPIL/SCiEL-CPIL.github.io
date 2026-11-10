import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119

# Precompute edges
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {}
for idx, (i, j) in enumerate(EDGES):
    EDGE_IDX[(i, j)] = idx
    EDGE_IDX[(j, i)] = idx

# Precompute 4-cycles
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                CYCLES.append((a, b, c, d))
                CYCLES.append((a, b, d, c))
                CYCLES.append((a, c, b, d))

def count_violations_xor(v, k):
    """
    v: list of N integers, each < 2^k
    Returns number of rainbow 4-cycles.
    """
    violations = 0
    for a, b, c, d in CYCLES:
        c1 = v[a] ^ v[b]
        c2 = v[b] ^ v[c]
        c3 = v[c] ^ v[d]
        c4 = v[d] ^ v[a]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            violations += 1
    return violations

def count_colors_xor(v, k):
    colors = set()
    for i in range(N):
        for j in range(i + 1, N):
            colors.add(v[i] ^ v[j])
    return len(colors)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_score = 0
    best_v = None
    best_k = 0
    
    # Try k from 4 to 8
    # k=4 -> 15 colors max
    # k=5 -> 31 colors max
    # k=6 -> 63 colors max
    # k=7 -> 127 colors max (but MAX_COLOR is 119, so we map)
    
    for k in range(4, 9):
        if time.time() > deadline - 5:
            break
            
        # Random restarts for this k
        for restart in range(10):
            if time.time() > deadline - 2:
                break
                
            # Initialize v
            v = [random.randint(0, (1 << k) - 1) for _ in range(N)]
            
            # Local search
            current_violations = count_violations_xor(v, k)
            
            # If valid, check score
            if current_violations == 0:
                score = count_colors_xor(v, k)
                if score > best_score:
                    best_score = score
                    best_v = list(v)
                    best_k = k
            
            # Improve
            for _ in range(500):
                if time.time() > deadline - 1:
                    break
                    
                if current_violations == 0:
                    # Try to increase colors by changing a vertex
                    # Or just keep it
                    pass
                else:
                    # Try to fix a violation
                    # Pick a random vertex involved in a violation
                    # For simplicity, pick random vertex and random new value
                    idx = random.randint(0, N - 1)
                    old_val = v[idx]
                    new_val = random.randint(0, (1 << k) - 1)
                    
                    v[idx] = new_val
                    new_violations = count_violations_xor(v, k)
                    
                    if new_violations < current_violations:
                        current_violations = new_violations
                    else:
                        v[idx] = old_val
            
            # Final check
            if count_violations_xor(v, k) == 0:
                score = count_colors_xor(v, k)
                if score > best_score:
                    best_score = score
                    best_v = list(v)
                    best_k = k

    if best_v is None:
        # Fallback to simple valid coloring
        # Use k=2, which is always valid
        k = 2
        best_v = [i % 4 for i in range(N)] # Just random
        # Actually, use a known valid one: v_i = i in Z_2^2 is not enough for 16 vertices.
        # We need to map 16 vertices to Z_2^k.
        # If k=4, and we just use v_i = i, it might have violations.
        # Let's just use the best found. If none, use a trivial one.
        # Trivial: all 0. Score 1.
        best_v = [0] * N
        best_k = 1
        best_score = 1

    # Map colors to 0..119
    # The colors are v[i] ^ v[j]
    # We need to return a list of 120 integers.
    colors = [0] * NUM_EDGES
    color_map = {}
    next_color_id = 0
    
    for e, (i, j) in enumerate(EDGES):
        c = best_v[i] ^ best_v[j]
        if c not in color_map:
            color_map[c] = next_color_id
            next_color_id += 1
        colors[e] = color_map[c]
        
    # Ensure colors are within 0..119
    # next_color_id should be <= 120
    # If it exceeds, we might need to merge colors or use a different mapping.
    # But max distinct colors for k=7 is 127, which is > 119.
    # However, the number of *edges* is 120. So we can have at most 120 distinct colors.
    # The constraint is "integer colors in 0..119".
    # So if we use 120 distinct colors, they must be a permutation of 0..119.
    
    return colors
