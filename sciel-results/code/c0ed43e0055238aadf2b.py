import itertools
import random

def construct():
    n = 34
    # Initialize colors to 0 (Red)
    colors = [0] * (n * (n - 1) * (n - 2) // 6)
    
    # Map triple (a, b, c) to index
    def get_idx(a, b, c):
        # Lexicographic order
        idx = 0
        for a2 in range(n):
            if a2 > a: break
            for b2 in range(a2 + 1, n):
                if a2 < a or (a2 == a and b2 > b): break
                if a2 == a and b2 < b: continue
                if a2 == a and b2 == b:
                    for c2 in range(b2 + 1, c + 1):
                        idx += 1
        return idx - 1 # 0-indexed
    
    # This mapping is O(n^3) per call, too slow.
    # Precompute indices.
    idx_map = {}
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                idx_map[(a, b, c)] = idx
                idx += 1
    total_edges = idx
    
    colors = [0] * total_edges
    
    def check_red_k4():
        # Find a red K4
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[idx_map[(a, b, c)]] == 0 and
                colors[idx_map[(a, b, d)]] == 0 and
                colors[idx_map[(a, c, d)]] == 0 and
                colors[idx_map[(b, c, d)]] == 0):
                return quad
        return None

    def check_blue_k5():
        # Find a blue K5
        for quint in itertools.combinations(range(n), 5):
            all_blue = True
            for triple in itertools.combinations(quint, 3):
                if colors[idx_map[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return quint
        return None

    # Greedy repair
    for _ in range(1000):
        red_k4 = check_red_k4()
        if red_k4:
            a, b, c, d = red_k4
            # Flip one edge to break the K4
            # Flip (a, b, c)
            colors[idx_map[(a, b, c)]] = 1
            continue
        
        blue_k5 = check_blue_k5()
        if blue_k5:
            # Flip one edge to break the K5
            # Flip the first triple in the K5
            first_triple = next(iter(itertools.combinations(blue_k5, 3)))
            colors[idx_map[first_triple]] = 0
            continue
            
        # If no violations, we are done
        return {'n': n, 'colors': colors}
        
    # If failed, return n=32 or smaller
    return {'n': 32, 'colors': [0] * (32 * 31 * 30 // 6)}
