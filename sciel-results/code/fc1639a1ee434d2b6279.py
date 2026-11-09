"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _check_validity(colors, n=128):
    """Check if coloring satisfies: every nonconstant 4-term AP has >=3 distinct colors."""
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            c0 = colors[a]
            c1 = colors[a + d]
            c2 = colors[a + 2*d]
            c3 = colors[a + 3*d]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _precompute_aps(n=128):
    """Precompute all 4-term APs as index tuples."""
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
    return aps

def _search_group_theoretic(deadline):
    """Use group-theoretic approach: map positions to F_2^7 and use nonlinear coloring."""
    n = 128
    aps = _precompute_aps(n)
    best_colors = None
    best_count = 999
    
    # Map position i to vector in F_2^7: use binary representation
    # Position i (0-indexed) maps to 7-bit binary of i
    pos_to_vec = {}
    for i in range(n):
        # Convert i to 7-bit binary
        bits = []
        temp = i
        for b in range(7):
            bits.append(temp & 1)
            temp >>= 1
        pos_to_vec[i] = tuple(bits)
    
    # Try different nonlinear functions from F_2^7 to small color sets
    # Function 1: Hamming weight mod k
    # Function 2: Sum of selected pairs (quadratic form)
    # Function 3: Combination of linear and quadratic terms
    
    # Precompute hamming weights
    hamming_weights = [sum(v) for v in pos_to_vec.values()]
    
    # Strategy: Try various functions and optimize via local search
    # Function family: f(x) = (sum_{i<j} x_i * x_j * M[i][j]) mod k
    # where M is a 7x7 binary matrix
    
    # Try k = 2, 3, 4 colors
    for k in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Try random quadratic forms
        for trial in range(500):
            if time.time() > deadline:
                break
            
            # Random 7x7 binary matrix (upper triangle only matters)
            M = np.zeros((7, 7), dtype=int)
            for i in range(7):
                for j in range(i+1, 7):
                    M[i][j] = random.randint(0, 1)
                    M[j][i] = M[i][j]
            
            # Compute colors: f(x) = sum_{i<j} x_i*x_j*M[i][j] mod k
            colors = []
            for i in range(n):
                v = pos_to_vec[i]
                val = 0
                for a in range(7):
                    for b in range(a+1, 7):
                        if M[a][b]:
                            val += v[a] * v[b]
                colors.append(val % k)
            
            # Check validity
            if _check_validity(colors, n):
                cnt = len(set(colors))
                if cnt < best_count:
                    best_count = cnt
                    best_colors = list(colors)
                if best_count == 2:
                    return best_colors
            
            # Local search: flip random entries in M
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                # Flip a random entry
                a = random.randint(0, 6)
                b = random.randint(0, 6)
                if a == b:
                    continue
                M[a][b] = 1 - M[a][b]
                M[b][a] = M[a][b]
                
                # Recompute colors
                colors = []
                for i in range(n):
                    v = pos_to_vec[i]
                    val = 0
                    for x in range(7):
                        for y in range(x+1, 7):
                            if M[x][y]:
                                val += v[x] * v[y]
                    colors.append(val % k)
                
                if _check_validity(colors, n):
                    cnt = len(set(colors))
                    if cnt < best_count:
                        best_count = cnt
                        best_colors = list(colors)
                        if best_count == 2:
                            return best_colors
    
    # Try linear + constant offset: f(x) = (sum_i x_i * c_i) mod k
    for k in [2, 3, 4]:
        if time.time() > deadline:
            break
        
        for trial in range(200):
            if time.time() > deadline:
                break
            
            c = [random.randint(0, 1) for _ in range(7)]
            offset = random.randint(0, k-1)
            
            colors = []
            for i in range(n):
                v = pos_to_vec[i]
                val = offset
                for j in range(7):
                    val += v[j] * c[j]
                colors.append(val % k)
            
            if _check_validity(colors, n):
                cnt = len(set(colors))
                if cnt < best_count:
                    best_count = cnt
                    best_colors = list(colors)
                if best_count == 2:
                    return best_colors
    
    # Try mixed: quadratic + linear
    for k in [2, 3, 4]:
        if time.time() > deadline:
            break
        
        for trial in range(300):
            if time.time() > deadline:
                break
            
            M = np.zeros((7, 7), dtype=int)
            for i in range(7):
                for j in range(i+1, 7):
                    M[i][j] = random.randint(0, 1)
                    M[j][i] = M[i][j]
            
            c = [random.randint(0, 1) for _ in range(7)]
            offset = random.randint(0, k-1)
            
            colors = []
            for i in range(n):
                v = pos_to_vec[i]
                val = offset
                # Linear part
                for j in range(7):
                    val += v[j] * c[j]
                # Quadratic part
                for a in range(7):
                    for b in range(a+1, 7):
                        if M[a][b]:
                            val += v[a] * v[b]
                colors.append(val % k)
            
            if _check_validity(colors, n):
                cnt = len(set(colors))
                if cnt < best_count:
                    best_count = cnt
                    best_colors = list(colors)
                if best_count == 2:
                    return best_colors
    
    return best_colors

def _search_periodic(deadline):
    """Search for valid periodic colorings with small periods and few colors."""
    best = None
    best_count = 999
    
    # Try periods from small to larger
    for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 60, 64, 72, 80, 96, 120, 128]:
        if time.time() > deadline:
            break
        
        # Try with 2 colors first, then 3
        for num_colors in [2, 3]:
            if time.time() > deadline:
                break
            
            if p * num_colors**p > 200000:  # too many patterns, skip
                continue
            
            # Generate all possible base patterns of length p using num_colors colors
            for pattern in itertools.product(range(num_colors), repeat=p):
                # Check if this pattern extended to 128 is valid
                colors = [pattern[i % p] for i in range(128)]
                
                # Quick check: count distinct colors actually used
                actual_colors = set(pattern)
                if len(actual_colors) > num_colors:
                    continue
                if len(actual_colors) < 2:
                    continue
                
                # Validate
                if _check_validity(colors):
                    cnt = len(actual_colors)
                    if cnt < best_count:
                        best_count = cnt
                        best = list(colors)
                    if best_count == 2:
                        return best
    
    return best

def _search_direct(deadline):
    """Direct greedy/backtracking search for minimal colors."""
    n = 128
    aps = _precompute_aps(n)
    
    # For 2 colors: every 4-term AP must have exactly 2 of one color and 2 of the other
    for pattern in itertools.product([0, 1], repeat=4):
        colors = [pattern[i % 4] for i in range(n)]
        valid = True
        for (a, b, c, d) in aps:
            vals = {colors[a], colors[b], colors[c], colors[d]}
            if len(vals) < 3:
                valid = False
                break
        if valid:
            return colors
    
    for pattern in itertools.product([0, 1], repeat=6):
        colors = [pattern[i % 6] for i in range(n)]
        valid = True
        for (a, b, c, d) in aps:
            vals = {colors[a], colors[b], colors[c], colors[d]}
            if len(vals) < 3:
                valid = False
                break
        if valid:
            return colors
    
    for pattern in itertools.product([0, 1], repeat=8):
        colors = [pattern[i % 8] for i in range(n)]
        valid = True
        for (a, b, c, d) in aps:
            vals = {colors[a], colors[b], colors[c], colors[d]}
            if len(vals) < 3:
                valid = False
                break
        if valid:
            return colors
    
    for p in [4, 6, 8, 10, 12]:
        for pattern in itertools.product([0, 1, 2], repeat=p):
            actual = set(pattern)
            if len(actual) < 2:
                continue
            colors = [pattern[i % p] for i in range(n)]
            valid = True
            for (a, b, c, d) in aps:
                vals = {colors[a], colors[b], colors[c], colors[d]}
                if len(vals) < 3:
                    valid = False
                    break
            if valid:
                return colors
    
    return None

def construct():
    """Return n=128 integer color labels in 0..127. Every nonconstant four-term progression
    of positions must use AT LEAST three distinct colors. Minimize number of labels used."""
    
    deadline = time.time() + 80  # safety margin
    
    # First try group-theoretic approach (most promising for finding few-color solutions)
    result = _search_group_theoretic(deadline)
    if result is not None:
        if _check_validity(result):
            return result
    
    # Try direct search
    result = _search_direct(deadline)
    if result is not None:
        if _check_validity(result):
            return result
    
    # Try periodic search
    result = _search_periodic(deadline)
    if result is not None:
        if _check_validity(result):
            return result
    
    # Fallback: greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3 + 1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2 + 1))):
                colors.append(color)
                break
    
    return colors
# EVOLVE-BLOCK-END
