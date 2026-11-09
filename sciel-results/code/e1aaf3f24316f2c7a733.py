import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions as index arrays
    prog_a = []
    prog_d = []
    for a in range(n):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 < n:
                prog_a.append(a)
                prog_d.append(d)
    
    num_progs = len(prog_a)
    if num_progs == 0:
        return list(range(n))
    
    prog_a = np.array(prog_a, dtype=np.int32)
    prog_d = np.array(prog_d, dtype=np.int32)
    
    # Index arrays for vectorized access
    idx0 = prog_a
    idx1 = prog_a + prog_d
    idx2 = prog_a + 2 * prog_d
    idx3 = prog_a + 3 * prog_d
    
    def count_violations_vec(colors):
        """Vectorized violation count using NumPy"""
        c0 = colors[idx0]
        c1 = colors[idx1]
        c2 = colors[idx2]
        c3 = colors[idx3]
        
        # Violation if < 3 distinct colors, i.e., <= 2 distinct
        # Count distinct: check patterns
        # All same
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        # Three same, one different
        three_same = ((c0 == c1) & (c1 == c2) & (c2 != c3)) | \
                     ((c0 == c1) & (c1 != c2) & (c2 == c3)) | \
                     ((c0 != c1) & (c1 == c2) & (c2 == c3)) | \
                     ((c0 == c1) & (c1 != c2) & (c2 != c3) & (c1 == c3))
        # Two pairs: c0=c1, c2=c3, c0!=c2
        two_pairs_1 = (c0 == c1) & (c2 == c3) & (c0 != c2)
        # c0=c2, c1=c3, c0!=c1
        two_pairs_2 = (c0 == c2) & (c1 == c3) & (c0 != c1)
        # c0=c3, c1=c2, c0!=c1
        two_pairs_3 = (c0 == c3) & (c1 == c2) & (c0 != c1)
        
        violation = all_same | three_same | two_pairs_1 | two_pairs_2 | two_pairs_3
        return np.sum(violation)
    
    # Try a known good construction: use 4 colors
    # c(i) = (i mod 2) * 2 + (i // 2 mod 2)
    colors4 = np.array([(i % 2) * 2 + ((i // 2) % 2) for i in range(n)], dtype=np.int32)
    v4 = count_violations_vec(colors4)
    if v4 == 0:
        return colors4.tolist()
    
    # Try 6 colors: c(i) = i mod 6
    colors6 = np.array([i % 6 for i in range(n)], dtype=np.int32)
    v6 = count_violations_vec(colors6)
    if v6 == 0:
        return colors6.tolist()
    
    # Try 9 colors with 2D structure
    colors9 = np.array([(i % 3) * 3 + ((i // 3) % 3) for i in range(n)], dtype=np.int32)
    v9 = count_violations_vec(colors9)
    if v9 == 0:
        return colors9.tolist()
    
    # Best fallback: use 128 colors (trivially valid)
    colors_full = np.arange(n, dtype=np.int32)
    vf = count_violations_vec(colors_full)
    if vf == 0:
        return colors_full.tolist()
    
    # Local search with vectorized evaluation
    best_colors = colors9.copy()
    best_viol = v9
    
    # Simulated annealing
    np.random.seed(42)
    current = best_colors.copy()
    current_viol = best_viol
    
    t_start = time.time()
    max_iter = 100000
    iter_count = 0
    
    while iter_count < max_iter and time.time() < deadline - 5:
        iter_count += 1
        # Pick random position
        pos = np.random.randint(n)
        old_c = current[pos]
        # Pick random new color
        new_c = np.random.randint(128)
        if new_c == old_c:
            continue
        
        current[pos] = new_c
        new_viol = count_violations_vec(current)
        delta = new_viol - current_viol
        
        if delta <= 0 or np.random.random() < np.exp(-delta / max(0.1, 1.0 - iter_count / max_iter)):
            current_viol = new_viol
            if new_viol < best_viol:
                best_viol = new_viol
                best_colors = current.copy()
                if best_viol == 0:
                    return best_colors.tolist()
        else:
            current[pos] = old_c
    
    return best_colors.tolist()
