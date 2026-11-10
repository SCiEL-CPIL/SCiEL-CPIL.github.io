"""Rewritten program for minimizing monochromatic 4-term APs in [1,256]."""
import time
import random
import numpy as np

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 1..256 (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _count_mono(aps, colors):
    """Count monochromatic 4-APs."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _count_mono_fast(aps_arr, colors):
    """Vectorized count of monochromatic 4-APs."""
    # aps_arr shape (M, 4), colors shape (N,)
    c = colors[aps_arr]  # shape (M, 4)
    # mono if all 4 equal: c[:,0]==c[:,1]==c[:,2]==c[:,3]
    return int(np.all(c == c[:, 0:1], axis=1).sum())

def _local_search(aps_arr, colors, max_iter=200, time_budget=10.0):
    """Greedy local search: flip bits that reduce mono-AP count."""
    colors = list(colors)
    n = len(colors)
    best = _count_mono_fast(aps_arr, colors)
    
    # Build index: for each position, which APs involve it
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(aps_arr):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[d].append(idx)
    
    start = time.time()
    improved = True
    iters = 0
    while improved and iters < max_iter and (time.time() - start) < time_budget:
        improved = False
        iters += 1
        for i in range(n):
            # Compute delta if we flip i
            # Current contribution: for each AP involving i, check if mono
            # After flip, check again
            cur_mono = 0
            new_mono = 0
            for ap_idx in pos_aps[i]:
                a, b, c, d = aps_arr[ap_idx]
                ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
                if ca == cb == cc == cd:
                    cur_mono += 1
                # After flipping i
                if i == a:
                    na = 1 - ca
                elif i == b:
                    nb = 1 - cb
                elif i == c:
                    nc = 1 - cc
                elif i == d:
                    nd = 1 - cd
                else:
                    continue
                # Check if new coloring is mono
                vals = []
                if i == a:
                    vals = [na, cb, cc, cd]
                elif i == b:
                    vals = [ca, nb, cc, cd]
                elif i == c:
                    vals = [ca, cb, nc, cd]
                elif i == d:
                    vals = [ca, cb, cc, nd]
                if len(set(vals)) == 1:
                    new_mono += 1
            delta = new_mono - cur_mono
            if delta < 0:
                colors[i] = 1 - colors[i]
                best += delta
                improved = True
    
    return colors, best

def _simulated_annealing(aps_arr, colors, max_iter=5000, time_budget=20.0):
    """Simulated annealing to escape local minima."""
    colors = list(colors)
    n = len(colors)
    best = _count_mono_fast(aps_arr, colors)
    
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(aps_arr):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[d].append(idx)
    
    T = 10.0
    cooling = 0.9995
    start = time.time()
    iters = 0
    while iters < max_iter and (time.time() - start) < time_budget:
        iters += 1
        i = random.randint(0, n - 1)
        old_color = colors[i]
        colors[i] = 1 - old_color
        
        # Compute delta
        cur_mono = 0
        new_mono = 0
        for ap_idx in pos_aps[i]:
            a, b, c, d = aps_arr[ap_idx]
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            # This is the NEW state (after flip)
            if ca == cb == cc == cd:
                new_mono += 1
            # Old state: flip i back
            if i == a:
                na = old_color
            elif i == b:
                nb = old_color
            elif i == c:
                nc = old_color
            elif i == d:
                nd = old_color
            vals = []
            if i == a:
                vals = [na, cb, cc, cd]
            elif i == b:
                vals = [ca, nb, cc, cd]
            elif i == c:
                vals = [ca, cb, nc, cd]
            elif i == d:
                vals = [ca, cb, cc, nd]
            if len(set(vals)) == 1:
                cur_mono += 1
        delta = new_mono - cur_mono
        
        if delta <= 0 or random.random() < np.exp(-delta / T):
            best = min(best, _count_mono_fast(aps_arr, colors))
        else:
            colors[i] = old_color
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return colors, best

def construct():
    """Return 256 colours in {0,1} minimizing monochromatic 4-APs."""
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    aps = _build_aps()
    aps_arr = np.array(aps, dtype=np.int32)
    
    # Initial coloring: try a few heuristics
    # 1. Block of 2 alternating
    colors1 = [(i // 2) % 2 for i in range(256)]
    # 2. Random
    random.seed(42)
    colors2 = [random.randint(0, 1) for _ in range(256)]
    # 3. Thue-Morse-like (van der Corput)
    colors3 = [bin(i).count('1') % 2 for i in range(256)]
    # 4. Block of 4: 0011 pattern
    colors4 = [(i // 4) % 2 for i in range(256)]
    # 5. Period 5: 00110 pattern
    colors5 = [((i % 5) < 2) for i in range(256)]
    
    candidates = [colors1, colors2, colors3, colors4, colors5]
    
    best_colors = None
    best_score = float('inf')
    
    for c in candidates:
        s = _count_mono_fast(aps_arr, c)
        if s < best_score:
            best_score = s
            best_colors = c[:]
    
    # Local search on best
    if time.time() < deadline - 20:
        best_colors, best_score = _local_search(aps_arr, best_colors, max_iter=100, time_budget=15.0)
    
    # Try simulated annealing from a few random restarts
    while time.time() < deadline - 30:
        random.seed(int(time.time() * 1000) % 100000)
        c = [random.randint(0, 1) for _ in range(256)]
        # Quick local search first
        c, s = _local_search(aps_arr, c, max_iter=30, time_budget=5.0)
        if s < best_score:
            best_score = s
            best_colors = c[:]
        # Then SA
        c, s = _simulated_annealing(aps_arr, c, max_iter=3000, time_budget=10.0)
        if s < best_score:
            best_score = s
            best_colors = c[:]
    
    # Final local search polish
    if time.time() < deadline - 10:
        best_colors, best_score = _local_search(aps_arr, best_colors, max_iter=50, time_budget=8.0)
    
    return best_colors

# EVOLVE-BLOCK-END
