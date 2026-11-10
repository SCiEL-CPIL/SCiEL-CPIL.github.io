import time
import random
import numpy as np

N = 256

def _build_aps():
    """Build all 4-term arithmetic progressions as (a, a+d, a+2d, a+3d) 0-indexed."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _count_mono(colors):
    """Count monochromatic 4-APs using vectorized check."""
    # colors is a numpy array of 0/1
    count = 0
    for d in range(1, N // 3 + 1):
        # All a such that a+3d < N
        max_a = N - 3 * d
        # Check if all four positions have same color
        for a in range(max_a):
            if colors[a] == colors[a+d] == colors[a+2*d] == colors[a+3*d]:
                count += 1
    return count

def _count_mono_fast(colors):
    """Faster count using numpy slicing."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        # Vectorized: check all starting positions at once
        c0 = colors[:max_a]
        c1 = colors[d:d+max_a]
        c2 = colors[2*d:2*d+max_a]
        c3 = colors[3*d:3*d+max_a]
        count += np.sum((c0 == c1) & (c1 == c2) & (c2 == c3))
    return int(count)

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Precompute all 4-APs
    aps = _build_aps()
    n_aps = len(aps)
    
    # Precompute for each position, which APs it participates in
    # pos_aps[i] = list of (ap_index, role) where role is 0,1,2,3
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[d].append((idx, 3))
    
    def _eval_flip(colors, pos):
        """Compute change in mono count if we flip position pos."""
        delta = 0
        for ap_idx, role in pos_aps[pos]:
            a, b, c, d = aps[ap_idx]
            # Current state: is this AP mono?
            ca, cb, cc, cd = colors[a], colors[b], colors[c], colors[d]
            was_mono = (ca == cb == cc == cd)
            
            # New state after flipping pos
            new_ca = 1 - ca if role == 0 else ca
            new_cb = 1 - cb if role == 1 else cb
            new_cc = 1 - cc if role == 2 else cc
            new_cd = 1 - cd if role == 3 else cd
            is_mono = (new_ca == new_cb == new_cc == new_cd)
            
            if is_mono and not was_mono:
                delta += 1
            elif was_mono and not is_mono:
                delta -= 1
        return delta
    
    # Start with random initialization
    best_colors = None
    best_score = float('inf')
    
    start_time = time.time()
    
    # Try multiple random restarts
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        colors = np.random.randint(0, 2, N)
        current_score = _count_mono_fast(colors)
        
        # Simulated annealing
        T_start = 5.0
        T_end = 0.1
        n_steps = 50000
        alpha = (T_end / T_start) ** (1.0 / n_steps)
        T = T_start
        
        best_in_run = colors.copy()
        best_in_run_score = current_score
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick random position to flip
            pos = random.randint(0, N - 1)
            delta = _eval_flip(colors, pos)
            
            # Accept if better or with probability exp(-delta/T)
            if delta <= 0:
                colors[pos] = 1 - colors[pos]
                current_score += delta
                if current_score < best_in_run_score:
                    best_in_run_score = current_score
                    best_in_run = colors.copy()
            else:
                if random.random() < np.exp(-delta / T):
                    colors[pos] = 1 - colors[pos]
                    current_score += delta
            
            T *= alpha
        
        # Track global best
        if best_in_run_score < best_score:
            best_score = best_in_run_score
            best_colors = best_in_run.copy()
        
        # If we've found a very good solution, do a local search polish
        if best_score <= 20:
            # Quick local search: try all single flips
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for pos in range(N):
                    delta = _eval_flip(best_colors, pos)
                    if delta < 0:
                        best_colors[pos] = 1 - best_colors[pos]
                        best_score += delta
                        improved = True
    
    # Final local search polish
    if best_colors is not None and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for pos in range(N):
                delta = _eval_flip(best_colors, pos)
                if delta < 0:
                    best_colors[pos] = 1 - best_colors[pos]
                    best_score += delta
                    improved = True
    
    if best_colors is None:
        # Fallback: alternating pattern
        best_colors = np.array([(i // 2) % 2 for i in range(N)])
    
    return [int(c) for c in best_colors]
