import time
import random
import numpy as np
from itertools import combinations

_N = 16
_NUM_EDGES = _N * (_N - 1) // 2
_K4S = list(combinations(range(_N), 4))
_NUM_K4S = len(_K4S)

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute K4 edge indices
_K4_EDGE_IDXS = []
for quad in _K4S:
    edges = []
    for a in range(4):
        for b in range(a + 1, 4):
            i, j = quad[a], quad[b]
            edges.append(_edge_idx(i, j))
    _K4_EDGE_IDXS.append(edges)

# Precompute for each edge, the list of (k4_idx, pos_in_k4)
_EDGE_TO_K4S = [[] for _ in range(_NUM_EDGES)]
for k4_idx, edges in enumerate(_K4_EDGE_IDXS):
    for pos, e_idx in enumerate(edges):
        _EDGE_TO_K4S[e_idx].append((k4_idx, pos))

# Precompute for each K4 and each position, the indices of the other 5 edges
# This allows fast mask calculation for the "other" colors
_K4_OTHER_EDGES = []
for k4_idx in range(_NUM_K4S):
    e_idxs = _K4_EDGE_IDXS[k4_idx]
    others = []
    for pos in range(6):
        others.append([e_idxs[p] for p in range(6) if p != pos])
    _K4_OTHER_EDGES.append(others)

def _count_bad_k4s(colors):
    """Count K4s with fewer than 5 distinct colors."""
    bad = 0
    for k4_idx in range(_NUM_K4S):
        e_idxs = _K4_EDGE_IDXS[k4_idx]
        mask = 0
        for e in e_idxs:
            mask |= (1 << colors[e])
        if mask.bit_count() < 5:
            bad += 1
    return bad

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    best_colors = None
    best_num_colors = 100  # Initialize high
    
    # We want to minimize colors used.
    # Try target colors in descending order of likelihood of success: 15, 14, 13
    # Start with 15 as 13 is very hard.
    
    # Helper to run SA for a specific target number of colors
    def run_sa(target_colors, seed):
        rng = random.Random(seed)
        # Initialize randomly
        colors = np.array([rng.randint(0, target_colors - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
        
        bad = _count_bad_k4s(colors)
        
        # Simulated Annealing Parameters
        T_start = 1.0
        T_end = 0.01
        steps = 20000
        # We don't want to exceed deadline, so estimate time per step
        # ~ 91 K4s per edge, each check is fast.
        
        # If initial is already valid, great
        if bad == 0:
            return colors, bad
        
        # Annealing schedule
        # Use exponential cooling
        # T = T_start * (T_end/T_start)^(step/steps)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Calculate temperature
            # Simple linear or exponential
            # Let's use exponential
            progress = step / steps
            T = T_start * (T_end / T_start) ** progress
            
            # Pick random edge
            e_idx = rng.randint(0, _NUM_EDGES - 1)
            old_color = colors[e_idx]
            
            # Pick new color
            new_color = rng.randint(0, target_colors - 1)
            if new_color == old_color:
                continue
            
            # Calculate delta bad
            delta_bad = 0
            for k4_idx, pos in _EDGE_TO_K4S[e_idx]:
                others = _K4_OTHER_EDGES[k4_idx][pos]
                
                # Calculate mask of other 5 edges
                mask_others = 0
                for e in others:
                    mask_others |= (1 << colors[e])
                
                # Old distinct count
                mask_old = mask_others | (1 << old_color)
                d_old = mask_old.bit_count()
                
                # New distinct count
                mask_new = mask_others | (1 << new_color)
                d_new = mask_new.bit_count()
                
                # Bad if < 5
                bad_old = 1 if d_old < 5 else 0
                bad_new = 1 if d_new < 5 else 0
                
                delta_bad += bad_new - bad_old
            
            # Acceptance
            if delta_bad <= 0:
                colors[e_idx] = new_color
                bad += delta_bad
            else:
                # Prob = exp(-delta_bad / T)
                # If T is very small, prob is 0
                if T > 0.01:
                    prob = np.exp(-delta_bad / T)
                    if rng.random() < prob:
                        colors[e_idx] = new_color
                        bad += delta_bad
            
            if bad == 0:
                break
        
        return colors, bad

    # Try to find valid colorings with fewer colors
    # Start with 15, then 14, then 13
    # We prioritize lower color counts.
    
    # Strategy:
    # 1. Try 13 colors with a few restarts. If found, return.
    # 2. Try 14 colors with a few restarts. If found, return.
    # 3. Try 15 colors with a few restarts. If found, return.
    # 4. Fallback to 17 (known valid) or best found.
    
    # However, finding 13 or 14 might be rare.
    # Let's allocate time to each.
    
    # Time allocation:
    # 13 colors: 20s
    # 14 colors: 30s
    # 15 colors: 30s
    # Polish/Refine: 20s
    # Reserve: 45s (buffer)
    
    # Actually, let's just run a loop that tries to improve the best found.
    # Start with 15, as it's the upper bound of the known range and likely easier to find valid instances.
    # Then try to reduce.
    
    # Let's try 15 first.
    for target in [15, 14, 13]:
        if time.time() > deadline - 10:
            break
        
        # Number of restarts based on time remaining
        # Let's do 5 restarts for 15, 3 for 14, 2 for 13
        restarts = 5 if target == 15 else (3 if target == 14 else 2)
        
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
            
            # Seed for reproducibility
            seed = target * 100 + r
            c, bad = run_sa(target, seed)
            
            if bad == 0:
                # Valid coloring with 'target' colors?
                # Check actual colors used
                actual_colors = len(set(c.tolist()))
                if actual_colors < best_num_colors:
                    best_num_colors = actual_colors
                    best_colors = c.copy()
                
                # If we found a valid 15-coloring, we can try to polish it to use fewer.
                # But let's first see if we can find 14 or 13 directly in subsequent loops.
                # If target is 15 and we found valid, we break out of this target loop?
                # No, we might find a 14-coloring in the next target loop.
                # But if we found a 15-coloring, it's better than 17.
                
                # If target is 14 and we found valid, it's better than 15.
                # If target is 13 and we found valid, it's best.
                
                # Optimization: If we found a valid coloring with target colors,
                # we can try a "color removal" pass to see if we can reduce the count further.
                # This is essentially running SA with a penalty for using extra colors,
                # or just trying to recolor edges to use a smaller subset.
                
                # Let's implement a simple "try to remove color" heuristic.
                # Pick a color used, try to recolor all edges of that color to other colors.
                # This is hard.
                
                # Alternative: Just continue to next target.
                # If we found valid 15, next target is 14.
                # If we found valid 14, next target is 13.
                # If we found valid 13, we can stop.
                
                if target == 13:
                    # Found the minimum possible (theoretically)
                    return best_colors.tolist()
                
                # We can also try to improve the current best by running SA 
                # with the current best's color set, but restricting to a smaller palette?
                # No, just let the main loop handle targets.
                
            else:
                # If not valid, we can't use it.
                pass
    
    # If we didn't find anything better than 17, return the 17-coloring
    if best_colors is None:
        return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
    
    # Final validation
    if _count_bad_k4s(best_colors) > 0:
        return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
        
    return best_colors.tolist()

