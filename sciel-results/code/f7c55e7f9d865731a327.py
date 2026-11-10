"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import numpy as np
from math import log

N = 16
NUM_PAIRS = N * (N - 1) // 2  # 120
MAX_COLOR = 119  # colors 0..119

def pair_index(i, j):
    """Return the index in the flat array for pair (i,j) where i<j."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute the three cycles for each 4-subset
def build_cycles():
    from itertools import combinations
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        e1 = [pair_index(a,b), pair_index(b,c), pair_index(c,d), pair_index(d,a)]
        e2 = [pair_index(a,b), pair_index(b,d), pair_index(d,c), pair_index(c,a)]
        e3 = [pair_index(a,c), pair_index(c,b), pair_index(b,d), pair_index(d,a)]
        cycles.append(tuple(e1))
        cycles.append(tuple(e2))
        cycles.append(tuple(e3))
    return cycles

ALL_CYCLES = build_cycles()
NUM_CYCLES = len(ALL_CYCLES)

# Precompute for each edge index, which cycles it belongs to
EDGE_CYCLES = [[] for _ in range(NUM_PAIRS)]
for ci, cyc in enumerate(ALL_CYCLES):
    for idx in cyc:
        EDGE_CYCLES[idx].append(ci)

def count_colors(coloring):
    return len(set(coloring))

def is_valid(coloring):
    for cyc in ALL_CYCLES:
        c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_violation(coloring, cyc):
    c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
    return len({c0, c1, c2, c3}) == 4

def compute_entropy(coloring):
    """Compute Shannon entropy of the color distribution."""
    if not coloring:
        return 0.0
    from collections import Counter
    counts = Counter(coloring)
    total = len(coloring)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * log(p)
    return entropy

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    best_coloring = None
    best_score = 0
    
    rng = random.Random(42)
    
    def repair(coloring):
        """Fix violations by forcing edges in violating cycles to match."""
        for _ in range(10000):
            if time.time() > deadline - 5.0:
                break
            violated = False
            # Shuffle cycles to avoid bias
            cyc_list = ALL_CYCLES[:]
            rng.shuffle(cyc_list)
            for cyc in cyc_list:
                if check_violation(coloring, cyc):
                    violated = True
                    # Pick two edges in the cycle with different colors
                    c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
                    # Find a pair to merge
                    # Try to merge edge 0 with one of the others
                    # Or just pick a random edge and set it to a random other edge's color
                    idx_to_change = rng.choice(cyc)
                    # Pick a target color from the other 3 edges
                    other_indices = [i for i in cyc if i != idx_to_change]
                    target_color = coloring[rng.choice(other_indices)]
                    coloring[idx_to_change] = target_color
                    break
            if not violated:
                break
        return is_valid(coloring)
    
    def parametric_coloring(a, b, c, M):
        """Generate coloring: color(i,j) = (a*i + b*j + c) % M, mapped to 0..119"""
        coloring = []
        for i in range(N):
            for j in range(i+1, N):
                coloring.append((a*i + b*j + c) % M)
        return coloring
    
    def simulated_annealing(coloring, max_iters=50000):
        """Simulated annealing with entropy-based objective to maximize color diversity."""
        nonlocal best_score, best_coloring
        
        current_coloring = coloring[:]
        current_score = count_colors(current_coloring)
        current_entropy = compute_entropy(current_coloring)
        
        # Track best
        best_local_score = current_score
        best_local_coloring = current_coloring[:]
        
        # Annealing parameters
        T_start = 10.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
        
        used_colors = set(current_coloring)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Temperature
            T = T_start * (cooling_rate ** it)
            
            idx = rng.randint(0, NUM_PAIRS - 1)
            old_color = current_coloring[idx]
            
            # Strategy: 70% try new unused color, 30% try random color
            if rng.random() < 0.7 and len(used_colors) < MAX_COLOR + 1:
                unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                if unused:
                    new_color = rng.choice(unused)
                else:
                    new_color = rng.randint(0, MAX_COLOR)
            else:
                new_color = rng.randint(0, MAX_COLOR)
            
            if new_color == old_color:
                continue
            
            current_coloring[idx] = new_color
            
            # Check validity
            valid = True
            for ci in EDGE_CYCLES[idx]:
                cyc = ALL_CYCLES[ci]
                c0, c1, c2, c3 = current_coloring[cyc[0]], current_coloring[cyc[1]], current_coloring[cyc[2]], current_coloring[cyc[3]]
                if len({c0, c1, c2, c3}) == 4:
                    valid = False
                    break
            
            if not valid:
                # Revert
                current_coloring[idx] = old_color
                continue
            
            # Compute new metrics
            new_score = count_colors(current_coloring)
            new_entropy = compute_entropy(current_coloring)
            
            # Objective: maximize score, with entropy as secondary criterion
            # Use a combined objective: score * 10 + entropy
            old_obj = current_score * 10 + current_entropy
            new_obj = new_score * 10 + new_entropy
            
            delta_obj = new_obj - old_obj
            
            # Accept if better, or with probability exp(delta_obj / T) if worse
            if delta_obj >= 0 or rng.random() < np.exp(delta_obj / T):
                # Accept the move
                # Update used_colors
                if new_color not in used_colors:
                    used_colors.add(new_color)
                if old_color in used_colors:
                    if not any(current_coloring[k] == old_color for k in range(NUM_PAIRS) if k != idx):
                        used_colors.discard(old_color)
                
                current_score = new_score
                current_entropy = new_entropy
                
                # Update best local
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_coloring = current_coloring[:]
            else:
                # Revert
                current_coloring[idx] = old_color
        
        # Update global best
        if best_local_score > best_score:
            best_score = best_local_score
            best_coloring = best_local_coloring[:]
        
        return best_coloring, best_score
    
    def local_search(coloring, max_iters=50000):
        """Local search to maximize color count."""
        nonlocal best_score, best_coloring
        current_score = count_colors(coloring)
        used_colors = set(coloring)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            idx = rng.randint(0, NUM_PAIRS - 1)
            old_color = coloring[idx]
            
            # Strategy: 80% try new color, 20% try existing color (shuffle)
            if rng.random() < 0.8:
                # Try a new unused color
                if len(used_colors) < MAX_COLOR + 1:
                    unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                    if unused:
                        new_color = rng.choice(unused)
                    else:
                        new_color = rng.randint(0, MAX_COLOR)
                else:
                    new_color = rng.randint(0, MAX_COLOR)
            else:
                # Try an existing color (might merge, reducing score, but helps exploration)
                if used_colors:
                    new_color = rng.choice(list(used_colors))
                else:
                    new_color = rng.randint(0, MAX_COLOR)
            
            if new_color == old_color:
                continue
            
            coloring[idx] = new_color
            
            # Check validity
            valid = True
            for ci in EDGE_CYCLES[idx]:
                cyc = ALL_CYCLES[ci]
                c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
                if len({c0, c1, c2, c3}) == 4:
                    valid = False
                    break
            
            if valid:
                # Update used colors
                if new_color not in used_colors:
                    used_colors.add(new_color)
                if old_color in used_colors:
                    if not any(coloring[k] == old_color for k in range(NUM_PAIRS) if k != idx):
                        used_colors.discard(old_color)
                
                new_score = len(used_colors)
                if new_score > current_score:
                    current_score = new_score
                    # Update best
                    if current_score > best_score:
                        best_score = current_score
                        best_coloring = coloring[:]
                # Accept if score >= current_score - 1 (allow small drops)
                if new_score < current_score - 1:
                    # Revert if drop is too big
                    coloring[idx] = old_color
                    if old_color not in used_colors:
                        used_colors.add(old_color)
                    if new_color in used_colors:
                         if not any(coloring[k] == new_color for k in range(NUM_PAIRS) if k != idx):
                            used_colors.discard(new_color)
            else:
                coloring[idx] = old_color
        
        return best_coloring, best_score
    
    # Initialization: Try parametric colorings and random low-color palettes
    param_configs = []
    for M in [17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        for a in range(1, min(M, 8)):
            for b in range(1, min(M, 8)):
                for c in range(0, 3):
                    param_configs.append((a, b, c, M))
    
    rng.shuffle(param_configs)
    
    # Try parametric initializations
    for a, b, c, M in param_configs[:50]:
        if time.time() > deadline - 15.0:
            break
        
        coloring = parametric_coloring(a, b, c, M)
        
        # Repair
        if repair(coloring):
            # Update best with the repaired coloring if it's better
            rep_score = count_colors(coloring)
            if rep_score > best_score:
                best_score = rep_score
                best_coloring = coloring[:]
            
            # Simulated Annealing
            _, score = simulated_annealing(coloring, max_iters=15000)
    
    # Also try random initializations
    for init_attempt in range(10):
        if time.time() > deadline - 10.0:
            break
        
        num_init_colors = rng.randint(3, 8)
        coloring = [rng.randint(0, num_init_colors - 1) for _ in range(NUM_PAIRS)]
        
        # Repair
        if repair(coloring):
            rep_score = count_colors(coloring)
            if rep_score > best_score:
                best_score = rep_score
                best_coloring = coloring[:]
            
            # Simulated Annealing
            _, score = simulated_annealing(coloring, max_iters=10000)
    
    # If we have a good result, do more aggressive refinement
    if best_coloring is not None and best_score > 0:
        # Further improvement: try to increase color count with targeted moves
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for idx in range(NUM_PAIRS):
                if time.time() > deadline - 2.0:
                    break
                old_color = best_coloring[idx]
                used_colors = set(best_coloring)
                for new_color in range(MAX_COLOR + 1):
                    if new_color in used_colors:
                        continue
                    best_coloring[idx] = new_color
                    # Check affected cycles
                    valid = True
                    for ci in EDGE_CYCLES[idx]:
                        cyc = ALL_CYCLES[ci]
                        c0, c1, c2, c3 = best_coloring[cyc[0]], best_coloring[cyc[1]], best_coloring[cyc[2]], best_coloring[cyc[3]]
                        if len({c0, c1, c2, c3}) == 4:
                            valid = False
                            break
                    if valid:
                        improved = True
                        break
                    else:
                        best_coloring[idx] = old_color
            if not improved:
                break
    
    if best_coloring is None:
        return [0] * NUM_PAIRS
    
    return best_coloring

# EVOLVE-BLOCK-END
