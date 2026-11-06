"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge index mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each 4-vertex subset, the 3 cycles
def precompute_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on {a,b,c,d}:
        # (a-b-c-d-a): edges (a,b), (b,c), (c,d), (d,a)
        e1 = (edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a))
        # (a-b-d-c-a): edges (a,b), (b,d), (d,c), (c,a)
        e2 = (edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a))
        # (a-c-b-d-a): edges (a,c), (c,b), (b,d), (d,a)
        e3 = (edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a))
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles

ALL_CYCLES = precompute_cycles()
NUM_CYCLES = len(ALL_CYCLES)  # 5460

def count_colors(coloring):
    return len(set(coloring))

def is_valid(coloring):
    """Check if no 4-cycle has 4 distinct colors."""
    for e1, e2, e3, e4 in ALL_CYCLES:
        if len({coloring[e1], coloring[e2], coloring[e3], coloring[e4]}) == 4:
            return False
    return True

def count_violations(coloring):
    """Count number of 4-cycles with 4 distinct colors."""
    v = 0
    for e1, e2, e3, e4 in ALL_CYCLES:
        if len({coloring[e1], coloring[e2], coloring[e3], coloring[e4]}) == 4:
            v += 1
    return v

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    random.seed(42)
    
    # Start with all edges color 0
    best = [0] * NUM_EDGES
    best_score = 1  # 1 color used
    
    # Strategy: Greedily introduce new colors, then local search
    # Phase 1: Greedy construction - try to add each new color to some edges
    
    # First, let's try a structured approach based on vertex coloring
    # Partition vertices into groups and use group-based coloring
    
    # Try different partition sizes
    for num_groups in range(2, 9):
        if time.time() > deadline - 10:
            break
        
        # Assign vertices to groups
        group_assign = list(range(N))
        # Random permutation then assign
        random.shuffle(group_assign)
        
        # Assign groups: first num_groups vertices get groups 0..num_groups-1
        # Actually, let's do balanced assignment
        groups = [0] * N
        for i in range(N):
            groups[i] = i % num_groups
        
        # Build coloring based on group pairs
        # For each pair of groups (g1, g2), assign a color
        group_pairs = {}
        color_counter = 1  # start at 1, 0 is default
        
        for g1 in range(num_groups):
            for g2 in range(g1 + 1, num_groups):
                group_pairs[(g1, g2)] = color_counter
                color_counter += 1
        
        # Build coloring
        coloring = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                g1, g2 = groups[i], groups[j]
                key = (min(g1, g2), max(g1, g2))
                coloring[edge_idx(i, j)] = group_pairs.get(key, 0)
        
        # Check validity
        if count_violations(coloring) == 0:
            score = count_colors(coloring)
            if score > best_score:
                best_score = score
                best = coloring[:]
        
        # Now do local search to improve
        coloring = coloring[:]
        current_score = count_colors(coloring)
        current_violations = count_violations(coloring)
        
        # Local search: try changing individual edge colors
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_color = coloring[e]
            
            # Try a new color
            new_color = random.randint(0, 119)
            if new_color == old_color:
                continue
            
            coloring[e] = new_color
            
            # Check if still valid
            if count_violations(coloring) == 0:
                new_score = count_colors(coloring)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = coloring[:]
                else:
                    # Keep if same score (helps exploration)
                    if random.random() < 0.3:
                        current_score = new_score
            else:
                coloring[e] = old_color
        
        # Phase 2: Try to add new colors to edges that currently have common colors
        for iteration in range(200):
            if time.time() > deadline - 3:
                break
            
            # Find edges with most common colors
            from collections import Counter
            color_counts = Counter(coloring)
            common_colors = [c for c, cnt in color_counts.items() if cnt > 1]
            
            if not common_colors:
                break
            
            target_color = random.choice(common_colors)
            edges_with_color = [i for i in range(NUM_EDGES) if coloring[i] == target_color]
            
            if not edges_with_color:
                continue
            
            e = random.choice(edges_with_color)
            old_color = coloring[e]
            
            # Try a new unused color
            used_colors = set(coloring)
            available = [c for c in range(120) if c not in used_colors]
            if not available:
                break
            
            new_color = random.choice(available)
            coloring[e] = new_color
            
            if count_violations(coloring) == 0:
                new_score = count_colors(coloring)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = coloring[:]
            else:
                coloring[e] = old_color
    
    # Phase 3: Simulated annealing to maximize color count
    if time.time() < deadline - 5:
        coloring = best[:]
        current_score = count_colors(coloring)
        current_violations = count_violations(coloring)
        
        T = 5.0
        cooling = 0.995
        
        while time.time() < deadline - 2 and T > 0.1:
            T *= cooling
            
            e = random.randint(0, NUM_EDGES - 1)
            old_color = coloring[e]
            new_color = random.randint(0, 119)
            
            if new_color == old_color:
                continue
            
            coloring[e] = new_color
            
            new_violations = count_violations(coloring)
            new_score = count_colors(coloring)
            
            # Accept if valid and better, or probabilistically
            if new_violations == 0:
                delta_score = new_score - current_score
                if delta_score >= 0:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = coloring[:]
                else:
                    if random.random() < np.exp(delta_score / T):
                        current_score = new_score
            else:
                # Reject if introduces violations
                coloring[e] = old_color
    
    return best
# EVOLVE-BLOCK-END
