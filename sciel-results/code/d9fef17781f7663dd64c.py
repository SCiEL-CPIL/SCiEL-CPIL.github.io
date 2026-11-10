"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import numpy as np

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
                    # Check if old color is still used
                    # Optimization: maintain count, but for simplicity, recompute if needed
                    # For now, just update set if it becomes empty? 
                    # Actually, we can just check if any other edge has old_color
                    # This is O(N). Acceptable for N=120.
                    if not any(coloring[k] == old_color for k in range(NUM_PAIRS) if k != idx):
                        used_colors.discard(old_color)
                
                new_score = len(used_colors)
                if new_score > current_score:
                    current_score = new_score
                    # Update best
                    if current_score > best_score:
                        best_score = current_score
                        best_coloring = coloring[:]
                # Note: We accept moves that keep score same or decrease it slightly?
                # To escape local maxima, we should accept some decreases.
                # But for simplicity, let's only accept if score >= current_score 
                # OR if we are in a "shuffle" mode?
                # Let's accept if valid. The repair step handles invalidity.
                # But if we accept a decrease, we might drift.
                # Let's accept if new_score >= current_score - 1 (allow small drops)
                if new_score < current_score - 1:
                    # Revert if drop is too big
                    coloring[idx] = old_color
                    # Restore used_colors
                    if old_color not in used_colors:
                        used_colors.add(old_color)
                    if new_color in used_colors:
                         if not any(coloring[k] == new_color for k in range(NUM_PAIRS) if k != idx):
                            used_colors.discard(new_color)
            else:
                coloring[idx] = old_color
        
        return best_coloring, best_score

    # Initialization: Try several random low-color palettes
    for init_attempt in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Start with random colors from a small set
        num_init_colors = rng.randint(2, 6)
        coloring = [rng.randint(0, num_init_colors - 1) for _ in range(NUM_PAIRS)]
        
        # Repair
        if repair(coloring):
            # Update best with the repaired coloring if it's better
            rep_score = count_colors(coloring)
            if rep_score > best_score:
                best_score = rep_score
                best_coloring = coloring[:]
            
            # Local Search
            _, score = local_search(coloring, max_iters=10000)
        
    # If we didn't find anything, return a trivial valid coloring
    if best_coloring is None:
        return [0] * NUM_PAIRS
    
    return best_coloring

# EVOLVE-BLOCK-END
