"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    """Return the index in the lexicographic edge list for edge (i,j) with i<j."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles as tuples of 4 edge indices
# For each 4-vertex subset {a,b,c,d} with a<b<c<d, the 3 cycles are:
#   (a,b,c,d): edges (a,b),(b,c),(c,d),(d,a)
#   (a,b,d,c): edges (a,b),(b,d),(d,c),(c,a)
#   (a,c,b,d): edges (a,c),(c,b),(b,d),(d,a)
_CYCLES = []
for a, b, c, d in combinations(range(N), 4):
    # Cycle 1: a-b-c-d-a
    _CYCLES.append((_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)))
    # Cycle 2: a-b-d-c-a
    _CYCLES.append((_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)))
    # Cycle 3: a-c-b-d-a
    _CYCLES.append((_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)))

_NUM_CYCLES = len(_CYCLES)
_CYCLE_ARRAY = np.array(_CYCLES, dtype=np.int32)

def _count_colors(colors):
    """Count the number of distinct colors used."""
    return len(set(colors))

def _is_valid(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    c_arr = np.array(colors, dtype=np.int32)
    for cyc in _CYCLES:
        e0, e1, e2, e3 = cyc
        if len({c_arr[e0], c_arr[e1], c_arr[e2], c_arr[e3]}) == 4:
            return False
    return True

def _find_violation(colors):
    """Find a violating 4-cycle (four distinct colors) or return None."""
    c_arr = np.array(colors, dtype=np.int32)
    for cyc in _CYCLES:
        e0, e1, e2, e3 = cyc
        if len({c_arr[e0], c_arr[e1], c_arr[e2], c_arr[e3]}) == 4:
            return cyc
    return None

def construct():
    deadline = time.time() + 85.0
    random.seed(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1  # all zeros means 1 color
    
    # Start with a greedy approach: assign colors to maximize diversity
    # while avoiding 4-distinct-color cycles
    
    # Strategy: Use a small palette and try to extend it
    # Start with 2 colors, then try to add more
    
    def _greedy_build(num_colors):
        """Build a coloring using at most num_colors colors, trying to use all of them."""
        colors = [0] * NUM_EDGES
        used = set()
        
        # Assign edges in a structured way
        # Use a round-robin-like assignment to spread colors
        for idx in range(NUM_EDGES):
            # Try to assign a color that doesn't create a violation
            assigned = False
            for c in range(num_colors):
                colors[idx] = c
                # Check if this creates any new violation involving edge idx
                valid = True
                for cyc in _CYCLES:
                    if idx in cyc:
                        e0, e1, e2, e3 = cyc
                        if len({colors[e0], colors[e1], colors[e2], colors[e3]}) == 4:
                            valid = False
                            break
                if valid:
                    used.add(c)
                    assigned = True
                    break
            if not assigned:
                colors[idx] = 0
                used.add(0)
        
        return colors, len(used)
    
    # Try different numbers of colors
    for nc in range(2, 120):
        if time.time() > deadline:
            break
        colors, used_count = _greedy_build(nc)
        if used_count > best_count:
            best_colors = colors[:]
            best_count = used_count
            # Try to improve further
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try swapping colors on random edges
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, NUM_EDGES - 1)
                    old_c = colors[idx]
                    new_c = random.randint(0, nc + 10)
                    if new_c == old_c:
                        continue
                    colors[idx] = new_c
                    if _is_valid(colors):
                        new_used = len(set(colors))
                        if new_used > best_count:
                            best_colors = colors[:]
                            best_count = new_used
                            improved = True
                    else:
                        colors[idx] = old_c
    
    return best_colors

# EVOLVE-BLOCK-END
