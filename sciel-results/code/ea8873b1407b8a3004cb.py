"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precompute edge indices for lexicographic ordering
EDGE_INDEX = {}
for i in range(N):
    for j in range(i + 1, N):
        EDGE_INDEX[(i, j)] = i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles (sets of 4 vertices) and the 3 possible cycle edge sets
# For 4 vertices {a,b,c,d}, the three 4-cycles are:
# (a,b,c,d) -> edges (a,b),(b,c),(c,d),(d,a)
# (a,b,d,c) -> edges (a,b),(b,d),(d,c),(c,a)
# (a,c,b,d) -> edges (a,c),(c,b),(b,d),(d,a)
# We need to check that no 4-cycle has 4 distinct edge colors.

FOUR_VERT_SUBSETS = []
CYCLES_PER_SUBSET = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                FOUR_VERT_SUBSETS.append((a, b, c, d))
                # Three cycles on {a,b,c,d}
                # Cycle 1: a-b-c-d-a
                edges1 = [(a,b), (b,c), (c,d), (d,a)]
                # Cycle 2: a-b-d-c-a
                edges2 = [(a,b), (b,d), (d,c), (c,a)]
                # Cycle 3: a-c-b-d-a
                edges3 = [(a,c), (c,b), (b,d), (d,a)]
                CYCLES_PER_SUBSET.append((edges1, edges2, edges3))

def count_distinct_colors(colors):
    """Count distinct colors in a coloring."""
    return len(set(colors))

def is_valid(colors):
    """Check if no 4-cycle has 4 distinct edge colors."""
    for idx, (a, b, c, d) in enumerate(FOUR_VERT_SUBSETS):
        edges1, edges2, edges3 = CYCLES_PER_SUBSET[idx]
        for edges in (edges1, edges2, edges3):
            cset = set()
            for (u, v) in edges:
                if u > v:
                    u, v = v, u
                cset.add(colors[EDGE_INDEX[(u, v)]])
            if len(cset) == 4:
                return False
    return True

def recursive_block_coloring(n, offset):
    """
    Recursively construct a rainbow-4-cycle-free coloring for K_n.
    n must be a power of 2.
    Returns a list of colors for edges in lexicographic order for vertices 0..n-1.
    Uses offset for color shifting.
    """
    if n == 1:
        return []
    if n == 2:
        return [offset]
    
    half = n // 2
    # Left half: vertices 0..half-1
    left_colors = recursive_block_coloring(half, offset)
    # Right half: vertices half..n-1
    right_colors = recursive_block_coloring(half, offset + len(left_colors))
    
    # Inter-block color
    inter_color = offset + len(left_colors) + len(right_colors)
    
    # Build full edge list
    # Edges are in lexicographic order: (i,j) with i<j
    # Left edges: (i,j) where i<j<half
    # Right edges: (i,j) where half<=i<j<n
    # Inter edges: (i,j) where i<half<=j
    
    result = [0] * (n * (n - 1) // 2)
    
    left_edge_count = half * (half - 1) // 2
    right_edge_count = half * (half - 1) // 2
    inter_edge_count = half * half
    
    idx = 0
    # Fill in order: first all left edges, then inter edges, then right edges?
    # No, lexicographic order is by (i,j).
    # For i in 0..n-1:
    #   for j in i+1..n-1:
    #     if i < half and j < half: left edge
    #     if i < half and j >= half: inter edge
    #     if i >= half and j >= half: right edge
    
    left_idx = 0
    right_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if i < half and j < half:
                result[idx] = left_colors[left_idx]
                left_idx += 1
            elif i < half and j >= half:
                result[idx] = inter_color
            else:  # i >= half and j >= half
                result[idx] = right_colors[right_idx]
                right_idx += 1
            idx += 1
    
    return result

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0  # 50 second deadline, leaving 10s margin
    
    # Start with the recursive block construction
    base_colors = recursive_block_coloring(N, 0)
    best_colors = base_colors[:]
    best_score = count_distinct_colors(base_colors)
    
    # Now try to improve by local search: try to replace some colors with new ones
    # while maintaining validity.
    
    # Strategy: for each edge, try assigning a new color (from 0 to 119) and check validity.
    # This is too slow for full search, so use a smarter approach.
    
    # Alternative: use simulated annealing or random local search.
    # Start from base_colors and try to increase color count.
    
    current_colors = base_colors[:]
    current_score = best_score
    
    # Try to add new colors by modifying edges
    # For each edge, if we change its color to a new one, check if validity is maintained.
    # To speed up, only check the 4-cycles that include that edge.
    
    # Precompute which 4-cycles include each edge
    EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
    for idx, (a, b, c, d) in enumerate(FOUR_VERT_SUBSETS):
        edges1, edges2, edges3 = CYCLES_PER_SUBSET[idx]
        for edges in (edges1, edges2, edges3):
            for (u, v) in edges:
                if u > v:
                    u, v = v, u
                eidx = EDGE_INDEX[(u, v)]
                EDGE_TO_CYCLES[eidx].append(idx)
    
    # Local search: try to change each edge to a new color
    # Use a greedy approach: try edges in random order, assign new colors if valid
    
    rng = random.Random(42)
    edge_order = list(range(NUM_EDGES))
    
    # Try multiple rounds of local improvement
    for round_num in range(200):
        if time.time() > deadline:
            break
        
        # Shuffle edge order
        rng.shuffle(edge_order)
        
        improved = False
        for eidx in edge_order:
            if time.time() > deadline:
                break
            
            old_color = current_colors[eidx]
            
            # Try to assign a new color
            # Choose a color not currently used on this edge, preferably a new one
            # Try colors from 0 to 119
            used_colors = set(current_colors)
            
            # Try to find a color that is either already used (to swap) or new
            # First, try to use a color that doesn't create a 4-distinct cycle
            # Check all 4-cycles containing this edge
            
            candidate_colors = list(range(120))
            rng.shuffle(candidate_colors)
            
            for new_color in candidate_colors:
                if new_color == old_color:
                    continue
                
                # Temporarily assign
                current_colors[eidx] = new_color
                
                # Check validity for all 4-cycles containing this edge
                valid = True
                for cycle_idx in EDGE_TO_CYCLES[eidx]:
                    a, b, c, d = FOUR_VERT_SUBSETS[cycle_idx]
                    edges1, edges2, edges3 = CYCLES_PER_SUBSET[cycle_idx]
                    for edges in (edges1, edges2, edges3):
                        cset = set()
                        for (u, v) in edges:
                            if u > v:
                                u, v = v, u
                            cset.add(current_colors[EDGE_INDEX[(u, v)]])
                        if len(cset) == 4:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Accept the change
                    new_score = count_distinct_colors(current_colors)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_colors = current_colors[:]
                    improved = True
                    break
                else:
                    # Revert
                    current_colors[eidx] = old_color
        
        if not improved:
            # No improvement in this round, try a different starting point
            # Or break if we've done enough rounds
            if round_num > 50:
                break
    
    # Return the best coloring found
    return best_colors
# EVOLVE-BLOCK-END
