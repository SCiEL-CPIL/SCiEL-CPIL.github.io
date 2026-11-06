import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the lexicographic index of edge (i,j) where 0 <= i < j < N."""
    if i > j:
        i, j = j, i
    # Row i has edges (i, i+1), (i, i+2), ..., (i, N-1)
    # Number of edges before row i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Position within row i: j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def build_cycle_data():
    """Precompute all 4-cycles and which edges they contain.
    Returns list of (edges_tuple, vertex_set) for all 3 cycles on each 4-vertex subset."""
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on 4 vertices:
        # 1. a-b-c-d-a: edges (a,b), (b,c), (c,d), (d,a)
        # 2. a-b-d-c-a: edges (a,b), (b,d), (d,c), (c,a)
        # 3. a-c-b-d-a: edges (a,c), (c,b), (b,d), (d,a)
        cycles.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)))
        cycles.append((edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)))
        cycles.append((edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)))
    return cycles

def check_validity(colors, cycles):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for e1, e2, e3, e4 in cycles:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    """
    Construct a coloring of K_16 edges with no rainbow 4-cycle.
    Maximizes the number of distinct colors used.
    
    Strategy: Simulated annealing with multi-edge perturbations and
    multi-start. Uses fast incremental validity checking.
    """
    start_time = time.time()
    deadline = start_time + 75.0  # Leave 15s margin
    
    cycles = build_cycle_data()
    
    # Precompute: for each edge, which cycles contain it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3, e4) in enumerate(cycles):
        for e in (e1, e2, e3, e4):
            edge_to_cycles[e].append(ci)
    
    def is_valid_incremental(colors, changed_edges):
        """Check validity only for cycles containing any of the changed edges."""
        affected = set()
        for e in changed_edges:
            affected.update(edge_to_cycles[e])
        for ci in affected:
            e1, e2, e3, e4 = cycles[ci]
            if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
                return False
        return True
    
    random.seed(42)
    
    best_colors = None
    best_count = 0
    
    # Multi-start: try different initial colorings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Create a random valid starting coloring
        # Use transitive coloring with random vertex permutation
        perm = list(range(N))
        random.shuffle(perm)
        
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_idx(i, j)
                colors[idx] = perm[i]
        
        if not check_validity(colors, cycles):
            # Fallback
            for i in range(N):
                for j in range(i + 1, N):
                    idx = edge_idx(i, j)
                    colors[idx] = i
        
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        # Simulated annealing local search
        T = 3.0
        T_min = 0.05
        cooling = 0.9998
        
        steps_this_trial = 0
        max_steps = 30000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Change 1-2 random edges to random colors
            num_changes = 1 if random.random() < 0.6 else 2
            edges_to_change = random.sample(range(NUM_EDGES), num_changes)
            old_values = [(e, colors[e]) for e in edges_to_change]
            
            for e in edges_to_change:
                colors[e] = random.randint(0, 119)
            
            if is_valid_incremental(colors, edges_to_change):
                new_count = count_colors(colors)
                delta = cur_count - new_count
                # Accept if better, or with probability exp(-delta/T) if worse
                if new_count >= cur_count or (delta > 0 and random.random() < 2.718281828 ** (-delta / T)):
                    cur_count = new_count
                    if cur_count > best_count:
                        best_count = cur_count
                        best_colors = colors[:]
                    steps_this_trial += 1
                else:
                    for e, old_val in old_values:
                        colors[e] = old_val
            else:
                for e, old_val in old_values:
                    colors[e] = old_val
            
            # Early stop if no improvement for a while
            if steps_this_trial > 10000 and cur_count <= best_count:
                break
    
    # Final check
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_idx(i, j)
                best_colors[idx] = i
    
    if not check_validity(best_colors, cycles):
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_idx(i, j)
                best_colors[idx] = i
    
    return best_colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
    # Quick validity check
    cycles = build_cycle_data()
    print(f"Valid: {check_validity(result, cycles)}")
