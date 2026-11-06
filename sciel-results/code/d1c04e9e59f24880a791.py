import time
import sys
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Precompute edge indices
    # Edge (i, j) with i < j.
    # Index = sum_{k=0}^{i-1} (N-1-k) + (j - i - 1)
    # sum_{k=0}^{i-1} (N-1-k) = i(N-1) - i(i-1)/2
    edge_indices = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_indices[(i, j)] = idx
            edge_indices[(j, i)] = idx
            idx += 1
            
    # Precompute all 4-cycles (sets of 4 edges)
    # A 4-cycle on vertices {a, b, c, d} can be formed in 3 ways (perfect matchings of K4):
    # 1. (a,b), (b,c), (c,d), (d,a) -> Edges: (a,b), (b,c), (c,d), (a,d)
    # 2. (a,b), (b,d), (d,c), (c,a) -> Edges: (a,b), (b,d), (c,d), (a,c)
    # 3. (a,c), (c,b), (b,d), (d,a) -> Edges: (a,c), (b,c), (b,d), (a,d)
    
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = edge_indices[(a, b)]
                    e2 = edge_indices[(b, c)]
                    e3 = edge_indices[(c, d)]
                    e4 = edge_indices[(d, a)]
                    cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 2: a-b-d-c-a
                    e1 = edge_indices[(a, b)]
                    e2 = edge_indices[(b, d)]
                    e3 = edge_indices[(c, d)]
                    e4 = edge_indices[(a, c)]
                    cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 3: a-c-b-d-a
                    e1 = edge_indices[(a, c)]
                    e2 = edge_indices[(b, c)]
                    e3 = edge_indices[(b, d)]
                    e4 = edge_indices[(a, d)]
                    cycles.append((e1, e2, e3, e4))
    
    # Convert cycles to a numpy array for vectorization
    # Shape: (num_cycles, 4)
    cycles_arr = np.array(cycles, dtype=np.int32)
    
    # Initialize with a valid coloring (e.g., all 0s, or the recursive bisection one)
    # Let's use the recursive bisection as a strong start, but we need to be careful with time.
    # Actually, starting from all 0s is safe but slow to improve.
    # Let's use the recursive bisection logic to generate an initial valid coloring.
    
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    color_id = 0
    def process_partition(vertices):
        nonlocal color_id
        if len(vertices) <= 1:
            return
        
        mid = len(vertices) // 2
        left = vertices[:mid]
        right = vertices[mid:]
        
        for u in left:
            for v in right:
                idx = get_edge_idx(u, v)
                colors[idx] = color_id
        
        color_id += 1
        
        if time.time() > deadline:
            return

        process_partition(left)
        process_partition(right)

    process_partition(list(range(N)))
    
    current_colors = colors.copy()
    current_distinct = len(set(current_colors.tolist()))
    best_colors = current_colors.copy()
    best_distinct = current_distinct
    
    # Local search
    # We want to maximize distinct colors.
    # Strategy: Pick a random edge, try to assign it a new color (or a color not currently in its conflicting cycles).
    # Actually, simpler: Pick a random edge, assign it a random color from 0..119.
    # Check if valid. If valid, accept. If it increases distinct count, definitely accept.
    # If invalid, reject.
    # To be smarter: Only try colors that are "promising".
    # But with 120 colors, random is okay if we do enough iterations.
    
    # Let's use a temperature-based acceptance (Simulated Annealing)
    # Objective: Maximize distinct colors.
    # Penalty for invalidity: Infinite (must always be valid).
    
    # So we only accept moves that result in a valid coloring.
    # Among valid colorings, we prefer higher distinct count.
    # If distinct count is same, we might accept to escape local optima?
    # But "distinct count" is a step function.
    # Let's just accept if distinct count increases.
    # If distinct count is same, accept with probability exp(-1/T) to explore?
    # Or just random walk among valid colorings with same distinct count?
    
    # Let's try:
    # 1. Pick random edge e.
    # 2. Pick random new color c.
    # 3. If c == current_color[e], skip.
    # 4. Apply change.
    # 5. Check validity.
    #    - If invalid, revert.
    #    - If valid:
    #       - new_distinct = len(set(new_colors))
    #       - if new_distinct > current_distinct: accept, update best.
    #       - if new_distinct == current_distinct: accept with prob 0.5 (random walk).
    #       - if new_distinct < current_distinct: reject (revert).
    
    # Optimization:
    # Checking validity by scanning all cycles is O(num_cycles) = O(1820).
    # We can optimize by only checking cycles involving the changed edge.
    # Build an adjacency list: for each edge, list of cycle indices it belongs to.
    
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(i)
    
    # Convert to numpy arrays for faster access?
    # Or just use lists of indices.
    
    # Let's use a vectorized check for the specific cycles involving the edge.
    
    t = 1.0
    cooling_rate = 0.9999
    
    # Count iterations
    iters = 0
    
    while time.time() < deadline:
        # Pick random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Pick new color
        # Bias towards new colors?
        # Let's just pick random from 0..119
        new_color = random.randint(0, MAX_COLORS - 1)
        
        if new_color == old_color:
            continue
            
        # Apply change
        current_colors[e] = new_color
        
        # Check validity for cycles involving e
        valid = True
        cycle_indices = edge_to_cycles[e]
        if cycle_indices:
            # Get the sub-array of cycles
            sub_cycles = cycles_arr[cycle_indices]
            # Get colors of edges in these cycles
            # sub_cycles shape: (k, 4)
            # colors shape: (120,)
            cycle_colors = current_colors[sub_cycles]
            # Check if any row has 4 distinct values
            # Sort each row
            sorted_colors = np.sort(cycle_colors, axis=1)
            # Check if all 4 are distinct: sorted[0]!=sorted[1], sorted[1]!=sorted[2], sorted[2]!=sorted[3]
            # Or simply: len(set(row)) == 4
            # Vectorized:
            distinct_mask = (sorted_colors[:, 0] != sorted_colors[:, 1]) & \
                            (sorted_colors[:, 1] != sorted_colors[:, 2]) & \
                            (sorted_colors[:, 2] != sorted_colors[:, 3])
            if np.any(distinct_mask):
                valid = False
        
        if not valid:
            # Revert
            current_colors[e] = old_color
        else:
            # Valid
            new_distinct = len(set(current_colors.tolist()))
            old_distinct = len(set(np.append(current_colors[:e], np.append(current_colors[e+1:], [old_color])).tolist()))
            # This is expensive. Let's maintain a count of color frequencies.
            
            # Actually, let's just calculate distinct count. It's O(120).
            # 120 is small.
            
            if new_distinct > current_distinct:
                current_distinct = new_distinct
                if new_distinct > best_distinct:
                    best_distinct = new_distinct
                    best_colors = current_colors.copy()
                # Accept
            elif new_distinct == current_distinct:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    # Accept
                    pass
                else:
                    # Reject
                    current_colors[e] = old_color
            else:
                # Reject
                current_colors[e] = old_color
                
        iters += 1
        if iters % 1000 == 0:
            t *= cooling_rate
            
    return best_colors.tolist()

# Keep the function name and signature
# The task asks for construct() -> 120 integer colors
