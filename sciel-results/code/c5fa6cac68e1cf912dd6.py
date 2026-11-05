import time
import random
import numpy as np
import itertools

# Deadline guard: 150s limit, aim to finish in 140s
_DEADLINE = 140.0

def _generate_partitions(n):
    """
    Precompute all (subset, partition) pairs and their edge indices.
    Returns a numpy array of shape (num_subsets * 15, 12) containing edge indices.
    """
    subsets = list(itertools.combinations(range(n), 6))
    num_subsets = len(subsets)
    
    # The 15 ways to partition 6 items {0,1,2,3,4,5} into 3 pairs
    # We generate these once
    matchings = []
    for a in range(1, 6):
        R = [x for x in range(1, 6) if x != a]
        R.sort()
        s = R[0]
        for c in R[1:]:
            R2 = [x for x in R if x != s and x != c]
            d1, d2 = R2
            matchings.append(((0, a), (s, c), (d1, d2)))
    
    # Precompute the 12 cross-edge position indices for each matching
    # Positions are 0..5. Edges are between different pairs.
    cross_pos_indices = []
    for m in matchings:
        pairs = list(m)
        cross = []
        for x in range(3):
            for y in range(x + 1, 3):
                for u in pairs[x]:
                    for v in pairs[y]:
                        cross.append((u, v))
        # cross is a list of 12 (u,v) tuples
        cross_pos_indices.append(cross)
        
    # Now build the global edge index array
    # Total entries = num_subsets * 15
    # Each entry is 12 edge indices
    # Edge index for (i,j) is i*n + j (assuming i<j)
    
    all_edges = []
    
    # Optimization: Process in batches or just loop. 
    # 593k * 15 = ~9M entries. 
    # Creating a list of 9M lists of 12 ints is memory heavy and slow in Python.
    # Better to use a large numpy array.
    
    total_entries = num_subsets * 15
    # We will fill a numpy array of shape (total_entries, 12)
    # But building it in a Python loop is slow.
    
    # Alternative: 
    # 1. Get all subset vertex arrays: shape (num_subsets, 6)
    # 2. For each of the 15 matchings, extract the 12 edges.
    
    subsets_arr = np.array(subsets, dtype=np.int32) # (N, 6)
    
    edge_indices_list = []
    
    for match in cross_pos_indices:
        # match is list of 12 (u,v)
        # For each subset, get the vertices
        # v_u = subsets_arr[:, u]
        # v_v = subsets_arr[:, v]
        # edge_idx = min(v_u, v_v)*n + max(v_u, v_v)
        
        # Vectorized operation for all subsets for this matching
        # We need to create an array of shape (N, 12)
        edges_for_match = np.zeros((num_subsets, 12), dtype=np.int32)
        
        for k, (u, v) in enumerate(match):
            vu = subsets_arr[:, u]
            vv = subsets_arr[:, v]
            
            # Ensure u < v for index calculation? 
            # The graph is undirected. Edge (i,j) is same as (j,i).
            # We store colors in a matrix C where C[i,j] = C[j,i].
            # So we can just use i*n+j.
            # But to be safe and consistent with "lexicographic order" if we ever flatten,
            # let's just use the matrix index.
            
            idx = vu * n + vv
            edges_for_match[:, k] = idx
            
        edge_indices_list.append(edges_for_match)
        
    # Concatenate all 15 matchings along axis 0? 
    # No, we want a flat list of (subset, partition) -> 12 edges.
    # Shape: (num_subsets * 15, 12)
    
    # Reshape edge_indices_list (15, N, 12) to (15*N, 12)
    # But we need to interleave or just stack? 
    # It doesn't matter for the penalty calculation as long as we process them all.
    
    all_edges = np.concatenate([e.reshape(-1, 12) for e in edge_indices_list], axis=0)
    
    return all_edges

def _paley29():
    """
    Construct the known valid Paley(29) coloring.
    """
    n = 29
    # Quadratic residues modulo 29
    squares = set()
    for x in range(1, 29):
        squares.add((x * x) % 29)
        
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}

def _validate(n, colors):
    """
    Fast validation using the precomputed edge indices.
    colors: flat list or numpy array of length n*n (symmetric)
    """
    # Convert to numpy array for fast access
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype=np.int8)
        
    # all_edges is a global precomputed array for n=30? 
    # No, we need to pass the edge indices appropriate for n.
    # The _generate_partitions depends on n.
    # So we should generate them inside the search function or pass them.
    
    # For this helper, let's assume we have the edges.
    pass

def construct():
    start_time = time.time()
    
    # 1. Baseline: Paley 29
    baseline = _paley29()
    best_result = baseline
    best_n = 29
    
    # 2. Try to find n=30
    target_n = 30
    
    # Precompute partitions for n=30
    # This might take a few seconds and some memory.
    # 593775 subsets * 15 * 12 * 4 bytes = ~42 MB. Acceptable.
    
    try:
        all_edges = _generate_partitions(target_n)
    except Exception as e:
        # If precomputation fails or is too slow, return baseline
        return baseline

    # Initialize coloring for n=30
    # Start with Paley 29 embedded in 30?
    # Or random?
    # Let's try starting from a random coloring and annealing.
    # Or start from Paley 29 + random connections to vertex 29.
    
    rng = random.Random(42)
    
    # Create a numpy array for the graph matrix
    # C[i, j] is the color of edge (i, j)
    # We only care about i < j, but we can fill the whole matrix for easy indexing.
    
    # Strategy: 
    # 1. Set C[0..28, 0..28] based on Paley 29.
    # 2. Set C[29, 0..28] and C[0..28, 29] randomly.
    
    C = np.zeros((target_n, target_n), dtype=np.int8)
    
    # Fill Paley 29 part
    p29 = baseline['colors']
    idx = 0
    for i in range(29):
        for j in range(i + 1, 29):
            c = p29[idx]
            C[i, j] = c
            C[j, i] = c
            idx += 1
            
    # Fill connections to vertex 29
    for i in range(29):
        c = rng.randint(0, 1)
        C[i, 29] = c
        C[29, i] = c
        
    # Local Search
    # Objective: Minimize number of monochromatic K2,2,2
    
    # all_edges shape: (N_total, 12)
    # N_total = 593775 * 15 = 8906625
    
    # To calculate penalty:
    # For each row in all_edges, get the 12 colors.
    # Check if all are 0 or all are 1.
    
    # Vectorized penalty calculation:
    # Get colors for all edges at once: 
    # rows = all_edges[:, 0]
    # cols = all_edges[:, 1] ...
    # This is hard to vectorize perfectly without creating huge intermediate arrays.
    
    # Alternative: 
    # C is (30,30). 
    # all_edges is (M, 12) where M is ~9M.
    # We can do:
    # vals = C[all_edges[:, 0] // target_n, all_edges[:, 0] % target_n] ... 
    # This creates an array of shape (M, 12).
    # M is 9M. 9M * 12 * 1 byte = 108 MB. 
    # This is feasible in memory and fast in NumPy.
    
    def get_penalty(C):
        # Extract all edge colors
        # all_edges contains indices i*n+j.
        # We need to convert these to (row, col) or just use them if C was flat.
        # Let's keep C as flat matrix (n*n) for easier indexing with precomputed indices.
        # C_flat shape (n*n)
        # all_edges indices are already i*n+j.
        
        # Get colors
        # vals shape (M, 12)
        vals = C_flat[all_edges]
        
        # Check if all 0
        all_zero = np.all(vals == 0, axis=1)
        # Check if all 1
        all_one = np.all(vals == 1, axis=1)
        
        # Penalty is count of rows where all_zero OR all_one
        penalty = np.sum(all_zero | all_one)
        return penalty

    C_flat = C.flatten()
    
    # Initial penalty
    current_penalty = get_penalty(C_flat)
    best_penalty = current_penalty
    best_C = C_flat.copy()
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Pairs to flip
    # We can just pick a random index in C_flat where i < j.
    # Or precompute list of valid edge indices.
    valid_edges = [i * target_n + j for i in range(target_n) for j in range(i + 1, target_n)]
    
    steps = 0
    while time.time() - start_time < _DEADLINE - 5:
        # Pick a random edge
        edge_idx = rng.choice(valid_edges)
        
        # Flip it
        C_flat[edge_idx] = 1 - C_flat[edge_idx]
        # Symmetry: 
        # i = edge_idx // target_n
        # j = edge_idx % target_n
        # sym_idx = j * target_n + i
        # C_flat[sym_idx] = C_flat[edge_idx]
        
        # Calculate new penalty
        new_penalty = get_penalty(C_flat)
        
        delta = new_penalty - current_penalty
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_penalty = new_penalty
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_C = C_flat.copy()
                
                if best_penalty == 0:
                    # Found valid coloring!
                    # Extract colors in lexicographic order
                    colors_out = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            colors_out.append(int(C_flat[i * target_n + j]))
                    return {'n': target_n, 'colors': colors_out}
        else:
            # Revert
            C_flat[edge_idx] = 1 - C_flat[edge_idx]
            
        T = max(min_T, T * cooling_rate)
        steps += 1
        
        # If we are stuck for a long time, maybe restart?
        # For now, just continue.
        
    # If we didn't find 0 penalty, return the best found? 
    # No, we must return a VALID coloring. 
    # If best_penalty > 0, the coloring is invalid. 
    # We should return the baseline (29) in that case.
    
    return best_result

if __name__ == "__main__":
    construct()
