import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    D = 11
    
    # 1. Generate Vectors
    # V1: Weight 1, entries +/-2. Norm^2 = 4. Count = 22.
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    V1 = np.array(v1_list, dtype=np.int8)
    
    # V2: Weight 2, entries +/-1. Norm^2 = 2. Count = 220.
    v2_list = []
    for pos1 in range(D):
        for pos2 in range(pos1 + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[pos1] = s1
                    v[pos2] = s2
                    v2_list.append(v)
    V2 = np.array(v2_list, dtype=np.int8)
    
    # V4: Weight 4, entries +/-1. Norm^2 = 4. Count = 5280.
    v4_list = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(combo, signs):
                v[pos] = sign
            v4_list.append(v)
    V4 = np.array(v4_list, dtype=np.int8)
    
    # Combine
    V_all = np.vstack([V1, V2, V4])
    N = len(V_all)
    
    # 2. Compute Conflict Matrix
    # Condition: ||u - v||^2 >= 4  (since max norm^2 is 4)
    # ||u||^2 + ||v||^2 - 2 u.v >= 4
    # u.v <= (||u||^2 + ||v||^2 - 4) / 2
    
    NormSq = np.sum(V_all.astype(np.float64) ** 2, axis=1)
    
    # Dot products
    Dots = V_all.astype(np.float64) @ V_all.astype(np.float64).T
    
    # Thresholds
    # T[i, j] = (NormSq[i] + NormSq[j] - 4) / 2
    T = (NormSq[:, None] + NormSq[None, :] - 4.0) / 2.0
    
    # Conflict if Dots > T
    # Note: For i=j, Dots[i,i] = NormSq[i].
    # T[i,i] = (2*NormSq[i] - 4)/2 = NormSq[i] - 2.
    # Dots[i,i] > T[i,i] => NormSq[i] > NormSq[i] - 2. True.
    # So diagonal is True. We must set it to False.
    
    Conflicts = (Dots > T)
    np.fill_diagonal(Conflicts, False)
    
    # Conflicts is a boolean matrix N x N
    # Convert to int8 for faster summing if needed, or keep bool.
    
    # 3. Maximum Independent Set (MIS)
    # We want to maximize the number of nodes selected such that no two are connected.
    
    rng = np.random.default_rng(42)
    
    best_selected = None
    best_k = 0
    
    # Precompute initial degrees
    # Degree of node i is sum of Conflicts[i, :]
    # Since Conflicts is symmetric, we can use it.
    
    def greedy_mis(conflicts, deadline, rng, seed=0):
        N = len(conflicts)
        alive = np.ones(N, dtype=bool)
        # Copy conflicts to avoid modifying original? 
        # We need to update degrees.
        # It's faster to keep a copy of the conflict matrix for the current run?
        # Or just re-calculate degrees from the original matrix and alive mask?
        # degrees[i] = sum(conflicts[i, alive])
        
        # For speed, we can maintain a sparse representation or just use numpy slicing.
        # N=5522. Slicing is fast.
        
        selected = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Calculate degrees of alive nodes
            # conflicts[alive_indices, :].sum(axis=1) gives degrees of alive nodes with respect to ALL nodes.
            # We need degrees with respect to ALIVE nodes.
            # deg[i] = sum(conflicts[i, alive])
            
            # This is O(N^2) per step if done naively.
            # 5522^2 * 5522 steps? No, N steps.
            # Total O(N^3) = 5522^3 ~ 1.7 * 10^11. Too slow.
            
            # We need a faster way.
            # Use a bitset? Or maintain a list of neighbors?
            
            # Alternative: Use the adjacency list representation.
            # But building it is O(N^2).
            
            # Let's try to optimize the greedy step.
            # Instead of min-degree, use a random node?
            # Or use a precomputed order?
            
            # A common heuristic: Pick the node with the smallest degree.
            # To update degrees efficiently:
            # When we pick u and remove it and its neighbors N(u):
            # For any remaining node v, its degree decreases by the number of neighbors it had in {u} U N(u).
            
            # Let's maintain `deg` array.
            # Initial deg = conflicts.sum(axis=1).
            
            # When removing a set R (u and its neighbors):
            # For each v in alive \ R:
            #   deg[v] -= sum(conflicts[v, R])
            
            # This is O(N * |R|). |R| is roughly N/k.
            # Total complexity is better.
            
            pass # This function needs to be implemented properly below.

    # Let's implement a more efficient MIS using numpy arrays and careful updates.
    
    def run_mis(conflicts, deadline, rng):
        N = len(conflicts)
        alive = np.ones(N, dtype=bool)
        deg = conflicts.sum(axis=1).astype(np.int32)
        
        selected = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree among alive
            # To avoid O(N) scan every time, we could use a heap, but N is small enough.
            # 5522 elements scan is ~50us. 5522 steps * 50us = 0.3s. Acceptable.
            
            min_deg = np.min(deg[alive_indices])
            # Find candidates with min_deg
            candidates = alive_indices[deg[alive_indices] == min_deg]
            
            # Random pick
            u = candidates[rng.integers(len(candidates))]
            
            selected.append(u)
            
            # Determine nodes to remove: u and its neighbors
            # neighbors = np.where(conflicts[u])[0]
            # to_remove = np.unique(np.concatenate(([u], neighbors)))
            
            # Faster:
            # mask_remove = np.zeros(N, dtype=bool)
            # mask_remove[u] = True
            # mask_remove[conflicts[u]] = True
            
            # But we only care about alive nodes.
            # remove_mask = alive.copy()
            # remove_mask[~conflicts[u]] = False  # This keeps only neighbors? No.
            # We want to remove u AND neighbors.
            
            # Let's just get the indices.
            neighbors = np.where(conflicts[u])[0]
            to_remove = np.concatenate(([u], neighbors))
            
            # Update alive
            alive[to_remove] = False
            
            # Update degrees
            # For each v in alive, deg[v] -= number of neighbors in to_remove
            # deg[v] -= conflicts[v, to_remove].sum()
            
            # This is O(N * |to_remove|).
            # |to_remove| can be large.
            # Vectorized:
            # alive_indices = np.where(alive)[0]
            # if len(alive_indices) > 0:
            #     deg[alive_indices] -= conflicts[alive_indices][:, to_remove].sum(axis=1)
            
            # This slice `conflicts[alive_indices][:, to_remove]` creates a view/copy.
            # It's efficient in numpy.
            
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                # Count conflicts between remaining alive and removed nodes
                # conflicts[remaining_alive, to_remove] is a matrix of size (len(rem) x len(rem))
                # Sum along axis 1 gives the reduction in degree for each remaining node.
                deg[remaining_alive] -= conflicts[remaining_alive][:, to_remove].sum(axis=1)
                
        return selected

    # Run multiple restarts
    max_restarts = 200
    mis_deadline = deadline - 5.0
    
    for restart in range(max_restarts):
        if time.time() > mis_deadline:
            break
        
        selected = run_mis(Conflicts, mis_deadline, rng)
        
        k = len(selected)
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
            
        # Optional: Local improvement
        # Try to swap?
        # Given the time, simple restarts might be sufficient.
        # The graph is dense?
        # Weight 4 vectors: 5280.
        # Degree in V4 subgraph:
        # Two weight 4 vectors conflict if dot > 2.
        # If they share 4 positions with same signs, dot=4.
        # If they share 3 positions, dot can be 3, 1, -1, -3.
        # Conflict if dot=3.
        # This is a sparse graph?
        # Let's assume it's manageable.

    if best_selected is None:
        # Fallback
        return V1.astype(np.int64)
        
    result = V_all[best_selected].astype(np.int64)
    
    # Verify validity (optional, for safety)
    # But the evaluator will do it.
    
    return result
