import numpy as np
import itertools
import time
from scipy import sparse

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    D = 11
    
    # 1. Generate W8 vectors (weight 8, entries +/-1, norm^2=8)
    # Positions: combinations of 8 from 11
    # Signs: 2^8
    v8_candidates = []
    for positions in itertools.combinations(range(D), 8):
        for signs in itertools.product([1, -1], repeat=8):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v8_candidates.append(v)
            
    vectors_v8 = np.array(v8_candidates, dtype=np.int8)
    N8 = len(vectors_v8)
    
    # 2. Generate W2 vectors (weight 2, entries +/-2, norm^2=8)
    v2_candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    v2_candidates.append(v)
                    
    vectors_v2 = np.array(v2_candidates, dtype=np.int8)
    N2 = len(vectors_v2)
    
    # We established W2 never conflicts with W8 (max dot 4, threshold 5)
    # So we can add all W2 vectors to the result.
    
    # 3. Find Max Independent Set in W8 graph
    # Conflict if dot >= 5
    
    # Compute dot products in blocks to save memory
    # We need a sparse conflict matrix or neighbor lists
    neighbors = []
    
    # Block size for dot product computation
    block_size = 1000
    
    # Initialize neighbor lists as lists of lists
    # N8 is 42240. List of lists is fine.
    adj_list = [[] for _ in range(N8)]
    
    # To speed up, we can compute dots for block i against all j > i
    # But we need symmetric.
    
    # Alternative: Compute full dot matrix in blocks and extract edges
    # Since N8 is 42k, N8^2 is 1.8e9. Too big for dense float.
    # But we can do it in chunks.
    
    # Actually, let's just use the sparse matrix construction
    # row_indices, col_indices, data
    
    rows = []
    cols = []
    
    for i_start in range(0, N8, block_size):
        i_end = min(i_start + block_size, N8)
        block = vectors_v8[i_start:i_end]
        
        # Compute dot products of block against all vectors
        # block (B x D) @ V8.T (D x N8) -> B x N8
        dots = block @ vectors_v8.T
        
        # We only care about j > i_start to avoid duplicates and self
        # Actually, we can just mark conflicts where dot >= 5
        # And i != j
        
        # For each row in this block
        for k in range(i_end - i_start):
            i = i_start + k
            # We only need j > i to build undirected graph
            # But dots matrix is B x N8.
            # We can slice it
            
            # Mask for j > i
            # We can just iterate or use argwhere
            # For speed, use numpy
            
            # Get indices where dot >= 5
            # Note: dot[i,i] is 8, so it will be included. We skip i==j.
            
            confs = np.where(dots[k] >= 5)[0]
            # Filter out i itself
            confs = confs[confs != i]
            # We only want j > i to add edge once?
            # If we add all, we double count.
            # Let's just add all, and deduplicate later or just use a set?
            # No, adj_list is better.
            
            # To avoid adding edges to j < i (which were already added when j was in a previous block),
            # we can restrict to j > i.
            confs = confs[confs > i]
            
            if len(confs) > 0:
                adj_list[i].extend(confs)
                # And add reverse edges?
                # No, if we only add j > i, we need to ensure we can traverse both ways.
                # So we must add i to adj_list[j] as well.
                for j in confs:
                    adj_list[j].append(i)
                    
    # 4. Greedy + SA on this graph
    # Degrees
    degrees = np.array([len(n) for n in adj_list], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    def greedy_min_degree():
        in_set = np.zeros(N8, dtype=bool)
        alive = np.ones(N8, dtype=bool)
        degs = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            min_deg = np.min(degs[alive_indices])
            cands = alive_indices[degs[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = adj_list[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Update degrees
                # degs[rem] -= count of neighbors in to_remove
                # This is slow if done naively.
                # For now, skip degree update for speed in greedy, or do it roughly.
                # Actually, just recompute or use a simpler heuristic.
                pass
                
        return in_set

    # SA
    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Conflict count for each node: number of neighbors in in_set
        cc = np.zeros(N8, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[adj_list[i]] += 1
            
        T = 2.5
        cooling = 0.9997
        min_T = 0.03
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        stuck = 0
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0: break
            
            # Pick v with min conflicts
            counts = cc[not_in_set]
            min_count = np.min(counts)
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]
            k = int(min_count)
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.4
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                to_remove = np.where(np.array([cc[j] > 0 for j in adj_list[v]]) & in_set)[0] 
                # Wait, to_remove should be neighbors of v that are in in_set.
                # adj_list[v] are neighbors.
                # We need to remove those that are in in_set.
                nbrs_in_set = [j for j in adj_list[v] if in_set[j]]
                if len(nbrs_in_set) > 0:
                    in_set[nbrs_in_set] = False
                    for u in nbrs_in_set:
                        cc[adj_list[u]] -= 1
                in_set[v] = True
                current_size += delta
                cc[adj_list[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck = 0
                else:
                    stuck += 1
            else:
                stuck += 1
                
            if stuck > 2000:
                T = max(T, 1.5)
                stuck = 0
            T *= cooling
            if T < min_T: T = min_T
            
        return local_best_in_set, local_best_size

    # Run SA
    best_in_set = np.zeros(N8, dtype=bool)
    best_size = 0
    
    # Initial greedy
    in_set = greedy_min_degree()
    if time.time() < deadline - 20:
        in_set, cur_size = sa_local_search(in_set, max_time=30)
        best_in_set = in_set
        best_size = cur_size
        
    # Result
    result_v8 = vectors_v8[best_in_set]
    final_vectors = np.vstack([result_v8, vectors_v2])
    
    return final_vectors
