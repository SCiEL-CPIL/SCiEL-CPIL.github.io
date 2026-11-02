import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    # 1. Generate D11 Minimal Vectors
    # Type 1: Weight 1, +/- 2
    vecs_t1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs_t1.append(v)
            
    # Type 2: Weight 4, +/- 1, even sum
    vecs_t2 = []
    for pos in itertools.combinations(range(D), 4):
        for mask in range(16):
            signs = [1 if (mask & (1<<j)) == 0 else -1 for j in range(4)]
            v = [0]*D
            for p, val in zip(pos, signs):
                v[p] = val
            vecs_t2.append(v)
                
    V_t1 = np.array(vecs_t1, dtype=np.int64)
    V_t2 = np.array(vecs_t2, dtype=np.int64)
    N = len(V_t2)
    
    # 2. Compute Conflict Matrix for Type 2
    # Conflict if dot product > 2
    dots = V_t2 @ V_t2.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists for faster access in local search
    # neighbors[i] is a list of indices j such that conflicts[i,j] is True
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    
    best_set = np.zeros(N, dtype=bool)
    best_count = 0
    
    n_restarts = 200
    
    for r in range(n_restarts):
        if time.time() - start_time > 100:
            break
            
        # Greedy phase
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            
            nbrs = neighbors[chosen]
            to_remove = np.union1d([chosen], nbrs)
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Update degrees
                # degrees[rem] -= number of neighbors in to_remove
                # conflicts[rem][:, to_remove].sum(axis=1) is efficient enough
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        count = np.sum(in_set)
        
        # Local Search: 1-for-2 swap
        # Try to remove a vertex u from in_set and add two vertices v, w
        # that were only blocked by u.
        improved = True
        while improved:
            improved = False
            in_set_indices = np.where(in_set)[0]
            
            # Shuffle to avoid bias
            np.random.shuffle(in_set_indices)
            
            for u in in_set_indices:
                if time.time() - start_time > 110:
                    break
                
                # Find candidates outside set that conflict ONLY with u (among set vertices)
                # A vertex x is a candidate if:
                # 1. x is not in in_set
                # 2. conflicts[x, in_set] is a subset of {u}
                
                # Efficient check:
                # Get all neighbors of u
                nbrs_u = neighbors[u]
                
                # Candidates must be in nbrs_u (if they don't conflict with u, they could be added without removing u, 
                # but we are removing u to free up space, so we look for those blocked by u)
                # Actually, if x does not conflict with u, it might be blocked by another vertex.
                # We want vertices that are blocked by u AND no other vertex in set.
                
                # Let's check all vertices not in set? Too slow.
                # Let's check neighbors of u.
                # If x is neighbor of u, and x is not in set, and x has no other neighbors in set (except u), 
                # then x becomes available if u is removed.
                
                # candidates = [x for x in nbrs_u if not in_set[x] and not np.any(conflicts[x, in_set] & ~in_set[u])]
                # This is still slow.
                
                # Optimization:
                # Count how many neighbors each vertex has in the current set.
                # neighbor_counts = conflicts @ in_set
                # A vertex x is "blocked only by u" if:
                # 1. not in_set[x]
                # 2. neighbor_counts[x] == 1
                # 3. conflicts[x, u] is True
                
                # We can maintain neighbor_counts dynamically?
                # For now, compute it for the current set.
                # This is O(N^2) per swap attempt.
                # With N=5000, 25M ops.
                # 100 swaps * 25M = 2.5B ops. Might be too slow in Python.
                
                # Alternative:
                # Only check if we can add 2 from the neighbors of u.
                
                # Let's try a simpler local search:
                # Try to add any vertex that is not in set and has 0 neighbors in set.
                # (This finds non-maximal sets, but our greedy produces maximal sets, so this will find nothing)
                
                # So we must do swaps.
                
                # Let's limit the scope:
                # Check if there are at least 2 vertices in nbrs_u that are not in set and have no other neighbors in set.
                
                # To make this fast, we can precompute "neighborhood intersection" logic?
                # Or just accept the cost and reduce restarts.
                
                # Let's try a cheaper check:
                # Randomly pick 2 candidates from nbrs_u that are not in set.
                # Check if they are compatible with the rest of the set (minus u).
                
                # This is heuristic.
                
                # Given time constraints, I will stick to the greedy but with more restarts and randomization, 
                # and a simple 1-for-2 swap using a limited check.
                
                # Check candidates in nbrs_u
                candidates = []
                for x in nbrs_u:
                    if in_set[x]:
                        continue
                    # Check if x has any other neighbor in set
                    # x conflicts with set iff np.any(conflicts[x] & in_set)
                    # We know x conflicts with u.
                    # Check if x conflicts with any other in_set.
                    # conflicts[x, in_set] has 1s.
                    # If sum(conflicts[x, in_set]) == 1, then only u is blocking it.
                    if np.sum(conflicts[x, in_set]) == 1:
                        candidates.append(x)
                        
                if len(candidates) >= 2:
                    # Try to find a pair in candidates that are not conflicting with each other
                    # and are not conflicting with each other's other neighbors?
                    # If v, w are candidates, they only conflict with u in the current set.
                    # If we remove u, they only conflict with each other (if they do).
                    # So we just need v and w to not conflict with each other.
                    
                    found_pair = False
                    for i in range(len(candidates)):
                        if found_pair: break
                        for j in range(i+1, len(candidates)):
                            if not conflicts[candidates[i], candidates[j]]:
                                # Swap!
                                in_set[u] = False
                                in_set[candidates[i]] = True
                                in_set[candidates[j]] = True
                                count += 1
                                improved = True
                                found_pair = True
                                break
                                
            if time.time() - start_time > 110:
                break

        if count > best_count:
            best_count = count
            best_set = in_set.copy()
            
    # Combine results
    V_t2_sel = V_t2[best_set]
    result = np.vstack([V_t1, V_t2_sel])
    
    return result
