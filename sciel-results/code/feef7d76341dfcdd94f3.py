import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
                
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1

    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    adj = [np.where(conflicts[i])[0] for i in range(N4)]
    
    rng = np.random.default_rng(42)
    
    def greedy_init():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degs = np.array([len(a) for a in adj], dtype=np.int32)
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            min_deg = np.min(degs[alive_idx])
            cands = alive_idx[degs[alive_idx] == min_deg]
            v = cands[rng.integers(len(cands))]
            
            in_set[v] = True
            nbrs = adj[v]
            to_remove = np.unique(np.concatenate(([v], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degs[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def tabu_search(init_set, max_time=20.0):
        in_set = init_set.copy()
        best_set = in_set.copy()
        best_size = in_set.sum()
        
        # Tabu List: node -> iteration when it becomes non-tabu
        tabu_until = np.zeros(N4, dtype=np.int64)
        tabu_len = 50
        
        # Conflict count
        conflict_count = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[adj[i]] += 1
            
        ts_deadline = time.time() + max_time
        iter_count = 0
        stuck = 0
        
        while time.time() < ts_deadline:
            iter_count += 1
            
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0: break
            
            counts = conflict_count[not_in]
            
            # Find best move
            # We want to maximize delta = 1 - k
            # Prefer non-tabu moves.
            # If all best moves are tabu, pick best tabu move (aspiration) or next best non-tabu.
            
            # Sort candidates by count (ascending)
            # This is expensive if done every iter.
            # Just find min count.
            min_c = np.min(counts)
            
            # Candidates with min count
            cands = not_in[counts == min_c]
            
            # Filter non-tabu
            non_tabu_cands = cands[tabu_until[cands] < iter_count]
            
            if len(non_tabu_cands) > 0:
                v = non_tabu_cands[rng.integers(len(non_tabu_cands))]
            else:
                # All min-count cands are tabu.
                # Check if any improve global best?
                # For now, pick a random tabu one (aspiration) or relax tabu.
                # Let's pick random from cands.
                v = cands[rng.integers(len(cands))]
                
            k = conflict_count[v]
            delta = 1 - k
            
            # Perform move
            nbrs_in_set = np.where(in_set[adj[v]])[0]
            
            # Update in_set
            if len(nbrs_in_set) > 0:
                in_set[nbrs_in_set] = False
                for u in nbrs_in_set:
                    conflict_count[adj[u]] -= 1
                    tabu_until[u] = iter_count + tabu_len
                    
            in_set[v] = True
            conflict_count[adj[v]] += 1
            tabu_until[v] = 0 # v is in set, so not tabu for addition?
            # Tabu is usually on the last moved node.
            # We marked removed nodes as tabu.
            
            current_size = in_set.sum()
            if current_size > best_size:
                best_size = current_size
                best_set = in_set.copy()
                stuck = 0
            else:
                stuck += 1
                
            if stuck > 2000:
                # Restart or diversify?
                # Simple restart: break and start new SA/Tabu
                break
                
        return best_set, best_size

    best_size = 0
    best_set = np.zeros(N4, dtype=bool)
    
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        init = greedy_init()
        size = init.sum()
        
        if size > best_size:
            best_size = size
            best_set = init.copy()
            
        remaining = deadline - 10 - time.time()
        if remaining < 2: break
        
        ts_time = min(remaining * 0.6, 15.0)
        ts_set, ts_size = tabu_search(init, max_time=ts_time)
        
        if ts_size > best_size:
            best_size = ts_size
            best_set = ts_set.copy()
            
        # Polish with SA if time remains
        remaining = deadline - 5 - time.time()
        if remaining > 3:
            # Simple SA polish
            # Re-use SA logic from before?
            # Let's just do another Tabu pass.
            ts_time2 = min(remaining * 0.5, 10.0)
            ts_set2, ts_size2 = tabu_search(ts_set, max_time=ts_time2)
            if ts_size2 > best_size:
                best_size = ts_size2
                best_set = ts_set2.copy()

    result_v4 = vectors_v4[best_set]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
