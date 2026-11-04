import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Precompute all 4-element supports
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    # Convert supports to bitmasks for fast intersection
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Build conflict matrix: conflict[i,j] = True if supports i and j intersect in >= 3 indices
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # Vectorized greedy set-cover heuristic for Maximum Independent Set
    # At each step, select the support that maximizes marginal gain / penalty ratio
    # Marginal gain: number of newly compatible supports (not in set, not conflicting with current set)
    # Penalty: number of supports that would be invalidated (conflict with the new support)
    
    def greedy_mis(conflict, N, deadline, rng=None):
        """Vectorized greedy MIS using submodular ratio heuristic."""
        in_set = np.zeros(N, dtype=bool)
        blocked = np.zeros(N, dtype=bool)  # Supports conflicting with in_set
        
        # Precompute degree (number of conflicts) for tie-breaking
        degree = np.sum(conflict, axis=1)
        
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Candidates: not in set, not blocked
            candidates = np.where(~(in_set | blocked))[0]
            if len(candidates) == 0:
                break
            
            # For each candidate, compute:
            # - gain: 1 (adding one support)
            # - penalty: number of supports it conflicts with that are currently available (not in set, not blocked)
            # We want to maximize gain/penalty, or minimize penalty since gain is constant
            # Actually, we want to minimize the number of future supports we block
            # penalty = number of available supports that conflict with candidate
            
            # Vectorized: for each candidate v, count how many available supports conflict with v
            # available = ~in_set & ~blocked
            available = ~(in_set | blocked)
            
            # conflict[candidates, :].sum(axis=1) gives total conflicts, but we want conflicts with available only
            # Use matrix multiplication or sum over available columns
            # penalty[v] = sum of conflict[v, available]
            # This is equivalent to (conflict[candidates] * available).sum(axis=1)
            
            # Efficient computation using boolean matrix
            # conflict is (N, N) bool, available is (N,) bool
            # For each candidate row, sum over available columns
            candidate_conflicts = conflict[candidates]  # (len(candidates), N)
            penalties = (candidate_conflicts & available[np.newaxis, :]).sum(axis=1)
            
            # Also consider: among candidates with same penalty, prefer those with lower degree (less restrictive)
            # Or use a combined score: minimize penalty, break ties by degree
            
            # Score: primarily minimize penalty, secondarily minimize degree (for future flexibility)
            candidate_degrees = degree[candidates]
            
            # Combined score: penalty * 1000 + degree (minimize)
            scores = penalties * 1000 + candidate_degrees
            
            # Select the candidate with minimum score
            best_idx_in_candidates = np.argmin(scores)
            best_candidate = candidates[best_idx_in_candidates]
            
            # Add to set
            in_set[best_candidate] = True
            blocked |= conflict[best_candidate]
            
            # Check time
            if time.time() > deadline - 5.0:
                break
        
        return in_set
    
    # Run greedy with multiple random orderings to get diverse solutions
    best_support_set = set()
    best_size = 0
    
    rng = np.random.RandomState(42)
    
    # Multiple restarts with different random perturbations
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # For each restart, we can use a slightly different heuristic
        # Option 1: Pure greedy (minimize penalty)
        # Option 2: Greedy with random tie-breaking
        # Option 3: Greedy with temperature-based exploration
        
        x = greedy_mis(conflict, N_S, deadline)
        size = int(np.sum(x))
        
        if size > best_size:
            best_size = size
            best_support_set = set(np.where(x)[0].tolist())
        
        # Also try a local search polish on the result
        if size > best_size - 2 and time.time() < deadline - 15.0:
            # Local search: try to add any available support
            in_set = x.copy()
            blocked = np.zeros(N_S, dtype=bool)
            in_indices = np.where(in_set)[0]
            if len(in_indices) > 0:
                blocked = np.any(conflict[in_indices], axis=0)
            
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                candidates = np.where(~(in_set | blocked))[0]
                if len(candidates) > 0:
                    # Add the one with minimum penalty
                    available = ~(in_set | blocked)
                    candidate_conflicts = conflict[candidates]
                    penalties = (candidate_conflicts & available[np.newaxis, :]).sum(axis=1)
                    best_cand = candidates[np.argmin(penalties)]
                    in_set[best_cand] = True
                    blocked |= conflict[best_cand]
                    improved = True
            
            size2 = int(np.sum(in_set))
            if size2 > best_size:
                best_size = size2
                best_support_set = set(np.where(in_set)[0].tolist())
    
    # Construct vectors
    vectors = []
    
    # Type A: 22 vectors (+/- 2 on each coordinate)
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
        
    # Type D: 16 sign vectors per selected support
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
