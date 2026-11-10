import time
import random

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    def compute_counts(S):
        """Compute sum counts for a set S."""
        counts = [0] * (2 * N + 1)
        for i in range(len(S)):
            x = S[i]
            counts[2 * x] += 1
            for j in range(i + 1, len(S)):
                y = S[j]
                counts[x + y] += 1
        return counts
    
    def can_add_fast(x, S, counts):
        """Check if x can be added to S with given counts."""
        # Check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for all y in S
        for y in S:
            if counts[x + y] >= 2:
                return False
        return True
    
    def add_fast(x, S, counts):
        """Add x to S and update counts."""
        for y in S:
            counts[x + y] += 1
        counts[2 * x] += 1
        S.append(x)
    
    def greedy_build(order, max_steps=None):
        """Greedy construction from a given order."""
        S = []
        counts = [0] * (2 * N + 1)
        in_set = [False] * (N + 1)
        for x in order:
            if max_steps is not None and len(S) >= max_steps:
                break
            if in_set[x]:
                continue
            if can_add_fast(x, S, counts):
                add_fast(x, S, counts)
                in_set[x] = True
        return S, counts, in_set
    
    def repair_from_state(S, counts, in_set, candidates):
        """Try to add candidates to the current set."""
        new_S = list(S)
        new_counts = list(counts)
        new_in_set = list(in_set)
        for x in candidates:
            if new_in_set[x]:
                continue
            if can_add_fast(x, new_S, new_counts):
                add_fast(x, new_S, new_counts)
                new_in_set[x] = True
        return new_S, new_counts, new_in_set
    
    # Initial greedy attempts
    best_size = 0
    best_set = []
    best_counts = []
    best_in_set = []
    
    # Try ascending order
    S, counts, in_set = greedy_build(list(range(1, N + 1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        best_counts = counts
        best_in_set = in_set
    
    # Try descending order
    S, counts, in_set = greedy_build(list(range(N, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        best_counts = counts
        best_in_set = in_set
    
    rng = random.Random(12345)
    
    # Iterated Local Search
    current_S = best_set
    current_counts = best_counts
    current_in_set = best_in_set
    
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline:
        # Perturbation: Remove K random elements
        K = rng.randint(1, 15)
        
        S_p = list(current_S)
        counts_p = list(current_counts)
        in_set_p = list(current_in_set)
        
        removed = []
        for _ in range(min(K, len(S_p))):
            idx = rng.randrange(len(S_p))
            x = S_p[idx]
            # Remove x from list
            S_p[idx] = S_p[-1]
            S_p.pop()
            in_set_p[x] = False
            removed.append(x)
            
            # Update counts: subtract x's contributions
            counts_p[2 * x] -= 1
            for y in S_p:
                counts_p[x + y] -= 1
        
        # Repair: Try to add elements back
        candidates = [x for x in range(1, N + 1) if not in_set_p[x]]
        rng.shuffle(candidates)
        
        S_r, counts_r, in_set_r = repair_from_state(S_p, counts_p, in_set_p, candidates)
        
        if len(S_r) > best_size:
            best_size = len(S_r)
            best_set = S_r
            best_counts = counts_r
            best_in_set = in_set_r
            current_S = S_r
            current_counts = counts_r
            current_in_set = in_set_r
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Try a fresh greedy with random order
                order = list(range(1, N + 1))
                rng.shuffle(order)
                S_g, counts_g, in_set_g = greedy_build(order)
                
                if len(S_g) > best_size:
                    best_size = len(S_g)
                    best_set = S_g
                    best_counts = counts_g
                    best_in_set = in_set_g
                    current_S = S_g
                    current_counts = counts_g
                    current_in_set = in_set_g
                no_improve_count = 0
    
    return best_set
