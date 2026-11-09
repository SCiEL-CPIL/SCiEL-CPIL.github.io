import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55

    target = 479001600
    target_exps = (10, 5, 2, 1, 1)
    primes = [2, 3, 5, 7, 11]

    # Generate all divisors of 12! that are > 12
    divisors = []
    for e1 in range(11):
        for e2 in range(6):
            for e3 in range(3):
                for e4 in range(2):
                    for e5 in range(2):
                        d = (2 ** e1) * (3 ** e2) * (5 ** e3) * (7 ** e4) * (11 ** e5)
                        if d > 12:
                            divisors.append((d, (e1, e2, e3, e4, e5)))

    divisors.sort()
    n_divs = len(divisors)

    if time.time() > deadline:
        return [target]

    # Map exponent tuples to indices
    exp_map = {}
    states = []
    for e1 in range(11):
        for e2 in range(6):
            for e3 in range(3):
                for e4 in range(2):
                    for e5 in range(2):
                        exp_map[(e1, e2, e3, e4, e5)] = len(states)
                        states.append((e1, e2, e3, e4, e5))
    
    n_states = len(states)
    start_idx = exp_map[target_exps]
    zero_idx = exp_map[(0, 0, 0, 0, 0)]

    # Precompute transitions
    # trans[div_idx][state_idx] = next_state_idx or -1
    # div_idx: 0..n_divs-1
    # state_idx: 0..n_states-1
    # This is n_divs * n_states ~ 600k entries.
    # We can compute on the fly or precompute.
    # On the fly is fine if we are careful.
    
    # DP: dp[i][s][c] is boolean?
    # Too much memory if 3D.
    # We want to know if we can reach zero_idx with count <= 12.
    # Let's use a set of (state_idx, count) for each step?
    # Or better:
    # reachable_counts[state_idx] = set of counts to reach this state.
    # Initialize: reachable_counts[start_idx] = {0}
    # For each divisor i:
    #   new_reachable = {}
    #   for s, counts in reachable_counts.items():
    #      for c in counts:
    #         if c < 12:
    #            next_s = subtract(div_exps[i], states[s])
    #            if next_s != -1:
    #               if next_s not in new_reachable: new_reachable[next_s] = set()
    #               new_reachable[next_s].add(c+1)
    #   merge new_reachable into reachable_counts
    #   if zero_idx in reachable_counts and 0 < min(reachable_counts[zero_idx]) <= 12: return True
    
    # This is efficient enough.

    def can_form(max_idx):
        # reachable: dict state_idx -> set of counts
        reachable = {start_idx: {0}}
        for i in range(max_idx + 1):
            d_val, d_exps = divisors[i]
            new_reachable = {}
            for s_idx, counts in reachable.items():
                s_exps = states[s_idx]
                # Check if s_exps >= d_exps
                if (s_exps[0] >= d_exps[0] and s_exps[1] >= d_exps[1] and 
                    s_exps[2] >= d_exps[2] and s_exps[3] >= d_exps[3] and 
                    s_exps[4] >= d_exps[4]):
                    next_exps = (s_exps[0]-d_exps[0], s_exps[1]-d_exps[1], 
                                 s_exps[2]-d_exps[2], s_exps[3]-d_exps[3], 
                                 s_exps[4]-d_exps[4])
                    next_idx = exp_map[next_exps]
                    for c in counts:
                        if c < 12:
                            if next_idx not in new_reachable:
                                new_reachable[next_idx] = set()
                            new_reachable[next_idx].add(c+1)
            # Merge
            for s_idx, counts in new_reachable.items():
                if s_idx in reachable:
                    reachable[s_idx].update(counts)
                else:
                    reachable[s_idx] = counts
            if zero_idx in reachable:
                # Check if any count is <= 12
                # Since we only add if c<12, max count is 12.
                # But we need count >= 1.
                # Start count is 0. So any reachable zero_idx has count >= 1.
                return True
        return False

    # Binary search
    lo, hi = 0, n_divs - 1
    best_idx = n_divs - 1
    found = False
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if can_form(mid):
            best_idx = mid
            hi = mid - 1
            found = True
        else:
            lo = mid + 1
            
    if time.time() > deadline or not found:
        return [target]

    # Reconstruct
    # We need to track which divisors were used.
    # Run DP again with parent tracking.
    # parent[i][s_idx] = (prev_s_idx, prev_count) ?
    # Actually, we just need the set of divisors.
    # Let's store:
    # best_counts[i][s_idx] = set of counts
    # and prev[i][s_idx] = set of (prev_s_idx, prev_count) that led here?
    # Simpler:
    # reachable[i] = dict s_idx -> set of counts
    # parent[i][s_idx] = set of (prev_s_idx, prev_count, div_idx) ?
    # Since we process divisors sequentially, we can store:
    # dp[i][s_idx] = set of counts reachable using divisors 0..i ending at s_idx.
    # prev[i][s_idx] = set of (prev_s_idx, prev_count) from dp[i-1] that transitioned to s_idx using div i.
    
    # Memory: 792 divisors * 792 states * set of counts (max 12).
    # This is too much memory/time to store all.
    
    # Alternative:
    # We know best_idx.
    # We can run a recursive search with memoization?
    # Or just run the DP again, but this time store the path.
    # Since the state space is small, we can store:
    # reachable_states[i] = {s_idx: set(counts)}
    # and for each s_idx in reachable_states[i], store one valid predecessor.
    # pred[i][s_idx] = (prev_s_idx, prev_count)
    
    reachable_states = [None] * (best_idx + 1)
    pred = [None] * (best_idx + 1)
    
    # Init
    init_reach = {start_idx: {0}}
    reachable_states[0] = init_reach
    pred[0] = {}
    
    for i in range(1, best_idx + 1):
        if time.time() > deadline:
            break
        prev_reach = reachable_states[i-1]
        curr_reach = {}
        curr_pred = {}
        
        # Carry over states not using divisor i
        for s_idx, counts in prev_reach.items():
            if s_idx not in curr_reach:
                curr_reach[s_idx] = set()
            curr_reach[s_idx].update(counts)
            # pred for carry over is same as prev
            # But we need to track if this state was *first* reached at i or earlier?
            # We just need *one* valid path.
            # If s_idx is in prev_reach, it's reachable.
            # If we also reach it via divisor i, we can pick either.
            # Let's prioritize using divisor i if it leads to solution?
            # Actually, we just need *any* solution.
            
        # Use divisor i
        d_val, d_exps = divisors[i]
        for s_idx, counts in prev_reach.items():
            s_exps = states[s_idx]
            if (s_exps[0] >= d_exps[0] and s_exps[1] >= d_exps[1] and 
                s_exps[2] >= d_exps[2] and s_exps[3] >= d_exps[3] and 
                s_exps[4] >= d_exps[4]):
                next_exps = (s_exps[0]-d_exps[0], s_exps[1]-d_exps[1], 
                             s_exps[2]-d_exps[2], s_exps[3]-d_exps[3], 
                             s_exps[4]-d_exps[4])
                next_idx = exp_map[next_exps]
                for c in counts:
                    if c < 12:
                        new_c = c + 1
                        if next_idx not in curr_reach:
                            curr_reach[next_idx] = set()
                        if new_c not in curr_reach[next_idx]:
                            curr_reach[next_idx].add(new_c)
                            # Record predecessor
                            if next_idx not in curr_pred:
                                curr_pred[next_idx] = {}
                            if new_c not in curr_pred[next_idx]:
                                curr_pred[next_idx][new_c] = (s_idx, c)
            
        reachable_states[i] = curr_reach
        pred[i] = curr_pred
        
        if zero_idx in curr_reach:
            break

    if zero_idx not in reachable_states[best_idx]:
         return [target]

    # Backtrack
    # We need to find a count c for zero_idx at step best_idx
    # such that we can backtrack to start_idx with count 0.
    # Since we stored preds, we can just pick any count in reachable_states[best_idx][zero_idx]
    # and follow the chain.
    
    # Note: pred[i][s_idx] maps count -> (prev_s_idx, prev_count)
    # If a state was carried over, it might not be in pred[i].
    # We should ensure that if a state is in curr_reach, we have a path.
    # The carry-over states are in prev_reach.
    # So if zero_idx is in curr_reach, and it was carried over, it was in prev_reach.
    # We can just recurse back.
    
    # To simplify, let's just store the full path in a different way?
    # Or just check if zero_idx is in pred[best_idx]?
    # If not, it was carried over. Then check pred[best_idx-1], etc.
    
    # Let's write a helper to find the path.
    def get_path(i, s_idx, c):
        if s_idx == start_idx and c == 0:
            return []
        
        # Check if it was formed by using divisor i
        if i > 0:
            p_pred = pred[i]
            if s_idx in p_pred and c in p_pred[s_idx]:
                prev_s_idx, prev_c = p_pred[s_idx][c]
                path = get_path(i-1, prev_s_idx, prev_c)
                if path is not None:
                    path.append(i)
                    return path
        
        # Check if it was carried over
        if i > 0:
            # It means s_idx, c was in reachable_states[i-1]
            if s_idx in reachable_states[i-1] and c in reachable_states[i-1][s_idx]:
                path = get_path(i-1, s_idx, c)
                if path is not None:
                    return path
        
        return None

    # Find a valid count for zero_idx at best_idx
    valid_count = None
    if zero_idx in reachable_states[best_idx]:
        for c in reachable_states[best_idx][zero_idx]:
            path = get_path(best_idx, zero_idx, c)
            if path is not None:
                valid_count = c
                break

    if valid_count is None:
        return [target]

    # Reconstruct factors
    factors = [divisors[i][0] for i in path]
    return factors
