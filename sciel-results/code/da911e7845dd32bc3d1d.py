import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def check_validity(P, d):
        """Check if adding step d to sequence P is valid.
        P is the list of values. d is the next step.
        Returns (new_P, new_used_diffs, is_valid)
        """
        P_next_val = P[-1] + d
        if P_next_val > 256:
            return None, None, False
        
        # Calculate new differences
        new_diffs = []
        for i in range(len(P)):
            diff = P_next_val - P[i]
            new_diffs.append(diff)
            
        return P_next_val, new_diffs, True

    def get_used_diffs(P):
        """Calculate all consecutive block sums (differences) for P."""
        used = set()
        for i in range(len(P)):
            s = 0
            for j in range(i, len(P)):
                s += P[j+1] - P[j] # Wait, P is values. 
                # Actually, the differences are P[j] - P[i].
                # Let's stick to the difference array representation for clarity in state.
        # This function is not used in the optimized version below, 
        # but conceptually the state is (differences_list, used_diffs_set).
        pass

    # State: (diffs_list, used_diffs_set, last_val)
    # diffs_list is the sequence of steps.
    # last_val is the current value in [1, 256].
    # We start with a single element, so diffs_list is empty? 
    # No, the problem asks for a sequence of integers.
    # Let a = [a_1, ..., a_n].
    # Consecutive block sums are sum(a_i..a_j).
    # This is equivalent to saying that the prefix sums S_k = sum(a_1..a_k) must have distinct differences S_j - S_i.
    # So we are looking for a sequence of prefix sums S_0=0, S_1, ..., S_n such that:
    # 1. S_k < S_{k+1} (strictly increasing)
    # 2. All S_j - S_i (j>i) are distinct.
    # 3. S_n <= 256? No, the elements a_i are in [1, 256]. 
    # Wait, the task says "sequence of integers in [1, 256]".
    # So a_i \in [1, 256].
    # This means the steps d_i = a_i are in [1, 256].
    # And the sum of any consecutive block is distinct.
    
    # Let's re-read carefully: "Return a strictly increasing sequence of integers in [1,256]."
    # This usually means the elements themselves are in [1, 256].
    # "Sums of all nonempty consecutive blocks must be distinct."
    
    # If the elements are in [1, 256], the max sum is 256 * n.
    # The constraint is on the sums of blocks.
    
    # My previous state representation:
    # P = [a_1, ..., a_k].
    # used_diffs = {P[j] - P[i] for j > i}.
    # This is correct for "distinct consecutive block sums" if we interpret P as prefix sums?
    # No, if P is the sequence of elements, the sum of block i..j is sum(P[i..j]).
    # These sums are distinct.
    # This is NOT the same as P[j] - P[i] being distinct.
    # P[j] - P[i] is the sum of elements from i+1 to j.
    # So yes, if we define S_k = sum(a_1..a_k), then the block sums are S_j - S_{i-1}.
    # So the condition is that all differences S_j - S_i are distinct.
    # And the elements a_k = S_k - S_{k-1} must be in [1, 256].
    # Also "strictly increasing sequence of integers" usually refers to the elements a_k?
    # "Return a strictly increasing sequence of integers in [1,256]."
    # Yes, a_1 < a_2 < ... < a_n.
    # So a_k are the elements.
    
    # So the state should be:
    # a: list of elements
    # S: list of prefix sums (S[0]=0, S[k]=sum(a[0..k-1]))
    # used_diffs: set of all S[j] - S[i] for j > i.
    
    # Let's refactor the state to (a, S, used_diffs).
    
    def expand(state):
        a, S, used_diffs = state
        last_a = a[-1] if a else 0
        last_S = S[-1]
        
        candidates = []
        # Next element a_next must be > last_a and in [1, 256]
        for a_next in range(last_a + 1, 257):
            S_next = last_S + a_next
            # Check if S_next - S[i] is already in used_diffs for any i
            valid = True
            new_diffs = []
            for i in range(len(S)):
                diff = S_next - S[i]
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if valid:
                new_a = a + [a_next]
                new_S = S + [S_next]
                new_used = used_diffs | set(new_diffs)
                candidates.append((new_a, new_S, new_used))
        
        return candidates

    def greedy_extend(a, S, used_diffs, time_limit):
        while time.time() < time_limit:
            found = False
            last_a = a[-1] if a else 0
            last_S = S[-1]
            
            for a_next in range(last_a + 1, 257):
                S_next = last_S + a_next
                valid = True
                new_diffs = []
                for i in range(len(S)):
                    diff = S_next - S[i]
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if valid:
                    a = a + [a_next]
                    S = S + [S_next]
                    used_diffs = used_diffs | set(new_diffs)
                    found = True
                    break
            
            if not found:
                break
        return a, S, used_diffs

    def dfs_extend(a, S, used_diffs, depth, time_limit, max_depth):
        """Backtracking search to find longer sequences."""
        if depth >= max_depth or time.time() > time_limit:
            return a, S, used_diffs
        
        last_a = a[-1] if a else 0
        last_S = S[-1]
        
        best_local = (a, S, used_diffs)
        
        for a_next in range(last_a + 1, 257):
            if time.time() > time_limit:
                break
                
            S_next = last_S + a_next
            valid = True
            new_diffs = []
            for i in range(len(S)):
                diff = S_next - S[i]
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if valid:
                new_a = a + [a_next]
                new_S = S + [S_next]
                new_used = used_diffs | set(new_diffs)
                
                # Recurse
                res_a, res_S, res_used = dfs_extend(new_a, new_S, new_used, depth + 1, time_limit, max_depth)
                
                if len(res_a) > len(best_local[0]):
                    best_local = (res_a, res_S, res_used)
                    
                # Pruning: if we found a sequence that is already longer than current best global, 
                # we can stop early? No, we want the longest.
                
        return best_local

    # Initial state: empty sequence
    # a = [], S = [0], used_diffs = set()
    init_state = ([], [0], set())
    
    best_seq = []
    best_len = 0
    
    # Beam Search
    beam = [init_state]
    
    # Heuristic: Smaller last element (more room for increasing), 
    # Smaller max prefix sum (more room in sum space? No, sum space is not bounded directly, 
    # but diffs must be distinct. Smaller values generally allow more distinct diffs?)
    # Actually, smaller a_next allows more future steps.
    
    for B in [20, 40, 60]:
        if time.time() > deadline - 10:
            break
            
        beam = [init_state]
        local_best = []
        
        while beam and time.time() < deadline - 15:
            next_beam = []
            for state in beam:
                candidates = expand(state)
                if candidates:
                    next_beam.extend(candidates)
                else:
                    a = state[0]
                    if len(a) > len(local_best):
                        local_best = a
            
            if not next_beam:
                break
                
            # Sort by heuristic: prefer smaller last element, then smaller prefix sum
            def state_score(s):
                a, S, used_diffs = s
                return (a[-1] if a else 0, S[-1])
            
            next_beam.sort(key=state_score)
            beam = next_beam[:B]
            
            # Track best
            for state in beam:
                a = state[0]
                if len(a) > len(local_best):
                    local_best = a
        
        # DFS on top beam states
        if beam and time.time() < deadline - 5:
            def state_score(s):
                a, S, used_diffs = s
                return (a[-1] if a else 0, S[-1])
            
            beam.sort(key=state_score)
            for state in beam[:min(10, len(beam))]:
                if time.time() > deadline - 2:
                    break
                a, S, used_diffs = state
                # Use DFS to explore deeper
                res_a, res_S, res_used = dfs_extend(a, S, used_diffs, len(a), deadline - 1, 200)
                if len(res_a) > len(local_best):
                    local_best = res_a
        
        if len(local_best) > best_len:
            best_seq = local_best
            best_len = len(local_best)
            
    return best_seq
