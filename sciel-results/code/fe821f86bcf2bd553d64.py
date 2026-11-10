"""Search-based program for longest prime subsequence with non-decreasing gaps <= 10000."""

import time

def construct():
    import numpy as np
    from bisect import bisect_left
    
    # Deadline guard
    deadline = time.time() + 78.0  # safety margin before 90s kill
    
    # Step 1: Sieve primes up to 10000
    sieve = np.ones(10001, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(10001 ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0].tolist()
    n = len(primes)
    
    # Pareto-front DP with predecessor tracking for reconstruction
    # For each prime index i, maintain a list of Pareto-optimal states:
    #   (last_gap, length, prev_index, prev_state_idx)
    # where last_gap = -1 means this is the start of the sequence (no prior gap).
    # Pareto-optimal: no other state has both >= length and <= last_gap.
    
    # pareto[i] = list of tuples (last_gap, length, prev_idx, prev_state_pos)
    # prev_idx = index of previous prime in the subsequence (-1 if start)
    # prev_state_pos = position in pareto[prev_idx] of the state we extended from (-1 if start)
    
    pareto = [[] for _ in range(n)]
    
    # Initialize: each prime can start a sequence of length 1 with last_gap = -1
    for i in range(n):
        pareto[i].append((-1, 1, -1, -1))
    
    best_len = 1
    best_i = 0
    best_state_pos = 0
    
    for i in range(n):
        pi = primes[i]
        new_states = []
        
        # Extend from all j < i
        for j in range(i):
            pj = primes[j]
            for pos_j, (lg, ln, prev_j_idx, prev_j_pos) in enumerate(pareto[j]):
                gap = pi - pj
                if lg == -1 or gap >= lg:
                    # Valid extension: new state ends at i, came from state at j
                    new_states.append((gap, ln + 1, j, pos_j))
        
        # Add to existing pareto[i] (which has the start state)
        pareto[i].extend(new_states)
        
        # Reduce to Pareto front
        # Sort by last_gap ascending; for same gap, keep max length
        pareto[i].sort(key=lambda x: (x[0], -x[1]))
        front = []
        max_len = 0
        for state in pareto[i]:
            lg, ln, _, _ = state
            if ln > max_len:
                front.append(state)
                max_len = ln
        pareto[i] = front
        
        # Track global best
        if front:
            # The state with max length in this front
            best_state = max(front, key=lambda x: x[1])
            if best_state[1] > best_len:
                best_len = best_state[1]
                best_i = i
                best_state_pos = front.index(best_state)
    
    # Reconstruct the best sequence
    seq = []
    cur_i = best_i
    cur_pos = best_state_pos
    
    while cur_i != -1:
        lg, ln, prev_idx, prev_pos = pareto[cur_i][cur_pos]
        seq.append(primes[cur_i])
        if prev_idx == -1:
            break
        cur_i = prev_idx
        cur_pos = prev_pos
    
    seq.reverse()
    
    # Verify the result
    assert len(seq) >= 1
    assert all(seq[k] <= 10000 for k in range(len(seq)))
    assert all(seq[k+1] > seq[k] for k in range(len(seq)-1))
    if len(seq) >= 2:
        gaps = [seq[k+1] - seq[k] for k in range(len(seq)-1)]
        assert all(gaps[k+1] >= gaps[k] for k in range(len(gaps)-1))
    
    return seq
