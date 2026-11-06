import time
import sys

def _compute_table():
    """Pre-compute minimal antichain representations for targets 1..128."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s safety margin before 90s kill
    
    # Generate all 2,3,5-smooth numbers <= 128
    smooth = set()
    for a in range(8):
        for b in range(5):
            for c in range(4):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.add(val)
    
    smooth_list = sorted(smooth)
    n = len(smooth_list)
    
    # Partition into chains by largest odd divisor for local antichain enforcement
    chains = {}
    for idx, val in enumerate(smooth_list):
        odd_part = val
        while odd_part % 2 == 0:
            odd_part //= 2
        if odd_part not in chains:
            chains[odd_part] = []
        chains[odd_part].append(idx)
    
    chain_indices = sorted(chains.keys(), key=lambda x: chains[x][0])
    chain_of = {}  # idx -> chain_id
    for cid, odd_part in enumerate(chain_indices):
        for idx in chains[odd_part]:
            chain_of[idx] = cid
    
    num_chains = len(chain_indices)
    
    reps = [None] * 129
    
    # k=1: smooth numbers
    for t in range(1, 129):
        if t in smooth:
            reps[t] = (t,)
    
    # k=2: pairs
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        for i in range(n):
            a = smooth_list[i]
            if a >= t:
                break
            b = t - a
            if b < a:
                continue
            if b in smooth:
                b_idx = smooth_list.index(b)
                # Antichain: different chains
                if chain_of[i] != chain_of[b_idx]:
                    reps[t] = (a, b)
                    break
    
    # k=3: triples
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        found = False
        for i in range(n):
            a = smooth_list[i]
            if a >= t:
                break
            for j in range(i + 1, n):
                b = smooth_list[j]
                if a + b >= t:
                    break
                c = t - a - b
                if c <= b:
                    continue
                if c in smooth:
                    c_idx = smooth_list.index(c)
                    if c_idx > j:
                        # Antichain: all three in different chains
                        if chain_of[i] != chain_of[j] and chain_of[i] != chain_of[c_idx] and chain_of[j] != chain_of[c_idx]:
                            reps[t] = (a, b, c)
                            found = True
                            break
            if found:
                break
    
    # Check if we need to do DFS for k>=4
    unresolved = [t for t in range(1, 129) if reps[t] is None]
    
    if unresolved:
        def dfs(rem, start_idx, current_terms, k_remaining):
            if time.time() > deadline:
                return None
            if k_remaining == 0:
                return current_terms if rem == 0 else None
            if rem <= 0:
                return None
            if start_idx >= n:
                return None
            
            if n - start_idx < k_remaining:
                return None
            
            # Pruning: minimum possible sum
            if k_remaining <= n - start_idx:
                min_possible = sum(smooth_list[start_idx:start_idx + k_remaining])
                if min_possible > rem:
                    return None
            
            # Pruning: maximum possible sum
            max_start = n - k_remaining
            if max_start >= start_idx:
                max_possible = sum(smooth_list[max_start:max_start + k_remaining])
                if max_possible < rem:
                    return None
            
            # Track which chains are already used
            used_chains = set(chain_of[idx] for idx, _ in current_terms)
            
            for idx in range(start_idx, n):
                val = smooth_list[idx]
                if val > rem:
                    break
                # Antichain: must be in a different chain from all existing terms
                if chain_of[idx] in used_chains:
                    continue
                res = dfs(rem - val, idx + 1, current_terms + [(idx, val)], k_remaining - 1)
                if res is not None:
                    return res
            return None
        
        for t in unresolved:
            if time.time() > deadline:
                break
            for k in range(4, 11):
                res = dfs(t, 0, [], k)
                if res is not None:
                    reps[t] = tuple(v for _, v in res)
                    break
    
    # Fallback
    for t in range(1, 129):
        if reps[t] is None:
            found = False
            for k in range(4, 31):
                if time.time() > deadline:
                    break
                res = dfs(t, 0, [], k)
                if res is not None:
                    reps[t] = tuple(v for _, v in res)
                    found = True
                    break
    
    return reps

_TABLE = _compute_table()

def construct():
    return [list(_TABLE[t]) for t in range(1, 129)]
