import time
import math
import sys

def _is_valid(seq):
    """Check if sequence has all distinct consecutive block sums."""
    n = len(seq)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_forward(seq, max_val=256):
    """Greedy forward extension."""
    result = list(seq)
    sums = set()
    for i in range(len(result)):
        s = 0
        for j in range(i, len(result)):
            s += result[j]
            sums.add(s)
    start = (result[-1] + 1) if result else 1
    for x in range(start, max_val + 1):
        new_sums = []
        t = x
        valid = True
        for a in reversed(result):
            t += a
            new_sums.append(t)
        for s in new_sums:
            if s in sums:
                valid = False
                break
        if valid:
            result.append(x)
            sums.update(new_sums)
    return result

def _dfs_bitmask(max_val=256, deadline=None, start_a=1):
    """
    DFS with bitmask for forbidden differences.
    We build prefix sums P[0]=0, P[1], P[2], ...
    a[k] = P[k+1] - P[k], must be strictly increasing, each in [1, max_val].
    All pairwise differences P[j]-P[i] (i<j) must be distinct.
    """
    if deadline is None:
        deadline = time.time() + 85
    
    best = []
    best_len = 0
    
    def dfs(P, forbidden_mask, last_a):
        nonlocal best, best_len
        
        n = len(P) - 1  # number of elements in sequence
        if n > best_len:
            best_len = n
            best = [P[i+1] - P[i] for i in range(n)]
        
        if last_a >= max_val:
            return
        
        # Upper bound: how many more can we add?
        remaining_slots = max_val - last_a
        if n + remaining_slots <= best_len:
            return
        
        # Sum capacity bound
        sum_rem = (max_val * (max_val + 1) // 2) - (last_a * (last_a + 1) // 2)
        P_max = P[-1] + sum_rem
        # Need m*(m-1)/2 <= P_max for m total P points
        m_bound = (1 + math.isqrt(1 + 8 * P_max)) // 2
        if m_bound - 1 <= best_len:
            return
        
        for a in range(last_a + 1, max_val + 1):
            if time.time() > deadline:
                return
            
            new_P = P[-1] + a
            valid = True
            new_bits = 0
            
            for i in range(len(P)):
                d = new_P - P[i]
                if (forbidden_mask >> d) & 1:
                    valid = False
                    break
                new_bits |= (1 << d)
            
            if not valid:
                continue
            
            dfs(P + [new_P], forbidden_mask | new_bits, a)
    
    dfs([0], 0, 0)
    return best

def _dfs_randomized(max_val=256, deadline=None, seed=0):
    """Randomized DFS: try candidates in random order."""
    import random
    if deadline is None:
        deadline = time.time() + 85
    
    rng = random.Random(seed)
    best = []
    best_len = 0
    
    def dfs(P, forbidden_mask, last_a):
        nonlocal best, best_len
        
        n = len(P) - 1
        if n > best_len:
            best_len = n
            best = [P[i+1] - P[i] for i in range(n)]
        
        if last_a >= max_val:
            return
        
        remaining_slots = max_val - last_a
        if n + remaining_slots <= best_len:
            return
        
        sum_rem = (max_val * (max_val + 1) // 2) - (last_a * (last_a + 1) // 2)
        P_max = P[-1] + sum_rem
        m_bound = (1 + math.isqrt(1 + 8 * P_max)) // 2
        if m_bound - 1 <= best_len:
            return
        
        candidates = list(range(last_a + 1, max_val + 1))
        rng.shuffle(candidates)
        
        for a in candidates:
            if time.time() > deadline:
                return
            
            new_P = P[-1] + a
            valid = True
            new_bits = 0
            
            for i in range(len(P)):
                d = new_P - P[i]
                if (forbidden_mask >> d) & 1:
                    valid = False
                    break
                new_bits |= (1 << d)
            
            if not valid:
                continue
            
            dfs(P + [new_P], forbidden_mask | new_bits, a)
    
    dfs([0], 0, 0)
    return best

def _dfs_backwards(max_val=256, deadline=None):
    """
    Try larger a values first (backwards order) to explore different regions.
    """
    if deadline is None:
        deadline = time.time() + 85
    
    best = []
    best_len = 0
    
    def dfs(P, forbidden_mask, last_a):
        nonlocal best, best_len
        
        n = len(P) - 1
        if n > best_len:
            best_len = n
            best = [P[i+1] - P[i] for i in range(n)]
        
        if last_a >= max_val:
            return
        
        remaining_slots = max_val - last_a
        if n + remaining_slots <= best_len:
            return
        
        sum_rem = (max_val * (max_val + 1) // 2) - (last_a * (last_a + 1) // 2)
        P_max = P[-1] + sum_rem
        m_bound = (1 + math.isqrt(1 + 8 * P_max)) // 2
        if m_bound - 1 <= best_len:
            return
        
        for a in range(max_val, last_a, -1):
            if time.time() > deadline:
                return
            
            new_P = P[-1] + a
            valid = True
            new_bits = 0
            
            for i in range(len(P)):
                d = new_P - P[i]
                if (forbidden_mask >> d) & 1:
                    valid = False
                    break
                new_bits |= (1 << d)
            
            if not valid:
                continue
            
            dfs(P + [new_P], forbidden_mask | new_bits, a)
    
    dfs([0], 0, 0)
    return best

def _beam_search(max_val=256, deadline=None, beam_width=200):
    """
    Beam search: keep top-k partial sequences at each level.
    """
    if deadline is None:
        deadline = time.time() + 85
    
    # State: (P_list, forbidden_mask, last_a)
    # Score: length (primary), then P[-1] (secondary - larger is better for capacity)
    states = [([0], 0, 0)]
    best = []
    best_len = 0
    
    while time.time() < deadline:
        next_states = []
        for P, mask, last_a in states:
            if last_a >= max_val:
                n = len(P) - 1
                if n > best_len:
                    best_len = n
                    best = [P[i+1] - P[i] for i in range(n)]
                continue
            
            for a in range(last_a + 1, max_val + 1):
                new_P = P[-1] + a
                valid = True
                new_bits = 0
                
                for i in range(len(P)):
                    d = new_P - P[i]
                    if (mask >> d) & 1:
                        valid = False
                        break
                    new_bits |= (1 << d)
                
                if not valid:
                    continue
                
                next_states.append((P + [new_P], mask | new_bits, a))
        
        if not next_states:
            break
        
        # Sort by length (desc), then P[-1] (desc)
        next_states.sort(key=lambda s: (len(s[0]), s[0][-1]), reverse=True)
        states = next_states[:beam_width]
        
        # Update best
        if states:
            top_len = len(states[0][0]) - 1
            if top_len > best_len:
                best_len = top_len
                best = [states[0][0][i+1] - states[0][0][i] for i in range(top_len)]
    
    return best

def _dfs_with_pruning(max_val=256, deadline=None, min_a=1):
    """
    DFS with additional pruning: skip a values that are too close together
    (they create small differences that are more likely to collide).
    """
    if deadline is None:
        deadline = time.time() + 85
    
    best = []
    best_len = 0
    
    def dfs(P, forbidden_mask, last_a):
        nonlocal best, best_len
        
        n = len(P) - 1
        if n > best_len:
            best_len = n
            best = [P[i+1] - P[i] for i in range(n)]
        
        if last_a >= max_val:
            return
        
        remaining_slots = max_val - last_a
        if n + remaining_slots <= best_len:
            return
        
        sum_rem = (max_val * (max_val + 1) // 2) - (last_a * (last_a + 1) // 2)
        P_max = P[-1] + sum_rem
        m_bound = (1 + math.isqrt(1 + 8 * P_max)) // 2
        if m_bound - 1 <= best_len:
            return
        
        # Pruning: if we have many P points, small differences are risky
        # Prefer larger jumps
        start = last_a + 1
        # If we need more elements, be more aggressive with pruning
        for a in range(start, max_val + 1):
            if time.time() > deadline:
                return
            
            new_P = P[-1] + a
            valid = True
            new_bits = 0
            
            for i in range(len(P)):
                d = new_P - P[i]
                if (forbidden_mask >> d) & 1:
                    valid = False
                    break
                new_bits |= (1 << d)
            
            if not valid:
                continue
            
            dfs(P + [new_P], forbidden_mask | new_bits, a)
    
    dfs([0], 0, 0)
    return best

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin
    
    # 1. Greedy for quick baseline
    greedy = _greedy_forward([])
    best = greedy
    best_len = len(greedy)
    
    # 2. Beam search (fast, good heuristic)
    if time.time() < deadline - 10:
        beam = _beam_search(deadline=deadline, beam_width=500)
        if len(beam) > best_len:
            best = beam
            best_len = len(best)
    
    # 3. DFS with bitmask (increasing order)
    if time.time() < deadline - 5:
        dfs1 = _dfs_bitmask(deadline=deadline)
        if len(dfs1) > best_len:
            best = dfs1
            best_len = len(best)
    
    # 4. DFS backwards (decreasing order)
    if time.time() < deadline - 5:
        dfs2 = _dfs_backwards(deadline=deadline)
        if len(dfs2) > best_len:
            best = dfs2
            best_len = len(best)
    
    # 5. Randomized DFS with multiple seeds
    for seed in [1, 7, 42, 123, 999]:
        if time.time() > deadline - 3:
            break
        r = _dfs_randomized(deadline=deadline, seed=seed)
        if len(r) > best_len:
            best = r
            best_len = len(best)
    
    # 6. DFS with pruning
    if time.time() < deadline - 2:
        dfs3 = _dfs_with_pruning(deadline=deadline)
        if len(dfs3) > best_len:
            best = dfs3
            best_len = len(best)
    
    # Final validation
    if best and not _is_valid(best):
        best = _greedy_forward([])
    
    if not best:
        best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return best

# EVOLVE-BLOCK-END
