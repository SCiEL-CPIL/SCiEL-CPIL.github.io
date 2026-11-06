import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    best_seq = []
    best_len = 0

    def check_valid(x, used, suffix_mask):
        """Check if x can be appended to the sequence."""
        if (used & (1 << x)):
            return False
        if ((used >> x) & suffix_mask):
            return False
        return True

    def add_to_state(x, used, suffix_mask):
        """Return new state after adding x."""
        new_used = used | (1 << x) | (suffix_mask << x)
        new_suffix_mask = (1 << x) | (suffix_mask << x)
        return new_used, new_suffix_mask

    def greedy_init(strategy=0):
        """Generate a sequence using a greedy strategy."""
        used = 0
        suffix_mask = 0
        seq = []
        
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if strategy == 0:
                # Standard increasing
                candidates = [x]
            elif strategy == 1:
                # Skip some to create gaps
                if x % 3 == 0:
                    continue
                candidates = [x]
            elif strategy == 2:
                # Try to maximize gaps
                if x % 2 == 0:
                    continue
                candidates = [x]
            else:
                candidates = [x]
            
            for c in candidates:
                if check_valid(c, used, suffix_mask):
                    used, suffix_mask = add_to_state(c, used, suffix_mask)
                    seq.append(c)
                    break
        
        return seq

    # Try multiple greedy strategies
    for strat in range(4):
        seq = greedy_init(strat)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq.copy()

    # Also try a "maximize gaps" greedy
    used = 0
    suffix_mask = 0
    seq = []
    # Start with larger gaps
    for x in range(1, 257, 2):  # Try odd numbers first
        if time.time() > deadline:
            break
        if check_valid(x, used, suffix_mask):
            used, suffix_mask = add_to_state(x, used, suffix_mask)
            seq.append(x)
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq.copy()

    # Try starting with larger numbers
    used = 0
    suffix_mask = 0
    seq = []
    for x in range(256, 0, -1):  # Descending, then we'll reverse
        if time.time() > deadline:
            break
        if check_valid(x, used, suffix_mask):
            used, suffix_mask = add_to_state(x, used, suffix_mask)
            seq.append(x)
    seq.reverse()
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq.copy()

    def dfs(last_val, used, suffix_mask, seq, depth, max_depth=30):
        """DFS with backtracking to find longer sequences."""
        nonlocal best_seq
        nonlocal best_len
        
        current_len = len(seq)
        
        if current_len > best_len:
            best_len = current_len
            best_seq = seq.copy()
        
        if time.time() > deadline:
            return
        
        if last_val >= 256:
            return
        
        # Upper bound: can we possibly beat best_len?
        # We need at least (best_len - current_len) more elements
        # Each new element adds at least 1 to the count
        # Simple bound: remaining values
        remaining = 256 - last_val
        if current_len + remaining <= best_len:
            return
        
        # Better bound: consider that each new element x requires
        # x + suffix_sums to be unused. As we add more elements,
        # the suffix_mask grows, making it harder to add more.
        # Heuristic bound: assume we can add at most sqrt(remaining) more
        # This is very conservative but helps prune
        if current_len + int((256 - last_val) ** 0.5) + 1 <= best_len:
            return
        
        if depth >= max_depth:
            return
        
        # Candidate ordering: try values that create fewer new conflicts
        # For each candidate x, count how many new sums it creates
        # Prefer candidates that create fewer conflicts
        candidates = []
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            if check_valid(x, used, suffix_mask):
                # Count new sums created
                new_sums = (1 << x) | (suffix_mask << x)
                # Count bits set
                conflict_count = bin(new_sums & ~used).count('1')
                candidates.append((conflict_count, x))
        
        # Sort by conflict count (fewer conflicts first)
        candidates.sort()
        
        for _, x in candidates:
            if time.time() > deadline:
                return
            
            new_used, new_suffix_mask = add_to_state(x, used, suffix_mask)
            seq.append(x)
            
            dfs(x, new_used, new_suffix_mask, seq, depth + 1)
            
            seq.pop()

    # Start DFS from empty sequence
    dfs(0, 0, 0, [], 0, max_depth=25)

    # Also try DFS from the best greedy sequence, trying to extend it
    if best_len > 0:
        last_val = best_seq[-1]
        used = 0
        suffix_mask = 0
        for x in best_seq:
            used, suffix_mask = add_to_state(x, used, suffix_mask)
        
        def dfs_extend(last_val, used, suffix_mask, seq, depth):
            nonlocal best_seq
            nonlocal best_len
            
            current_len = len(seq)
            
            if current_len > best_len:
                best_len = current_len
                best_seq = seq.copy()
            
            if time.time() > deadline:
                return
            
            if last_val >= 256:
                return
            
            remaining = 256 - last_val
            if current_len + remaining <= best_len:
                return
            
            if depth >= 15:
                return
            
            candidates = []
            for x in range(last_val + 1, 257):
                if time.time() > deadline:
                    return
                if check_valid(x, used, suffix_mask):
                    new_sums = (1 << x) | (suffix_mask << x)
                    conflict_count = bin(new_sums & ~used).count('1')
                    candidates.append((conflict_count, x))
            
            candidates.sort()
            
            for _, x in candidates:
                if time.time() > deadline:
                    return
                
                new_used, new_suffix_mask = add_to_state(x, used, suffix_mask)
                seq.append(x)
                
                dfs_extend(x, new_used, new_suffix_mask, seq, depth + 1)
                
                seq.pop()

        dfs_extend(last_val, used, suffix_mask, best_seq.copy(), 0)

    return best_seq
