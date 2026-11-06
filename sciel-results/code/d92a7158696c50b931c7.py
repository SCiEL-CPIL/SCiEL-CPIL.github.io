import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # State:
    # used: integer bitset where bit i is 1 if sum i is used
    # suffix_mask: integer bitset where bit s is 1 if s is a suffix sum of current seq
    # last_val: last element in sequence
    # seq: list of elements
    # best_seq: best sequence found so far
    # best_len: length of best_seq

    best_seq = []
    best_len = 0

    # Greedy initialization to get a decent starting point
    # We use the same bitset logic for speed
    used = 0
    suffix_mask = 0
    greedy_seq = []
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        # Check validity
        if (used & (1 << x)) or ((used >> x) & suffix_mask):
            continue
        
        # Add x
        greedy_seq.append(x)
        # Update used
        # New sums: x and x+s for all s in suffix_mask
        # (1 << x) sets bit x
        # (suffix_mask << x) sets bits x+s
        used |= (1 << x) | (suffix_mask << x)
        # Update suffix_mask
        # New suffixes: x and x+s for all s in suffix_mask
        suffix_mask = (1 << x) | (suffix_mask << x)

    if len(greedy_seq) > best_len:
        best_len = len(greedy_seq)
        best_seq = greedy_seq.copy()

    # DFS with backtracking
    # We start from the greedy sequence? Or from scratch?
    # Starting from scratch with the fast bitset check might allow finding a longer sequence
    # if the greedy got stuck in a local minimum.
    # However, restarting from scratch takes time.
    # Let's try DFS from scratch, but use the greedy result as the initial best.
    
    def dfs(last_val, used, suffix_mask, seq, depth):
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
        
        # Upper bound: max possible additional elements
        # Simple bound: 256 - last_val
        if current_len + (256 - last_val) <= best_len:
            return
            
        # Heuristic: If we can't even reach best_len with remaining range, prune.
        # More sophisticated bound:
        # The number of distinct sums required for k more elements is k(k+1)/2 + current_sums...
        # But simple bound is usually ok with fast search.

        # Iterate candidates
        # To maximize length, we might want to try smaller numbers first?
        # Or numbers that are "far" from existing sums?
        # Let's stick to increasing order for simplicity, as bitset check is fast.
        
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            # Check validity
            if (used & (1 << x)) or ((used >> x) & suffix_mask):
                continue
            
            # Apply change
            new_used = used | (1 << x) | (suffix_mask << x)
            new_suffix_mask = (1 << x) | (suffix_mask << x)
            seq.append(x)
            
            dfs(x, new_used, new_suffix_mask, seq, depth + 1)
            
            # Backtrack
            seq.pop()

    # Start DFS
    # We can also try starting DFS from the greedy state to extend it?
    # But the greedy state might be suboptimal.
    # Starting from 0 allows finding a completely different sequence.
    # Given the speedup, we can afford to search more.
    
    dfs(0, 0, 0, [], 0)

    return best_seq
