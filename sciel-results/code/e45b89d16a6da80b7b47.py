import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    MAX_SUM = 20000
    used_sums = bytearray(MAX_SUM + 1)
    
    def is_valid(x, suffix_sums):
        if used_sums[x]:
            return False
        for s in suffix_sums:
            if x + s > MAX_SUM:
                continue
            if used_sums[x + s]:
                return False
        return True

    def add_sums(x, suffix_sums):
        """Add x and all x+s to used_sums. Returns list of values added for backtracking."""
        added = [x]
        used_sums[x] = 1
        for s in suffix_sums:
            ns = x + s
            if ns <= MAX_SUM:
                used_sums[ns] = 1
                added.append(ns)
        return added

    def remove_sums(added):
        """Remove all values in 'added' from used_sums."""
        for v in added:
            used_sums[v] = 0

    # Greedy to get a good starting point
    greedy_seq = []
    suffix_sums = []
    
    for x in range(1, 257):
        if is_valid(x, suffix_sums):
            greedy_seq.append(x)
            added = add_sums(x, suffix_sums)
            new_suffix = [x] + [x + s for s in suffix_sums if x + s <= MAX_SUM]
            suffix_sums = new_suffix
            
    best_seq = greedy_seq.copy()
    best_len = len(best_seq)
    
    # Reset used_sums
    used_sums[:] = b'\x00' * (MAX_SUM + 1)
    
    # Build suffix_sums from greedy_seq for DFS start
    def build_suffix_sums(seq):
        ss = []
        for i in range(len(seq) - 1, -1, -1):
            s = seq[i]
            if ss:
                s += ss[0]
            ss.insert(0, s)
        return ss

    # DFS with proper backtracking
    def dfs(last_val, suffix_sums, seq, depth):
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
        remaining = 256 - last_val
        if current_len + remaining <= best_len:
            return

        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            if is_valid(x, suffix_sums):
                # Apply change
                added = add_sums(x, suffix_sums)
                new_suffix = [x] + [x + s for s in suffix_sums if x + s <= MAX_SUM]
                seq.append(x)
                
                dfs(x, new_suffix, seq, depth + 1)
                
                # Backtrack
                remove_sums(added)
                seq.pop()

    # Start DFS from empty sequence, but use greedy as initial best
    dfs(0, [], [], 0)
    
    # Local search: try to improve by replacing elements
    if time.time() < deadline - 5.0:
        # Try to extend or improve the best sequence found
        for attempt in range(10):
            if time.time() > deadline - 1.0:
                break
            current = best_seq.copy()
            # Try replacing each element with a larger one that still works
            improved = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 1.0:
                    break
                # Try to replace current[i] with a larger value
                for new_val in range(current[i] + 1, 257):
                    # Check if we can replace
                    test_seq = current.copy()
                    test_seq[i] = new_val
                    
                    # Validate the sequence
                    valid = True
                    # Check strictly increasing
                    for j in range(1, len(test_seq)):
                        if test_seq[j] <= test_seq[j-1]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check all consecutive block sums are distinct
                    sums = set()
                    for start in range(len(test_seq)):
                        s = 0
                        for end in range(start, len(test_seq)):
                            s += test_seq[end]
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    
                    if valid and len(test_seq) > best_len:
                        best_seq = test_seq
                        best_len = len(test_seq)
                        improved = True
                        break
                    elif valid and len(test_seq) == best_len:
                        # Keep the lexicographically smaller one for potential future improvements
                        pass
            
            if not improved:
                break
    
    return best_seq
