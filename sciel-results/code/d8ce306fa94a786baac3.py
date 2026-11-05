import time
import random

def _is_valid(seq):
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_forward_optimized():
    selected = []
    sums = set()
    for x in range(1, 257):
        # x is new last element
        # New sums are suffixes of selected + x
        valid = True
        new_sums = []
        current_sum = 0
        for a in reversed(selected):
            current_sum += a
            new_sum = current_sum + x
            if new_sum in sums:
                valid = False
                break
            new_sums.append(new_sum)
        
        if not valid:
            continue
        if x in sums:
            continue
        new_sums.append(x)
        
        selected.append(x)
        sums.update(new_sums)
    return selected

def _greedy_reverse_optimized():
    selected = []
    sums = set()
    for x in range(256, 0, -1):
        # x is new first element
        valid = True
        new_sums = []
        current_sum = 0
        for a in selected:
            current_sum += a
            new_sum = x + current_sum
            if new_sum in sums:
                valid = False
                break
            new_sums.append(new_sum)
        
        if not valid:
            continue
        if x in sums:
            continue
        new_sums.append(x)
        
        selected.insert(0, x)
        sums.update(new_sums)
    return selected

def _randomized_trial(candidates, deadline):
    """Try to build a valid sequence by inserting elements in given order."""
    selected = []
    sums = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        # Find insertion point in sorted order
        idx = 0
        for j in range(len(selected)):
            if selected[j] < x:
                idx = j + 1
            else:
                break
        
        # Check if inserting x at idx creates duplicate sums
        n = len(selected)
        new_sums = set()
        valid = True
        
        # Prefix sums from left: sum of selected[i:idx] for i from idx-1 down to 0
        left_sums = [0]
        for k in range(1, idx + 1):
            left_sums.append(left_sums[-1] + selected[idx - k])
        
        # Suffix sums from right: sum of selected[idx:j] for j from idx to n-1
        right_sums = [0]
        for k in range(1, n - idx + 1):
            right_sums.append(right_sums[-1] + selected[idx + k - 1])
        
        # Now check all new blocks
        for i in range(idx + 1):
            for j in range(n - idx + 1):
                if i == 0 and j == 0:
                    block_sum = x
                else:
                    block_sum = left_sums[i] + x + right_sums[j]
                
                if block_sum in sums or block_sum in new_sums:
                    valid = False
                    break
                new_sums.add(block_sum)
            
            if not valid:
                break
        
        if not valid:
            continue
        
        selected.insert(idx, x)
        sums.update(new_sums)
    
    return selected

def _backtracking_search(deadline, max_depth=15, branching_factor=3):
    """
    Depth-first backtracking search to explore non-greedy branches.
    At each step, try a limited number of valid candidates (not just the smallest).
    """
    best_seq = []
    
    def dfs(selected, sums, min_val, depth):
        nonlocal best_seq
        
        if time.time() > deadline:
            return
        
        if len(selected) > len(best_seq):
            best_seq = selected[:]
        
        if depth >= max_depth:
            return
        
        # Find valid candidates to append
        candidates = []
        for x in range(min_val, 257):
            # Check if x can be appended
            valid = True
            current_sum = 0
            for a in reversed(selected):
                current_sum += a
                new_sum = current_sum + x
                if new_sum in sums:
                    valid = False
                    break
            if valid and x not in sums:
                candidates.append(x)
            
            # Limit branching
            if len(candidates) >= branching_factor:
                break
        
        if not candidates:
            return
        
        # Try each candidate
        for x in candidates:
            if time.time() > deadline:
                return
            
            # Calculate new sums
            new_sums = []
            current_sum = 0
            for a in reversed(selected):
                current_sum += a
                new_sums.append(current_sum + x)
            new_sums.append(x)
            
            # Update sums
            old_sums = set(sums)
            sums.update(new_sums)
            selected.append(x)
            
            dfs(selected, sums, x + 1, depth + 1)
            
            # Backtrack
            selected.pop()
            sums = old_sums
    
    dfs([], set(), 1, 0)
    return best_seq

def _local_polish(seq, deadline):
    """
    Try to improve the sequence by replacing elements or extending it.
    """
    if not seq:
        return seq
    
    best = seq[:]
    sums = set()
    for i in range(len(best)):
        s = 0
        for j in range(i, len(best)):
            s += best[j]
            sums.add(s)
    
    # Try extending from the end
    last_val = best[-1]
    for x in range(last_val + 1, 257):
        if time.time() > deadline:
            break
        
        valid = True
        current_sum = 0
        for a in reversed(best):
            current_sum += a
            new_sum = current_sum + x
            if new_sum in sums:
                valid = False
                break
        if valid and x not in sums:
            new_sums = []
            current_sum = 0
            for a in reversed(best):
                current_sum += a
                new_sums.append(current_sum + x)
            new_sums.append(x)
            
            best.append(x)
            sums.update(new_sums)
    
    # Try replacing last element with a larger one
    if len(best) > 1:
        last = best[-1]
        for x in range(last + 1, 257):
            if time.time() > deadline:
                break
            
            # Remove last element's sums
            temp = best[:-1]
            new_sums_set = set()
            for i in range(len(temp)):
                s = 0
                for j in range(i, len(temp)):
                    s += temp[j]
                    new_sums_set.add(s)
            
            # Check if x can be appended
            valid = True
            current_sum = 0
            for a in reversed(temp):
                current_sum += a
                new_sum = current_sum + x
                if new_sum in new_sums_set:
                    valid = False
                    break
            if valid and x not in new_sums_set:
                best = temp + [x]
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_seq = []
    
    # 1. Optimized Greedies
    seq_fwd = _greedy_forward_optimized()
    if len(seq_fwd) > len(best_seq):
        best_seq = seq_fwd
        
    seq_rev = _greedy_reverse_optimized()
    if len(seq_rev) > len(best_seq):
        best_seq = seq_rev
    
    # 2. Backtracking search to explore non-greedy branches
    bt_seq = _backtracking_search(deadline, max_depth=12, branching_factor=4)
    if len(bt_seq) > len(best_seq):
        best_seq = bt_seq
    
    # 3. Randomized Restarts with efficient incremental validity check
    random.seed(123)
    
    for i in range(5000):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        selected = _randomized_trial(candidates, deadline)
        
        if len(selected) > len(best_seq):
            best_seq = selected
            
    # 4. Local polish on best sequence
    if len(best_seq) > 0:
        polished = _local_polish(best_seq, deadline)
        if len(polished) > len(best_seq):
            best_seq = polished
    
    return best_seq
