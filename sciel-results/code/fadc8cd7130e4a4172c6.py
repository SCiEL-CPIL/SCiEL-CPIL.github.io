import time

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
        # New blocks are those that include x
        # These are: suffix of selected[:idx] ending at idx-1, extended by x and prefix of selected[idx:]
        # Actually, new blocks are all consecutive blocks that contain x
        # A block containing x starts at some i <= idx and ends at some j >= idx (in the new array)
        # In the old array, these correspond to:
        #   - Blocks [i, idx-1] + x + [idx, j] where i is in [max(0, idx-?)] and j is in [min(n-1, idx+?)]
        # More precisely: new block = selected[i:idx] + [x] + selected[idx:j]
        # The sum = sum(selected[i:idx]) + x + sum(selected[idx:j])
        
        # We need to check all such blocks against existing sums
        # And also check that no two new blocks have the same sum
        
        n = len(selected)
        new_sums = set()
        valid = True
        
        # Prefix sums from left: sum of selected[i:idx] for i from idx-1 down to 0
        left_sums = [0]  # left_sums[k] = sum of last k elements of selected[:idx]
        for k in range(1, idx + 1):
            left_sums.append(left_sums[-1] + selected[idx - k])
        
        # Suffix sums from right: sum of selected[idx:j] for j from idx to n-1
        right_sums = [0]  # right_sums[k] = sum of first k elements of selected[idx:]
        for k in range(1, n - idx + 1):
            right_sums.append(right_sums[-1] + selected[idx + k - 1])
        
        # Now check all new blocks
        for i in range(idx + 1):  # i is number of elements from left (0 to idx)
            for j in range(n - idx + 1):  # j is number of elements from right (0 to n-idx)
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
        
        # Insert x
        selected.insert(idx, x)
        sums.update(new_sums)
    
    return selected

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
        
    # 2. Randomized Restarts with efficient incremental validity check
    import random
    random.seed(123)
    
    for i in range(10000):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        selected = _randomized_trial(candidates, deadline)
        
        if len(selected) > len(best_seq):
            best_seq = selected
            
    return best_seq
