"""Structurally different approach: Backward-construction DFS with tight forbidden-value pruning.

Key structural changes:
1. Backward construction: Build the sequence from the rightmost element to the leftmost.
2. Structural insight: When prepending an element x to a sequence of length k, 
   the new block sums are exactly {x, x+a_1, x+a_1+a_2, ..., x+sum(a_1..a_k)}.
   There are exactly k+1 new sums.
3. Tight pre-filter: Before checking bitmask collisions, we check if x or any 
   x + prefix_sum is already a forbidden sum. This is O(k) per candidate and 
   eliminates most candidates before the O(1) bitmask check.
4. The backward construction has a much smaller effective branching factor because
   the last element in a valid sequence is heavily constrained (it can't be too small 
   relative to the prefix sums, or it would create collisions with existing suffix sums).
"""
import time
import sys

_DEADLINE = 85.0  # Safety margin before 90s kill


def _all_block_sums(seq):
    """Compute set of all non-empty consecutive block sums for seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums


def _bitmask_from_set(sums):
    """Convert a set of positive integers to a bitmask (Python int)."""
    mask = 0
    for s in sums:
        if s >= 0:
            mask |= (1 << s)
    return mask


def construct():
    start_time = time.time()
    
    def _time_ok():
        return time.time() < _DEADLINE
    
    best_seq = []
    best_sums_mask = 0
    best_sums_set = set()
    
    def _update_best(seq):
        nonlocal best_seq, best_sums_mask, best_sums_set
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            best_sums_set = _all_block_sums(seq)
            best_sums_mask = _bitmask_from_set(best_sums_set)
    
    # Greedy baseline: build forward greedily
    greedy_seq = []
    greedy_sums = set()
    for x in range(1, 257):
        if x in greedy_sums:
            continue
        ok = True
        s = x
        for a in reversed(greedy_seq):
            s += a
            if s in greedy_sums:
                ok = False
                break
        if not ok:
            continue
        greedy_seq.append(x)
        s = x
        greedy_sums.add(s)
        for a in reversed(greedy_seq[:-1]):
            s += a
            greedy_sums.add(s)
    if len(greedy_seq) > len(best_seq):
        _update_best(greedy_seq)
    
    # Powers of 2 baseline
    pow2_seq = [2**i for i in range(9)]
    if len(pow2_seq) > len(best_seq):
        _update_best(pow2_seq)
    
    if not _time_ok():
        return best_seq
    
    # Backward-construction DFS
    # We build the sequence from right to left.
    # State: 
    #   current_seq: list of elements chosen so far (rightmost first)
    #   current_sums_mask: bitmask of all block sums so far
    #   prefix_sums: list where prefix_sums[i] = sum of current_seq[0..i] 
    #                (i.e., cumulative sums from the right)
    #   depth: number of elements chosen so far
    
    # When we prepend x to current_seq (which has length k):
    #   New block sums: x, x + current_seq[0], x + current_seq[0] + current_seq[1], ...
    #   = x + {0, prefix_sums[0], prefix_sums[1], ..., prefix_sums[k-1]}
    #   There are k+1 new sums.
    #   We need none of these to be in current_sums_mask.
    #   Also, x must be > all elements in current_seq (since sequence is increasing left-to-right,
    #   and we're building right-to-left, x must be < current_seq[0]).
    #   Wait: the sequence is strictly increasing left-to-right. So a_1 < a_2 < ... < a_n.
    #   If we build from right to left, we first choose a_n, then a_{n-1} < a_n, etc.
    #   So when prepending x, we need x < current_seq[0] (the current rightmost element).
    #   And x >= 1.
    
    def _backward_dfs(current_seq, current_sums_mask, prefix_sums, depth):
        nonlocal best_seq, best_sums_mask, best_sums_set
        
        if not _time_ok():
            return False
        
        # Update best if we have a longer sequence
        if depth > len(best_seq):
            # current_seq is right-to-left, so reverse it for the final answer
            full_seq = list(reversed(current_seq))
            _update_best(full_seq)
        
        # Pruning: upper bound on remaining elements
        if current_seq:
            min_next = 1  # x must be >= 1
            max_next = current_seq[0] - 1  # x must be < current_seq[0]
            if max_next < min_next:
                return False
            # Maximum additional elements we can add: at most max_next - min_next + 1
            # But also constrained by the sum range
            if depth + (max_next - min_next + 1) <= len(best_seq):
                return False
        else:
            min_next = 1
            max_next = 256
        
        # Try candidate x values from max_next down to min_next
        # Heuristic: try larger x first (closer to current rightmost) to keep the sequence "tight"
        # Actually, let's try both orders and see which works better.
        # For now, try from max_next down to min_next.
        
        found_longer = False
        for x in range(max_next, min_next - 1, -1):
            if not _time_ok():
                return found_longer
            
            # Pre-filter: check if x or x + any prefix_sum is already a forbidden sum
            # New sums: x + {0, prefix_sums[0], prefix_sums[1], ..., prefix_sums[depth-1]}
            valid = True
            for ps in [0] + prefix_sums:
                s = x + ps
                if (current_sums_mask >> s) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Valid move: prepend x
            new_prefix_sums = [x]
            if prefix_sums:
                # New prefix sums: x, x + prefix_sums[0], x + prefix_sums[1], ...
                for ps in prefix_sums:
                    new_prefix_sums.append(x + ps)
            
            new_sums_mask = current_sums_mask
            for ps in [0] + prefix_sums:
                s = x + ps
                new_sums_mask |= (1 << s)
            
            current_seq.append(x)
            found_longer |= _backward_dfs(current_seq, new_sums_mask, new_prefix_sums, depth + 1)
            current_seq.pop()
            
            if not _time_ok():
                return found_longer
        
        return found_longer
    
    # Start backward DFS from empty sequence
    # But we can also start from the best_seq as a seed
    # Let's do a fresh search first
    
    _backward_dfs([], 0, [], 0)
    
    if not _time_ok():
        return best_seq
    
    # If we have time, try to extend the best sequence further
    # Convert best_seq to right-to-left order for backward construction
    if best_seq:
        rtl_seq = list(reversed(best_seq))
        rtl_prefix_sums = []
        s = 0
        for a in rtl_seq:
            s += a
            rtl_prefix_sums.append(s)
        
        best_mask = best_sums_mask
        
        def _extend_backward(current_seq, current_sums_mask, prefix_sums, depth):
            nonlocal best_seq, best_sums_mask, best_sums_set
            
            if not _time_ok():
                return False
            
            if depth > len(best_seq):
                full_seq = list(reversed(current_seq))
                _update_best(full_seq)
            
            if current_seq:
                min_next = 1
                max_next = current_seq[0] - 1
                if max_next < min_next:
                    return False
                if depth + (max_next - min_next + 1) <= len(best_seq):
                    return False
            else:
                min_next = 1
                max_next = 256
            
            found = False
            for x in range(max_next, min_next - 1, -1):
                if not _time_ok():
                    return found
                
                valid = True
                for ps in [0] + prefix_sums:
                    s = x + ps
                    if (current_sums_mask >> s) & 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                new_prefix_sums = [x]
                if prefix_sums:
                    for ps in prefix_sums:
                        new_prefix_sums.append(x + ps)
                
                new_sums_mask = current_sums_mask
                for ps in [0] + prefix_sums:
                    s = x + ps
                    new_sums_mask |= (1 << s)
                
                current_seq.append(x)
                found |= _extend_backward(current_seq, new_sums_mask, new_prefix_sums, depth + 1)
                current_seq.pop()
                
                if not _time_ok():
                    return found
            
            return found
        
        _extend_backward(rtl_seq, best_mask, rtl_prefix_sums, len(rtl_seq))
    
    if not best_seq:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return best_seq
