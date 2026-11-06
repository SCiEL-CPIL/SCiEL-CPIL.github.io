"""
Strategy: Iterative deepening DFS with strong forward checking and
a greedy baseline. The key insight is that the problem is a constraint
satisfaction problem where we need to find a long sequence such that
all consecutive block sums are distinct.

We use:
1. A fast greedy baseline to get a starting point.
2. Iterative deepening: try to find a sequence of length L, then L+1, etc.
   For each target length, we do a DFS with:
   - Forward checking: at each step, verify that the next element has
     at least one valid choice (prune dead ends early).
   - Heuristic: try smaller values first to leave more room.
   - Time limit: if we run out of time, return the best found so far.
3. We also try a "random restart" approach: try different starting values
   and different orderings to escape local optima.

The critical optimization is the forward check: when considering adding x,
we immediately check if there exists at least one valid next element y > x
such that all new sums (from appending y) don't collide. This prunes
branches that would lead to dead ends within 1-2 steps.
"""

import time
import random
import sys

_DEADLINE = 85.0  # safety margin before 90s kill
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid_full(seq):
    """Final validation check."""
    if not seq:
        return True
    if any(v < 1 or v > 256 for v in seq):
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build():
    """Fast greedy baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _compute_block_sums(seq):
    """Compute all non-empty consecutive block sums for a sequence."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _new_sums_when_appending(x, seq):
    """
    Compute the new sums that would be created by appending x to seq.
    These are: x, x+seq[-1], x+seq[-1]+seq[-2], ...
    """
    n = len(seq)
    new_sums = [x]
    s = x
    for i in range(n-1, -1, -1):
        s += seq[i]
        new_sums.append(s)
    return new_sums

def _has_valid_extension(seq, sums):
    """
    Check if there exists at least one valid next element that can be appended.
    This is the forward check for pruning.
    """
    last_val = seq[-1] if seq else 0
    for x in range(last_val + 1, 257):
        new_sums = _new_sums_when_appending(x, seq)
        # Check collision with existing sums
        if any(ns in sums for ns in new_sums):
            continue
        # Check collision among new sums themselves
        if len(set(new_sums)) != len(new_sums):
            continue
        # This x is valid, so we have at least one extension
        return True
    return False

def _get_valid_candidates(seq, sums):
    """
    Get all valid candidates for the next element.
    """
    last_val = seq[-1] if seq else 0
    candidates = []
    for x in range(last_val + 1, 257):
        new_sums = _new_sums_when_appending(x, seq)
        if any(ns in sums for ns in new_sums):
            continue
        if len(set(new_sums)) != len(new_sums):
            continue
        candidates.append(x)
    return candidates

def _dfs(seq, sums, target_len, best_seq):
    """
    DFS with forward checking. Returns the best sequence found.
    """
    if len(seq) >= target_len:
        return seq[:]
    
    if _time_remaining() < 0.5:
        return seq[:]
    
    last_val = seq[-1] if seq else 0
    candidates = _get_valid_candidates(seq, sums)
    
    if not candidates:
        return seq[:]
    
    best = seq[:]
    
    # Try candidates in increasing order (smaller values leave more room)
    for x in candidates:
        new_seq = seq + [x]
        new_sums = sums.copy()
        
        # Add new sums
        ns = _new_sums_when_appending(x, seq)
        new_sums.update(ns)
        
        # Forward check: if we haven't reached target, ensure there's a next step
        if len(new_seq) < target_len:
            if not _has_valid_extension(new_seq, new_sums):
                continue
        
        result = _dfs(new_seq, new_sums, target_len, best_seq)
        if len(result) > len(best):
            best = result[:]
            
        if len(best) >= target_len:
            return best
            
    return best

def construct():
    start_time = time.time()
    random.seed(42)
    
    # 1. Get a solid baseline with greedy
    best_seq = _greedy_build()
    best_len = len(best_seq)
    
    # 2. Try to extend the greedy sequence using DFS
    if _time_remaining() > 2.0:
        sums = _compute_block_sums(best_seq)
        for extra in range(1, 15):
            target = best_len + extra
            if _time_remaining() < 2.0:
                break
            result = _dfs(best_seq, sums, target, best_seq)
            if len(result) > best_len:
                best_seq = result
                best_len = len(best_seq)
                sums = _compute_block_sums(best_seq)
    
    # 3. Try starting from scratch with different first values
    if _time_remaining() > 5.0:
        for first_val in [1, 2, 3, 4, 5, 6, 7, 8, 10, 13]:
            if _time_remaining() < 3.0:
                break
            
            # Try to find a sequence of length best_len + 1 from scratch
            target = best_len + 1
            seq = [first_val]
            sums_init = {first_val}
            
            result = _dfs(seq, sums_init, target, best_seq)
            
            if len(result) > best_len:
                best_seq = result
                best_len = len(best_seq)
                
                # If we improved, try to extend further
                if _time_remaining() > 3.0:
                    sums_new = _compute_block_sums(best_seq)
                    for extra in range(1, 10):
                        target2 = best_len + extra
                        if _time_remaining() < 2.0:
                            break
                        result2 = _dfs(best_seq, sums_new, target2, best_seq)
                        if len(result2) > best_len:
                            best_seq = result2
                            best_len = len(best_seq)
                            sums_new = _compute_block_sums(best_seq)
    
    # 4. Random restarts: try random starting points
    if _time_remaining() > 5.0:
        for _ in range(20):
            if _time_remaining() < 2.0:
                break
            
            # Random first value
            first_val = random.randint(1, 50)
            seq = [first_val]
            sums_init = {first_val}
            
            # Try to build a long sequence
            target = best_len + 1
            result = _dfs(seq, sums_init, target, best_seq)
            
            if len(result) > best_len:
                best_seq = result
                best_len = len(best_seq)
                
                if _time_remaining() > 3.0:
                    sums_new = _compute_block_sums(best_seq)
                    for extra in range(1, 10):
                        target2 = best_len + extra
                        if _time_remaining() < 2.0:
                            break
                        result2 = _dfs(best_seq, sums_new, target2, best_seq)
                        if len(result2) > best_len:
                            best_seq = result2
                            best_len = len(best_seq)
                            sums_new = _compute_block_sums(best_seq)
    
    # Final safety check
    if not _is_valid_full(best_seq):
        best_seq = _greedy_build()
        
    return best_seq
