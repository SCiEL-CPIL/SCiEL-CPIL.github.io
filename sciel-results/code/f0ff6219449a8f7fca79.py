"""
Structural change: model as a Constraint Satisfaction Problem (CSP) with
forward checking, dynamic value ordering, and multi-restart branch-and-bound.
This is a different algorithmic family from the previous heuristic DFS/beam approach.

Key structural differences:
1. CSP formulation: variables are positions, domains are valid values.
2. Forward checking: prune invalid values before recursing.
3. Multi-restart with randomized value ordering to escape local optima.
4. Local search refinement on top of global search results.
5. Tighter branch-and-bound using domain size estimates.
"""
import time
import random
import sys
from collections import defaultdict

def _check_valid(seq):
    """Verify strictly increasing in [1,256] and all consecutive block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _compute_new_sums(seq, v):
    """Compute new block sums when appending v to seq."""
    new_sums = [v]
    total = v
    for i in range(len(seq) - 1, -1, -1):
        total += seq[i]
        new_sums.append(total)
    return new_sums

def _construct_powers():
    """Direct construction: powers-of-2-like gaps guarantee unique block sums."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq

def _construct_greedy_smallest():
    """Deterministic greedy smallest - fast baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _construct_greedy_largest():
    """Greedy largest - different bias, may find different optima."""
    values, sums = [], set()
    for x in range(256, 0, -1):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    values.sort()
    return values

def _get_all_sums(seq):
    """Compute all consecutive block sums of seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _solve_csp(deadline, seed=None):
    """
    CSP-style solver with forward checking and randomized value ordering.
    Different from previous DFS: uses explicit domain pruning and
    least-constraining-value heuristic.
    """
    if seed is not None:
        random.seed(seed)
    
    best_seq = []
    best_len = 0
    
    def dfs(seq, sums, last_val):
        nonlocal best_seq, best_len
        if time.time() > deadline:
            return
        
        n = len(seq)
        if n > best_len:
            best_len = n
            best_seq = list(seq)
        
        # Branch-and-bound: max possible additional elements
        max_additional = 256 - last_val
        if n + max_additional <= best_len:
            return
        
        # Generate valid candidates with forward checking
        candidates = []
        for v in range(last_val + 1, 257):
            new_sums_list = _compute_new_sums(seq, v)
            valid = True
            for s in new_sums_list:
                if s in sums:
                    valid = False
                    break
            if valid:
                candidates.append((v, new_sums_list))
        
        if not candidates:
            return
        
        # Value ordering: mix of smallest (most room) and random for diversity
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > 8:
            # Take smallest 4, then random 4 from the rest
            ordered = candidates[:4] + random.sample(candidates[4:], min(4, len(candidates) - 4))
        else:
            ordered = candidates
        
        for v, new_sums_list in ordered:
            if time.time() > deadline:
                return
            new_sums = sums.union(new_sums_list)
            dfs(seq + [v], new_sums, v)
    
    dfs([], set(), 0)
    return best_seq

def _iterative_deepening_from_prefix(prefix, target_len, deadline, seed=None):
    """
    Iterative deepening: try to extend a given prefix to reach target_len.
    Uses beam-like search with pruning to find valid extensions.
    """
    if seed is not None:
        random.seed(seed)
    
    best = list(prefix)
    best_len = len(prefix)
    
    if best_len >= target_len:
        return best
    
    # Beam search from the prefix
    beam = [(prefix, set(_get_all_sums(prefix)))]
    
    for depth in range(len(prefix), target_len):
        if time.time() > deadline:
            break
        
        new_beam = []
        for seq, sums in beam:
            last_val = seq[-1] if seq else 0
            for v in range(last_val + 1, 257):
                new_sums_list = _compute_new_sums(seq, v)
                valid = True
                for s in new_sums_list:
                    if s in sums:
                        valid = False
                        break
                if valid:
                    new_beam.append((seq + [v], sums.union(new_sums_list)))
        
        if not new_beam:
            break
        
        # Keep only the most promising candidates (smallest last value = most room)
        new_beam.sort(key=lambda x: x[0][-1])
        beam = new_beam[:50]  # Beam width
        
        # Update best if we found a longer valid sequence
        if len(new_beam) > 0 and len(new_beam[0][0]) > best_len:
            best = list(new_beam[0][0])
            best_len = len(best)
        
        if len(beam) == 0:
            break
    
    return best

def _solve_csp_multi_restart(deadline, n_restarts=8):
    """
    Multi-restart CSP solver with different random seeds.
    Each restart explores a different region of the search space.
    """
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    remaining_time = deadline - time.time()
    if remaining_time <= 1:
        return best_seq
    
    per_restart = remaining_time / max(n_restarts, 1)
    
    for r in range(n_restarts):
        if time.time() > deadline - 2:
            break
        restart_deadline = min(deadline - 2, time.time() + per_restart)
        try:
            result = _solve_csp(restart_deadline, seed=r * 12345 + 678)
            if len(result) > best_len:
                best_seq = result
                best_len = len(result)
        except Exception:
            pass
    
    return best_seq

def _local_search_refine(seq, deadline):
    """
    Local search: try to extend the sequence by finding valid next elements.
    Also try alternative endings to unlock further extensions.
    Different from global search: operates on a near-complete sequence.
    """
    if not seq:
        return seq
    
    best = list(seq)
    best_len = len(best)
    
    # Try to extend greedily
    while time.time() < deadline - 1:
        extended = False
        for v in range(best[-1] + 1, 257):
            candidate = best + [v]
            if _check_valid(candidate):
                best = candidate
                best_len = len(best)
                extended = True
                break
        if not extended:
            break
    
    # Try replacing the last element with a different one to unlock extensions
    for pos in range(max(0, len(best) - 3), len(best)):
        if time.time() > deadline - 1:
            break
        original = best[pos:]
        for v in range((best[pos-1] + 1) if pos > 0 else 1, 257):
            candidate = best[:pos] + [v]
            if not _check_valid(candidate):
                continue
            # Try to extend this candidate
            extended_candidate = list(candidate)
            while time.time() < deadline - 1:
                ext_found = False
                for w in range(extended_candidate[-1] + 1, 257):
                    test = extended_candidate + [w]
                    if _check_valid(test):
                        extended_candidate = test
                        ext_found = True
                        break
                if not ext_found:
                    break
            if len(extended_candidate) > best_len:
                best = extended_candidate
                best_len = len(best)
    
    return best

def _pivot_search(base_seq, deadline, seed=None):
    """
    Pivot search: try replacing one or two middle elements to unlock longer sequences.
    The greedy approach gets stuck because early choices constrain later options.
    By trying alternative values at critical positions, we may find paths to longer sequences.
    """
    if seed is not None:
        random.seed(seed)
    
    best = list(base_seq)
    best_len = len(base_seq)
    n = len(base_seq)
    
    if n < 3:
        return best
    
    # Try replacing each position with alternative values
    for pos in range(1, n - 1):
        if time.time() > deadline - 2:
            break
        
        # Get the range of valid replacements
        lo = base_seq[pos-1] + 1
        hi = min(base_seq[pos+1] - 1, 256) if pos + 1 < n else 256
        
        # Try a subset of alternatives (randomized for diversity)
        candidates = list(range(lo, hi + 1))
        if len(candidates) > 10:
            candidates = [base_seq[pos]] + random.sample(candidates, 10)
        
        for v in candidates:
            if v == base_seq[pos]:
                continue
            if time.time() > deadline - 1:
                break
            
            # Create modified sequence
            candidate = base_seq[:pos] + [v] + base_seq[pos+1:]
            
            # Check if still valid up to this point
            if not _check_valid(candidate):
                continue
            
            # Try to extend from here
            extended = list(candidate)
            sums = _get_all_sums(extended)
            
            # Greedily extend
            while time.time() < deadline - 1:
                ext_found = False
                for w in range(extended[-1] + 1, 257):
                    new_sums_list = _compute_new_sums(extended, w)
                    valid = True
                    for s in new_sums_list:
                        if s in sums:
                            valid = False
                            break
                    if valid:
                        extended.append(w)
                        sums.update(new_sums_list)
                        ext_found = True
                        break
                if not ext_found:
                    break
            
            if len(extended) > best_len:
                best = extended
                best_len = len(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Multiple baselines
    best = _construct_greedy_smallest()
    best_len = len(best)
    
    try:
        greedy_large = _construct_greedy_largest()
        if len(greedy_large) > best_len:
            best = greedy_large
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Multi-restart CSP solver (structural change: different algorithmic family)
    try:
        csp_result = _solve_csp_multi_restart(deadline, n_restarts=6)
        if len(csp_result) > best_len:
            best = csp_result
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Pivot search: try replacing middle elements to unlock longer sequences
    try:
        pivot_result = _pivot_search(best, deadline, seed=42)
        if len(pivot_result) > best_len:
            best = pivot_result
            best_len = len(best)
    except Exception:
        pass
    
    # Local search refinement on the best sequence found
    try:
        refined = _local_search_refine(best, deadline)
        if len(refined) > best_len:
            best = refined
            best_len = len(best)
    except Exception:
        pass
    
    # Final verification
    if not _check_valid(best):
        best = _construct_powers()
    
    return best
