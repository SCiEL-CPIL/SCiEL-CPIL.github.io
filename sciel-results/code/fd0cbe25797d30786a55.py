import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _compute_sums(values):
    """Compute the set of all consecutive block sums for a sequence."""
    sums = set()
    n = len(values)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            sums.add(total)
    return sums

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = _compute_sums(current)
    
    while True:
        if not current:
            start = 1
        else:
            start = current[-1] + 1
        
        found = False
        for x in range(start, max_val + 1):
            # Check if x creates a collision
            valid = True
            total = x
            if total in cur_sums:
                valid = False
            if valid:
                # Check all tails
                tail_sum = x
                for a in reversed(current):
                    tail_sum += a
                    if tail_sum in cur_sums:
                        valid = False
                        break
            
            if valid:
                # Update sums
                cur_sums.add(x)
                tail_sum = x
                for a in reversed(current):
                    tail_sum += a
                    cur_sums.add(tail_sum)
                current.append(x)
                found = True
                break
        
        if not found:
            break
    
    return current

def _dfs_csp(max_val=256, deadline=None, initial_seq=None, initial_sums=None):
    """
    DFS with Constraint Propagation (Forward Checking).
    """
    if deadline is None:
        deadline = time.time() + 50
        
    best_seq = []
    best_len = 0
    
    if initial_seq is None:
        seq = []
        sums = set()
        last = 0
    else:
        seq = list(initial_seq)
        sums = set(initial_sums)
        last = seq[-1] if seq else 0
        
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = list(seq)
        
    sys.setrecursionlimit(10000)
    
    def dfs(seq, sums, last):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return
            
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
            
        if last >= max_val:
            return
            
        for x in range(last + 1, max_val + 1):
            # Quick check: x itself
            if x in sums:
                continue
                
            # Compute tails
            tails = [x]
            total = x
            valid = True
            for a in reversed(seq):
                total += a
                tails.append(total)
                if total in sums:
                    valid = False
                    break
            
            if valid:
                new_sums = set(sums)
                new_sums.update(tails)
                new_seq = seq + [x]
                dfs(new_seq, new_sums, x)
                
                # Pruning: if we found a very long sequence, we might prune?
                # No, keep it simple.
                
    dfs(seq, sums, last)
    
    return best_seq, best_len

def _beam_search(max_val=256, deadline=None, beam_width=20):
    """
    Beam search to find good candidates.
    State: (seq, sums)
    Score: length of seq (primary), then negative sum of values (secondary, prefer small sums)
    """
    if deadline is None:
        deadline = time.time() + 50
        
    # Initial beam: empty sequence
    beam = [([], set(), 0)]
    
    best_seq = []
    best_len = 0
    
    # Expand until no progress or deadline
    while time.time() < deadline:
        new_beam = []
        
        for seq, sums, last in beam:
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = list(seq)
                
            if last >= max_val:
                continue
                
            # Generate candidates
            candidates = []
            for x in range(last + 1, max_val + 1):
                if x in sums:
                    continue
                    
                tails = [x]
                total = x
                valid = True
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                    if total in sums:
                        valid = False
                        break
                        
                if valid:
                    new_sums = set(sums)
                    new_sums.update(tails)
                    new_seq = seq + [x]
                    # Score: length, then -sum (prefer small sums)
                    score = (len(new_seq), -sum(new_seq))
                    candidates.append((score, new_seq, new_sums, x))
            
            # Add to new beam
            new_beam.extend(candidates)
        
        if not new_beam:
            break
            
        # Sort by score (descending) and keep top beam_width
        new_beam.sort(key=lambda x: x[0], reverse=True)
        new_beam = new_beam[:beam_width]
        
        # Update beam
        beam = [(seq, sums, last) for _, seq, sums, last in new_beam]
        
        # If we can't expand further (all beams are stuck or max_val reached)
        if len(beam) == 0:
            break
            
        # Early termination if we've hit a theoretical limit? 
        # Not really, but if beam is stuck at max length for a while?
        # Let's just run until deadline.
        
    return best_seq, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    best = []
    best_len = 0
    
    # 1. Get a good initial lower bound using Greedy
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
        
    # 2. Run Beam Search
    # Beam search is often better than pure greedy for finding longer sequences
    # because it maintains multiple promising paths.
    beam_seq, beam_len = _beam_search(max_val=256, deadline=deadline - 20, beam_width=50)
    if beam_len > best_len:
        best = beam_seq
        best_len = beam_len
        
    # 3. Run DFS CSP from the best known state
    # This can sometimes find a slightly longer sequence by backtracking
    # from a strong initial guess.
    if time.time() < deadline - 5:
        initial_sums = _compute_sums(best)
        dfs_seq, dfs_len = _dfs_csp(max_val=256, deadline=deadline, initial_seq=best, initial_sums=initial_sums)
        if dfs_len > best_len:
            best = dfs_seq
            best_len = dfs_len
    
    # 4. If time remains, try a fresh DFS from empty
    if time.time() < deadline - 10:
        dfs_seq2, dfs_len2 = _dfs_csp(max_val=256, deadline=deadline)
        if dfs_len2 > best_len:
            best = dfs_seq2
            best_len = dfs_len2

    # Final validation
    if not _is_valid_full(best):
        # Fallback
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
