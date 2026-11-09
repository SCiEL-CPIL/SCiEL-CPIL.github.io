"""Improved program for increasing sequences with unique consecutive-block sums."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _all_block_sums(seq):
    """Return set of all nonempty consecutive block sums."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _try_extend(seq, available, sums_so_far):
    """Try to extend seq by one element from available, return (new_seq, new_sums) or None."""
    n = len(seq)
    # Precompute prefix sums for efficiency
    # For a new element x appended, new block sums are:
    # x, seq[-1]+x, seq[-2]+seq[-1]+x, ..., sum(seq)+x
    total = sum(seq)
    # We need all these new sums to not be in sums_so_far
    # And x must be in available
    # And x > seq[-1] if seq is nonempty
    
    best = None
    # Try values in available that are > last element
    for x in sorted(available):
        if seq and x <= seq[-1]:
            continue
        # Check if x causes any conflict
        # New sums: x, seq[-1]+x, seq[-2]+seq[-1]+x, ...
        conflict = False
        s = x
        for i in range(n-1, -1, -1):
            if s in sums_so_far:
                conflict = True
                break
            s += seq[i]
        if not conflict:
            return x
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_seq = []
    
    # Strategy 1: Greedy smallest-first (original)
    values = []
    sums = set()
    for x in range(1, 257):
        if values and x <= values[-1]:
            continue
        # Check if adding x creates any duplicate block sum
        n = len(values)
        s = x
        conflict = False
        for i in range(n-1, -1, -1):
            if s in sums:
                conflict = True
                break
            s += values[i]
        if not conflict:
            values.append(x)
            sums.add(x)
            s = x
            for i in range(n-1, -1, -1):
                s += values[i]
                sums.add(s)
    
    if len(values) > len(best_seq):
        best_seq = values[:]
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 2: Greedy with lookahead - try to maximize future options
    # At each step, try all valid next values and pick the one that leaves most options
    def greedy_with_lookahead():
        seq = []
        sums = set()
        used = set()
        
        while True:
            if time.time() > deadline:
                return seq
            candidates = []
            start = seq[-1] + 1 if seq else 1
            for x in range(start, 257):
                if x in used:
                    continue
                # Check validity
                n = len(seq)
                s = x
                conflict = False
                for i in range(n-1, -1, -1):
                    if s in sums:
                        conflict = True
                        break
                    s += seq[i]
                if conflict:
                    continue
                candidates.append(x)
            
            if not candidates:
                break
            
            # Pick the candidate that allows the most future extensions
            # Simple heuristic: pick the smallest to keep values low
            # Or pick one that maximizes gaps
            # Let's try: pick the one that minimizes the maximum block sum
            best_x = candidates[0]
            for x in candidates:
                # Compute max block sum if we add x
                total = sum(seq) + x
                # The largest new block sum would be total
                # We want to keep total small to leave room
                # So pick smallest x
                pass
            # Just pick smallest for now, but we could be smarter
            x = candidates[0]
            seq.append(x)
            used.add(x)
            # Update sums
            s = x
            sums.add(s)
            for i in range(len(seq)-2, -1, -1):
                s += seq[i]
                sums.add(s)
        
        return seq
    
    seq2 = greedy_with_lookahead()
    if len(seq2) > len(best_seq):
        best_seq = seq2[:]
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 3: Random restarts with different heuristics
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        seq = []
        sums = set()
        used = set()
        
        while True:
            if time.time() > deadline:
                break
            candidates = []
            start = seq[-1] + 1 if seq else 1
            for x in range(start, 257):
                if x in used:
                    continue
                n = len(seq)
                s = x
                conflict = False
                for i in range(n-1, -1, -1):
                    if s in sums:
                        conflict = True
                        break
                    s += seq[i]
                if conflict:
                    continue
                candidates.append(x)
            
            if not candidates:
                break
            
            # Randomly pick from top candidates or use different heuristic
            if restart % 3 == 0:
                x = random.choice(candidates)
            elif restart % 3 == 1:
                x = candidates[-1]  # Largest
            else:
                x = candidates[len(candidates)//2]  # Middle
            
            seq.append(x)
            used.add(x)
            s = x
            sums.add(s)
            for i in range(len(seq)-2, -1, -1):
                s += seq[i]
                sums.add(s)
        
        if len(seq) > len(best_seq):
            best_seq = seq[:]
    
    return best_seq

# EVOLVE-BLOCK-END
