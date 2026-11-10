"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Fast greedy construction of a valid sequence
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    if len(values) == 256:
        return values

    best_sequence = list(values)
    best_len = len(best_sequence)

    def compute_block_sums(seq):
        """Compute all consecutive block sums for a sequence."""
        sums = []
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.append(s)
        return sums

    def is_valid(seq):
        """Check if sequence is strictly increasing and all block sums are distinct."""
        if not seq:
            return True
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        sums = compute_block_sums(seq)
        return len(sums) == len(set(sums))

    # DFS with backtracking to find longer valid sequences
    # State: (sequence, used_set, sums_set)
    # We build sequences incrementally, trying to add values in different orders

    def dfs_build(seq, used, sums_set, depth, max_depth):
        """DFS to build a valid sequence by adding values one at a time."""
        nonlocal best_sequence, best_len
        
        if time.time() > deadline:
            return
        
        if len(seq) > best_len:
            best_sequence = list(seq)
            best_len = len(seq)
        
        if len(seq) == max_depth or depth >= 12:
            return
        
        # Try adding each unused value that maintains increasing order
        last_val = seq[-1] if seq else 0
        for x in range(last_val + 1, 257):
            if x in used:
                continue
            
            # Check if adding x creates duplicate block sums
            # New blocks ending at x: x, x+seq[-1], x+seq[-1]+seq[-2], ...
            new_sums = []
            total = x
            for a in reversed(seq):
                new_sums.append(total)
                total += a
            new_sums.append(x)  # the single element x
            
            # Check for collisions
            collision = False
            for s in new_sums:
                if s in sums_set:
                    collision = True
                    break
            
            if not collision:
                # Add x
                seq.append(x)
                used.add(x)
                sums_set.update(new_sums)
                
                dfs_build(seq, used, sums_set, depth + 1, max_depth)
                
                # Backtrack
                seq.pop()
                used.discard(x)
                for s in new_sums:
                    sums_set.discard(s)
        
        # Also try inserting values in the middle (not just appending)
        # This is more expensive but can find different structures
        if depth < 8 and len(seq) < 20:
            for pos in range(1, len(seq)):
                for x in range(1, 257):
                    if x in used:
                        continue
                    if x <= seq[pos-1] or x >= seq[pos]:
                        continue
                    
                    # Check if inserting x at pos creates duplicate block sums
                    # This is complex; skip for now and focus on appending
                    pass
    
    # Try DFS with different starting points
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Try different greedy initializations
        start_val = random.randint(1, 30)
        values2, sums2 = [], set()
        for x in range(start_val, 257):
            tails, total = [x], x
            for a in reversed(values2):
                total += a
                tails.append(total)
            if not sums2.intersection(tails):
                values2.append(x)
                sums2.update(tails)
        
        if len(values2) > best_len:
            best_sequence = list(values2)
            best_len = len(values2)
        
        # Now try to extend using DFS from this base
        if best_len < 256:
            seq = list(best_sequence)
            used = set(seq)
            sums_set = set(compute_block_sums(seq))
            
            # DFS to try to extend
            dfs_build(seq, used, sums_set, 0, min(256, best_len + 15))
    
    # Final verification
    if not is_valid(best_sequence):
        # Fall back to the original greedy result
        return values
    
    return best_sequence
# EVOLVE-BLOCK-END
