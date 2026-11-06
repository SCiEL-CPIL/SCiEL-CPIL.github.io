"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit (5s buffer)
    
    # 1. Generate a strong baseline using forward greedy
    seq = _forward_greedy(1, 256)
    best_seq = seq
    best_len = len(seq)
    
    # 2. Attempt to improve via randomized local search with insertions
    #    and a more targeted backtracking that tries to "repair" the greedy
    #    choice by looking ahead when a greedy choice leads to a shorter
    #    sequence than a slightly larger alternative.
    
    # Time budget for search
    while time.time() < deadline:
        # Strategy A: Try to insert a new element into best_seq
        # This is the most direct way to increase length.
        # We try all possible positions and a subset of values.
        improved = False
        current_len = len(best_seq)
        
        # To avoid O(N * 256 * check_cost) being too slow, we will
        # only try insertions if we haven't already exhausted small
        # gaps. But since values are in [1,256] and strictly increasing,
        # the number of possible insertions is limited.
        
        # Let's try a few random insertion attempts.
        # Actually, a systematic sweep is better.
        # For each position i (0..L), find the valid range of x.
        # Try a few x values in that range.
        
        # Optimization: We can pre-check which x values are "likely"
        # by checking if they create unique tails.
        
        for pos in range(current_len + 1):
            if time.time() > deadline:
                break
            
            low = best_seq[pos-1] + 1 if pos > 0 else 1
            high = best_seq[pos] - 1 if pos < current_len else 256
            
            if low > high:
                continue
            
            # Try a small number of candidates in [low, high]
            # If the range is small, try all. If large, try a few random + min/max.
            candidates = []
            if high - low < 10:
                candidates = range(low, high + 1)
            else:
                candidates = [low, low+1, high-1, high]
                # Add a few randoms
                for _ in range(5):
                    candidates.append(random.randint(low, high))
            
            for x in candidates:
                if time.time() > deadline:
                    break
                
                new_seq = best_seq[:pos] + [x] + best_seq[pos:]
                if _is_valid(new_seq):
                    best_seq = new_seq
                    best_len += 1
                    improved = True
                    break # Restart insertion search from new length
            
            if improved:
                break
        
        if improved:
            continue
        
        # Strategy B: If no insertion found, try to modify existing elements
        # to create room for an insertion.
        # This is harder. We will try to "shift" an element to a nearby value
        # to free up a slot.
        
        # For now, let's try a randomized restart with a slightly different
        # greedy heuristic: sometimes picking a slightly larger number
        # early allows for more numbers later.
        
        # We will implement a "perturbed greedy" that at each step,
        # with probability p, picks the k-th smallest valid number
        # instead of the smallest.
        
        # But this is expensive to restart from scratch.
        # Let's stick to local search for now.
        
        # If we are stuck, break out of the loop to avoid wasting time.
        # However, we might want to try a few more random swaps.
        
        # Let's try to swap two elements to see if it allows an insertion.
        # This is complex.
        
        # Instead, let's try to run a short backtracking search
        # starting from the current best_seq prefix, to see if we
        # can extend it by one more element.
        
        # This is equivalent to the insertion search above, but
        # backtracking might find a different prefix that allows
        # a longer total sequence.
        
        # Given the time constraints and the strength of the greedy,
        # if we can't insert, we are likely near the optimum.
        
        # Let's break if we've been trying for a while without improvement.
        # Actually, the loop condition handles the time.
        
        # To prevent infinite loops of failing insertions, we add a
        # counter or just break after a few failed attempts.
        # But the insertion search is systematic, so if it fails once,
        # it will fail again unless best_seq changes.
        # So we break if no improvement in a full sweep.
        
        break
    
    return best_seq

def _is_valid(seq):
    """Check if sequence is strictly increasing and all consecutive block sums are unique."""
    if not seq:
        return True
    
    # Check strictly increasing
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i+1]:
            return False
    
    # Check unique block sums
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

def _forward_greedy(start, end):
    """Standard greedy: pick smallest valid number."""
    values = []
    sums = set()
    
    for x in range(start, end + 1):
        # Calculate new block sums ending at x
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
            
    return values
# EVOLVE-BLOCK-END
