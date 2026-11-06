import time
import random

def _is_valid(seq):
    """Check if the sequence has unique block products."""
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _greedy_construct():
    """Greedy baseline."""
    values = []
    for x in range(2, 129):
        # Try inserting x at the end (since we iterate 2..128, it's always sorted)
        candidate = values + [x]
        if _is_valid(candidate):
            values = candidate
    return values

def _try_insert(seq, x):
    """Try to insert x into seq maintaining sorted order. Return new seq if valid, else None."""
    # Find insertion point
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    insert_idx = lo
    
    # If x is already in seq, skip (though we filter candidates)
    if insert_idx < len(seq) and seq[insert_idx] == x:
        return None

    new_seq = seq[:insert_idx] + [x] + seq[insert_idx:]
    if _is_valid(new_seq):
        return new_seq
    return None

def _try_swap(seq, i, x):
    """Try to replace seq[i] with x. Maintain sorted order."""
    # Check sorted order constraints
    if i > 0 and x <= seq[i-1]:
        return None
    if i < len(seq) - 1 and x >= seq[i+1]:
        return None
    if x == seq[i]:
        return None
    
    new_seq = seq[:i] + [x] + seq[i+1:]
    if _is_valid(new_seq):
        return new_seq
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin
    
    # 1. Greedy Initialization
    best_seq = _greedy_construct()
    best_len = len(best_seq)
    
    # 2. Local Search
    # We will try to improve best_seq.
    # Since we want to maximize length, we prioritize insertions.
    # If no insertions, we try swaps to potentially enable insertions.
    
    while time.time() < deadline:
        improved = False
        candidates = [x for x in range(2, 129) if x not in best_seq]
        
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        # Try all insertions
        for x in candidates:
            if time.time() > deadline:
                break
            new_seq = _try_insert(best_seq, x)
            if new_seq is not None:
                best_seq = new_seq
                best_len = len(best_seq)
                improved = True
                # Restart insertion loop with new best_seq
                candidates = [x for x in range(2, 129) if x not in best_seq]
                random.shuffle(candidates)
                break # Break to restart candidate generation
        
        if time.time() > deadline:
            break
            
        if not improved:
            # No insertions found. Try swaps.
            # We only care about swaps if they lead to a longer sequence later, 
            # but since we don't know the future, we just explore the space.
            # However, swaps don't increase length. 
            # A more effective move when stuck is "Remove 1, Insert 2" (net +1).
            # Let's try a simpler heuristic: Randomized restarts or deeper search?
            # Given the time, let's try swaps to change the configuration.
            
            swap_found = False
            # Try a limited number of swaps to avoid getting stuck in a loop
            # If we just keep swapping, we might cycle. 
            # To mitigate this, we can keep a history of visited states (hash of tuple).
            # But state space is large. 
            # Alternative: If no insertion, try to remove a "bad" element and see if we can insert two?
            
            # Let's try a "2-opt" style: Remove one, try to insert two.
            # This is expensive. 
            # Let's stick to swaps but limit them to avoid infinite loops.
            # Actually, if no insertion is possible, the sequence is "saturated".
            # Swapping might open a slot.
            
            for i in range(len(best_seq)):
                if time.time() > deadline:
                    break
                # Try swapping with random candidates
                # Limit candidates to a few to save time
                sample_candidates = random.sample(candidates, min(10, len(candidates))) if candidates else []
                for x in sample_candidates:
                    new_seq = _try_swap(best_seq, i, x)
                    if new_seq is not None:
                        # Accept swap? 
                        # Swapping doesn't increase length. 
                        # We should only accept if it seems "better" or random.
                        # Since we want to maximize length, a swap is only useful if it enables an insertion.
                        # We can't know that immediately.
                        # Let's accept a swap with low probability to explore, 
                        # or just accept it and see if the next loop finds an insertion.
                        best_seq = new_seq
                        swap_found = True
                        improved = True # Treat as progress to restart loops
                        break
                if improved:
                    break
            
            if not improved:
                # If no swaps found or no candidates, we might be stuck.
                # Try a random removal and then re-insertion?
                # Or just break if we've been stuck for a while.
                # For now, if no move is found, break to avoid wasting time.
                break

    return best_seq
