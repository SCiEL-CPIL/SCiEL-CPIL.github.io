"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find a strictly increasing sequence of integers in [1, 256] such that sums of all
    nonempty consecutive blocks are distinct. Maximize length.
    
    Uses randomized greedy search with multiple restarts to explore the space.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    best_seq = []
    best_len = 0
    
    # First, do a deterministic greedy pass to get a baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_seq = list(values)
    best_len = len(values)
    
    if time.time() > deadline - 5.0:
        return best_seq
    
    # Now do randomized searches to try to find longer sequences
    # Key insight: the order in which we pick numbers matters.
    # We can try different random orderings of candidates.
    
    for attempt in range(50000):
        if time.time() > deadline - 2.0:
            break
        
        # Randomly shuffle the candidate order, but bias towards smaller values
        # since smaller values allow more room for longer sequences
        candidates = list(range(1, 257))
        
        # Use a weighted random selection: smaller numbers get higher priority
        # but we also want diversity
        if attempt % 3 == 0:
            # Nearly sorted with small perturbations
            candidates = list(range(1, 257))
            # Swap a few random pairs
            for _ in range(10):
                i, j = random.randint(0, 255), random.randint(0, 255)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        elif attempt % 3 == 1:
            # Exponentially weighted random order
            weights = [1.0 / (i + 1) for i in range(256)]
            random.shuffle(candidates)
            # Sort by random priority but biased
            priorities = [random.random() ** 0.5 for _ in range(256)]  # bias towards lower
            pairs = list(zip(candidates, priorities))
            pairs.sort(key=lambda p: p[1])
            candidates = [p[0] for p in pairs]
        else:
            # Fully random
            random.shuffle(candidates)
        
        values, sums = [], set()
        for x in candidates:
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        
        # Must be strictly increasing - sort if needed, but we need to recheck
        # Actually, the sequence must be strictly increasing, so we need to ensure that.
        # The greedy above doesn't guarantee increasing order if we shuffle.
        # Let's check if values is strictly increasing
        if all(values[i] < values[i+1] for i in range(len(values)-1)):
            if len(values) > best_len:
                best_len = len(values)
                best_seq = list(values)
                if best_len == 256:  # Can't do better
                    return best_seq
        else:
            # Try to extract the longest increasing subsequence that also satisfies the sum constraint
            # This is complex, so just skip this attempt
            pass
    
    return best_seq
# EVOLVE-BLOCK-END
