import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    def check_add(values, sums, x):
        """
        Check if adding x to values keeps all consecutive block sums distinct.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        # Calculate new tails
        # tails[i] is the sum of the last i+1 elements in the new sequence
        # Actually, let's be precise:
        # If values = [v1, ..., vk], and we add x.
        # New sequence: [v1, ..., vk, x]
        # New consecutive blocks ending at x:
        # x, vk+x, v(k-1)+vk+x, ..., sum(values)+x
        # We need to check if any of these are in `sums`.
        # Also, the new sums themselves must be distinct (which they are if values are strictly increasing and positive? 
        # Wait, if values are strictly increasing positive integers, the sums of consecutive blocks ending at x are strictly increasing.
        # So we only need to check against the existing `sums` set.)
        
        total = x
        # Check x
        if total in sums:
            return False, None
        new_sums = {total}
        
        for a in reversed(values):
            total += a
            if total in sums:
                return False, None
            new_sums.add(total)
        
        # If we passed the check, the new sums are distinct from old sums.
        # Are they distinct from each other?
        # Since values are strictly increasing positive integers, the partial sums from the end are strictly increasing.
        # So yes, they are distinct.
        
        return True, new_sums

    def greedy_search(shuffled_candidates):
        """
        Attempt to build a sequence using the provided order of candidates.
        """
        values = []
        sums = set()
        
        for x in shuffled_candidates:
            if x <= values[-1] if values else False:
                continue
            valid, new_sums = check_add(values, sums, x)
            if valid:
                values.append(x)
                sums.update(new_sums)
                
        return values

    best_sequence = []
    
    # Initial deterministic run to get a baseline quickly
    # The original seed logic is equivalent to greedy with sorted candidates
    # We can just use the sorted list first
    candidates = list(range(1, 257))
    best_sequence = greedy_search(candidates)
    
    # Randomized restarts
    # We will shuffle the candidates and run greedy.
    # To make it more effective, we can bias the shuffle or use different strategies.
    # Simple shuffling is a good start.
    
    while time.time() < deadline:
        random.shuffle(candidates)
        seq = greedy_search(candidates)
        if len(seq) > len(best_sequence):
            best_sequence = seq
            # If we found a significantly better one, we might want to restart the shuffle logic or just keep going
            # Note: The maximum possible length is bounded.
            # If we find a very long sequence, we might be near the optimum.
            
    return best_sequence
# EVOLVE-BLOCK-END
