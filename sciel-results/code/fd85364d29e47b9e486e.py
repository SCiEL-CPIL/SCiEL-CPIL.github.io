"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def is_valid_set(S):
    """Check if the set S satisfies the conditions."""
    if not S:
        return True
    
    # Generate all non-empty subset sums
    sums = set()
    sums.add(0)
    for x in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums.update(new_sums)
    
    sums.discard(0)
    
    # Check for duplicate sums is implicitly handled by using a set, 
    # but the problem states "Equal sum values are counted once. 
    # Two different subsets with the same sum still make the set invalid".
    # So we must check if the number of subset sums equals 2^k - 1.
    k = len(S)
    if len(sums) != (1 << k) - 1:
        return False
        
    # Check divisibility
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        si = sorted_sums[i]
        for j in range(i + 1, len(sorted_sums)):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    # Candidates from 1 to 256
    candidates = list(range(1, 257))
    
    # DFS to find largest valid set
    # We sort candidates to try smaller ones first, as they allow more room?
    # Or larger ones to avoid divisibility?
    # Smaller numbers are more likely to divide larger sums.
    # So maybe trying larger numbers first is better?
    # But we want to maximize count, so we need small numbers to fit more.
    # Let's try a greedy approach first to get a baseline, then DFS around it.
    
    # Greedy: add smallest valid number
    current = []
    for x in candidates:
        if is_valid_set(current + [x]):
            current.append(x)
            if len(current) > best_size:
                best_set = current[:]
                best_size = len(current)
                
    # If greedy didn't find size > 3, try DFS
    # DFS with pruning
    def dfs(current_set, start_idx):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)
            
        # Pruning: if we can't beat best_size with remaining candidates
        # Max possible size is best_size
        # Remaining candidates: candidates[start_idx:]
        # This bound is weak, but okay.
        
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            # Optional: if x is too large to be useful?
            # Check validity
            if is_valid_set(current_set + [x]):
                dfs(current_set + [x], i + 1)
            # If we just want one path, break? No, we want max size.
            # But DFS on 256 candidates is too slow.
            # We need a better heuristic.
            # Let's limit the depth of DFS or the number of branches.
            # Given the time, a simple greedy with random restarts might be better.
            
    # Random restarts
    import random
    attempts = 0
    while time.time() < deadline and attempts < 1000:
        attempts += 1
        # Shuffle candidates? Or pick a random subset to start?
        # Let's try random permutations of a subset of candidates
        # Or just random greedy: pick random x, if valid, add.
        current = []
        # Use a random order
        rand_cands = candidates[:]
        random.shuffle(rand_cands)
        for x in rand_cands:
            if is_valid_set(current + [x]):
                current.append(x)
                if len(current) > best_size:
                    best_set = current[:]
                    best_size = len(current)
                    
    return best_set

# EVOLVE-BLOCK-END
