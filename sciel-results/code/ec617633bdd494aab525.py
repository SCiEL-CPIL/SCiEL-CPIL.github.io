"""
Program for large integer sets avoiding equal-gcd triples in [1,128].
Strategy: 
1. Greedy construction using prime signatures to get a large initial set.
2. Local search by removing the element with the highest "conflict count" (number of bad triples it participates in).
3. Repeat with different random orderings and restarts.
"""

def construct():
    import time
    import random
    from math import gcd
    from itertools import combinations

    DEADLINE = 85.0
    start_time = time.time()

    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            gcd_mat[i][j] = gcd(i + 1, j + 1)

    # Precompute all "bad triples"
    # A triple (i, j, k) is bad if gcd(i,j) == gcd(i,k) == gcd(j,k)
    # We store these as tuples of indices (0-indexed)
    bad_triples = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                if gcd_mat[i][j] == gcd_mat[i][k] == gcd_mat[j][k]:
                    bad_triples.append((i, j, k))

    # Map element index to list of bad triples it belongs to
    # This allows fast conflict counting
    element_to_triples = [[] for _ in range(N)]
    for t in bad_triples:
        for idx in t:
            element_to_triples[idx].append(t)

    def get_conflict_count(mask, idx):
        """
        Count how many bad triples involving 'idx' are fully present in 'mask'.
        """
        count = 0
        for t in element_to_triples[idx]:
            # Check if all 3 elements of t are in mask
            # t is a tuple of 3 indices
            if (mask >> t[0]) & 1 and (mask >> t[1]) & 1 and (mask >> t[2]) & 1:
                count += 1
        return count

    def is_valid_mask(mask):
        """
        Check if mask represents a valid set (no bad triples).
        """
        for t in bad_triples:
            if (mask >> t[0]) & 1 and (mask >> t[1]) & 1 and (mask >> t[2]) & 1:
                return False
        return True

    def mask_to_list(mask):
        res = []
        for i in range(N):
            if (mask >> i) & 1:
                res.append(i + 1)
        return res

    rng = random.Random(42)
    best_size = 0
    best_mask = 0

    # Strategy 1: Greedy Removal
    # Start with all elements, remove the one causing the most conflicts until valid.
    # To improve, we can try different random perturbations or orderings.
    
    # We will run multiple iterations of:
    # 1. Start with a random subset (or full set)
    # 2. While invalid: remove element with max conflict count (tie-break randomly)
    # 3. Try to add back elements (local search)
    
    # To maximize size, starting with the full set and removing minimally is a good baseline.
    # However, simple greedy removal might get stuck in a local minimum of "removals".
    # We can use "swap" moves: remove one, try to add others.
    
    # Let's implement a more robust local search:
    # State: mask
    # Move: 
    #   - If invalid: remove one element from a bad triple (prefer one in many bad triples)
    #   - If valid: try to add a random element. If it creates a conflict, don't add.
    #     If it doesn't, add it.
    #   - Also try: remove one element, then greedily add as many as possible.

    def improve_mask(start_mask):
        """
        Take a start_mask (possibly invalid) and improve it to a valid, larger set.
        """
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        
        # Phase 1: Remove elements to make valid
        # We track conflict counts dynamically or recompute?
        # Recomputing for all elements is O(N * T). T is number of bad triples.
        # T is roughly N^3 / something. N=128. T might be a few thousand.
        # Let's just recompute the "most conflicting" element each step.
        
        while not is_valid_mask(current_mask):
            # Find element with max conflict count
            max_conf = -1
            candidates = []
            for i in range(N):
                if (current_mask >> i) & 1:
                    c = get_conflict_count(current_mask, i)
                    if c > max_conf:
                        max_conf = c
                        candidates = [i]
                    elif c == max_conf and c > 0:
                        candidates.append(i)
            
            if not candidates:
                break # Should not happen if invalid
            
            # Pick one randomly
            to_remove = rng.choice(candidates)
            current_mask &= ~(1 << to_remove)
            current_size -= 1
            
            if time.time() > DEADLINE - 5:
                return current_mask, current_size

        # Phase 2: Local Search to add more elements
        # We try to add elements that are not in the mask.
        # We can do a greedy pass: try to add each missing element if possible.
        # To escape local optima, we can also try "swap" moves.
        
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            
            # Try adding a random missing element
            missing = [i for i in range(N) if not ((current_mask >> i) & 1)]
            if not missing:
                break
            
            rng.shuffle(missing)
            for i in missing:
                new_mask = current_mask | (1 << i)
                if is_valid_mask(new_mask):
                    current_mask = new_mask
                    current_size += 1
                    improved = True
                    break # Restart the loop to try adding more
            
            # If no single add worked, try a swap: remove one, add another
            if not improved:
                # Pick a random element to remove
                present = [i for i in range(N) if (current_mask >> i) & 1]
                if not present:
                    break
                to_remove = rng.choice(present)
                temp_mask = current_mask & ~(1 << to_remove)
                
                # Try to add any missing element to temp_mask
                missing = [i for i in range(N) if not ((temp_mask >> i) & 1) and i != to_remove]
                rng.shuffle(missing)
                for i in missing:
                    new_mask = temp_mask | (1 << i)
                    if is_valid_mask(new_mask):
                        # This swap maintains size, but might allow further additions later?
                        # Actually, just swapping doesn't increase size.
                        # We should only accept if it leads to a larger set eventually?
                        # For now, let's stick to pure additions for simplicity in this phase,
                        # or accept swap if it allows a subsequent addition?
                        # Let's try: accept swap, then try to add more.
                        current_mask = new_mask
                        # current_size stays same
                        improved = True
                        break

        return current_mask, current_size

    # Run multiple restarts
    # Start with full set, then random subsets of various sizes
    
    # Initial best: Greedy removal from full set
    full_mask = (1 << N) - 1
    m, s = improve_mask(full_mask)
    if s > best_size:
        best_size = s
        best_mask = m

    # Try starting from random subsets
    # Larger subsets are harder to fix, but might have higher potential?
    # Smaller subsets are easier to extend.
    # Let's try sizes: N, N-10, N-20, N-50, N-80
    
    for _ in range(50): # Restart limit
        if time.time() > DEADLINE - 10:
            break
            
        # Random subset size
        # Bias towards larger sizes initially, then smaller?
        # Let's just pick a random size between N//2 and N
        size = rng.randint(N // 2, N)
        subset = rng.sample(range(N), size)
        start_mask = 0
        for i in subset:
            start_mask |= (1 << i)
        
        m, s = improve_mask(start_mask)
        if s > best_size:
            best_size = s
            best_mask = m

    # One more specific strategy: 
    # Start with a known good structure? 
    # E.g., numbers with distinct prime factors? 
    # Or just rely on the search above.
    
    # Convert best mask to list
    result = mask_to_list(best_mask)
    
    # Fallback
    if not result:
        result = [1]
        
    # Final Validation
    # is_valid_mask uses bad_triples, which is exhaustive.
    # If best_mask is valid, result is valid.
    # But just in case, do a quick check on the list itself?
    # is_valid_mask is fast enough for one call.
    if not is_valid_mask(best_mask):
         # This shouldn't happen if improve_mask works correctly
         # But if it does, fallback
         result = [1, 2, 4, 8, 16, 32, 64, 128]
         # Check fallback
         fb_mask = 0
         for v in result:
             fb_mask |= (1 << (v-1))
         if not is_valid_mask(fb_mask):
             result = [1]

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Final size: {len(res)}")
    print(res)
