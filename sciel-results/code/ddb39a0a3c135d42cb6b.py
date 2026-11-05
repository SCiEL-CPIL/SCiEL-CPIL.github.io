"""Sidon set extension in [1, 284] containing {1,2,5,11}."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    seed_elements = [1, 2, 5, 11]
    seed_set = set(seed_elements)
    max_val = 284
    max_sum = 2 * max_val
    
    # Precompute all candidates
    all_candidates = list(range(1, max_val + 1))

    def compute_sums_bitset(a):
        """Compute bitset of all pairwise sums a[i]+a[j] for i<=j."""
        bits = 0
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i, n):
                bits |= 1 << (ai + a[j])
        return bits

    def can_add(x, a, sums_bits):
        """Check if x can be added to a without creating duplicate sums."""
        # Check if 2x conflicts
        if (sums_bits >> (2 * x)) & 1:
            return False
        # Check if x+y conflicts for any y in a
        for y in a:
            s = x + y
            if (sums_bits >> s) & 1:
                return False
        return True

    def add_element(x, a, sums_bits):
        """Add x to a and update sums_bits. Returns (a, sums_bits)."""
        a.append(x)
        a.sort()
        sums_bits |= 1 << (2 * x)
        for y in a:
            if y != x:
                sums_bits |= 1 << (x + y)
        return a, sums_bits

    def greedy_extend(a, sums_bits, order=None):
        """Greedy extension. If order is None, uses increasing order.
        Returns (a, sums_bits)."""
        a = list(a)
        sums_bits = sums_bits
        if order is None:
            order = list(range(1, max_val + 1))
        
        a_set = set(a)
        for x in order:
            if x in a_set:
                continue
            if can_add(x, a, sums_bits):
                a, sums_bits = add_element(x, a, sums_bits)
                a_set.add(x)
        return a, sums_bits

    def local_search(a, sums_bits, max_iters=200, deadline=None):
        """Local search: remove one non-seed, try adding a random candidate, then greedy extend.
        Returns best_a found."""
        a = list(a)
        sums_bits = sums_bits
        best_a = list(a)
        best_bits = sums_bits
        best_size = len(a)
        
        candidates = [x for x in range(1, max_val + 1) if x not in a]
        
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            if len(a) <= len(seed_elements):
                break
            
            # Pick a random non-seed element to remove
            removable = [x for x in a if x not in seed_set]
            if not removable:
                break
                
            x_remove = random.choice(removable)
            
            # Remove x_remove from a
            a.remove(x_remove)
            
            # Recompute sums (fast for small n)
            new_sums_bits = compute_sums_bitset(a)
            
            # Try to add a random candidate
            improved = False
            # Try a few random additions
            for _ in range(15):
                if not candidates:
                    break
                x_add = random.choice(candidates)
                if x_add in a:
                    continue
                if can_add(x_add, a, new_sums_bits):
                    a, new_sums_bits = add_element(x_add, a, new_sums_bits)
                    
                    # Greedy extend
                    a, new_sums_bits = greedy_extend(a, new_sums_bits)
                    
                    if len(a) > best_size:
                        best_a = list(a)
                        best_bits = new_sums_bits
                        best_size = len(a)
                    improved = True
                    break
            
            if not improved:
                # Restore x_remove
                a.append(x_remove)
                a.sort()
                new_sums_bits = compute_sums_bitset(a)
                # If we reverted, we should probably reset a to the best known state 
                # or just continue from the reverted state. 
                # Reverting to the state before removal is safer to avoid drift to smaller sets.
                # But we lost the original a state. Let's just continue from the reverted state.
                
        return best_a, best_bits

    # Initial greedy
    best_a = list(seed_elements)
    best_bits = compute_sums_bitset(best_a)
    best_a, best_bits = greedy_extend(best_a, best_bits)
    
    random.seed(42)
    
    # Multiple restarts with local search
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Randomized greedy start
        a = list(seed_elements)
        bits = compute_sums_bitset(a)
        
        # Shuffle candidates for greedy
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        # Sometimes use reverse or other orders
        if r % 5 == 0:
            candidates = list(range(max_val, 0, -1))
        elif r % 5 == 1:
            candidates = list(range(1, max_val + 1)) # increasing
        # else: already shuffled
        
        a, bits = greedy_extend(a, bits, order=candidates)
        
        if len(a) > len(best_a):
            best_a = a
            best_bits = bits
            
        # Local search
        a_ls, bits_ls = local_search(a, bits, max_iters=100, deadline=deadline - 5)
        
        if len(a_ls) > len(best_a):
            best_a = a_ls
            best_bits = bits_ls
            
        # If we've reached size 20, we can stop early? 
        # The task says max is 20. If we get 20, that's optimal.
        if len(best_a) >= 20:
            break

    # Final verification
    final_a = sorted(best_a)
    assert all(x in final_a for x in seed_elements), "Missing seed elements"
    assert all(1 <= x <= 284 for x in final_a), "Out of range"
    assert len(set(final_a)) == len(final_a), "Duplicates"
    
    # Verify Sidon property
    sums = set()
    n = len(final_a)
    valid = True
    for i in range(n):
        for j in range(i, n):
            s = final_a[i] + final_a[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    assert valid, "Invalid Sidon set"
    
    return final_a
# EVOLVE-BLOCK-END
