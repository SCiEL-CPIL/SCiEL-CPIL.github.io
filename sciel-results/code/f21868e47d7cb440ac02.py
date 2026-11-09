import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    def get_subset_sums(elements):
        """Get all nonempty subset sums as a set. Returns None if duplicate found."""
        sums = set()
        n = len(elements)
        # Use bitmask iteration
        for mask in range(1, 1 << n):
            s = 0
            # Optimization: only compute if we haven't found a duplicate yet
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return None  # Duplicate sum found
            sums.add(s)
        return sums

    def is_valid(elements):
        """Check if the set of elements is valid."""
        if not elements:
            return True
        
        sums = get_subset_sums(elements)
        if sums is None:
            return False
        
        # Check divisibility
        # Sort sums to check multiples efficiently
        sorted_sums = sorted(sums)
        # Create a set for O(1) lookup
        sum_set = set(sorted_sums)
        
        for s in sorted_sums:
            # Check if any multiple of s exists in the set
            # 2s, 3s, ...
            multiple = 2 * s
            while multiple <= sorted_sums[-1]:
                if multiple in sum_set:
                    return False
                multiple += s
        return True

    def try_add(elements, candidate):
        """Try adding candidate to elements. Returns new list if valid, else None."""
        new_elements = elements + [candidate]
        if is_valid(new_elements):
            return new_elements
        return None

    # Initial best set
    best_set = []
    
    # Pool of candidates
    # We want to maximize size, so we want to include as many numbers as possible.
    # Larger numbers are "safer" because they are less likely to divide other sums.
    # Let's try to build a set using a randomized greedy approach with restarts.
    
    max_val = 256
    all_numbers = list(range(1, max_val + 1))
    
    # Strategy:
    # 1. Try to build a set by adding numbers in a random order, preferring larger numbers.
    # 2. Use local search to improve.
    
    # We will run multiple iterations of a randomized greedy + local search.
    
    def run_local_search(initial_set, candidates_pool):
        current = list(initial_set)
        current_set = set(current)
        remaining = [x for x in candidates_pool if x not in current_set]
        
        # Greedy add
        # Sort remaining by descending value to prefer large numbers
        remaining.sort(reverse=True)
        
        for cand in remaining:
            if time.time() > deadline:
                break
            new_set = try_add(current, cand)
            if new_set is not None:
                current = new_set
                current_set.add(cand)
                remaining.remove(cand)
        
        # Local search: try swaps
        # We try to replace an element in current with one in remaining to potentially allow more additions later
        # This is complex. A simpler improvement:
        # Try to add any remaining element if we remove one existing element?
        # Actually, just trying to add is usually enough if we restart with different random orders.
        
        return current

    # Randomized search with multiple restarts
    best_size = 0
    best_final_set = []
    
    # Pre-shuffle the pool? No, we want to try different orders.
    
    # Let's try a few specific strategies:
    # 1. Descending order greedy
    # 2. Random order greedy (multiple times)
    # 3. Random order with bias towards larger numbers
    
    # Strategy 1: Descending
    if time.time() < deadline:
        candidates = list(range(256, 0, -1))
        current = []
        for cand in candidates:
            if time.time() > deadline:
                break
            new_set = try_add(current, cand)
            if new_set is not None:
                current = new_set
        
        if len(current) > len(best_final_set):
            best_final_set = current
    
    # Strategy 2: Randomized Greedy with Local Search
    # We will run this for a while.
    
    # To speed up is_valid, we can cache results?
    # The number of unique subsets of size up to 15 is large, but we don't re-check the same set often.
    
    # Let's use a faster divisibility check if possible.
    # The current one is O(M log M + M * MaxSum/min_sum).
    
    # Let's try to improve the search by using a "repair" mechanism.
    # If we can't add a candidate, maybe we can swap an existing element with a smaller one to make room?
    # This is getting complex. Let's stick to multiple randomized greedy passes.
    
    # We'll run 1000 randomized greedy passes, but stop if we find a set of size 12 or 13.
    
    # Actually, let's look at the structure.
    # If we pick numbers that are pairwise coprime and large, it's more likely to work.
    
    # Let's try a weighted random selection where larger numbers have higher probability.
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Create a weighted list
        # Weight = value
        # Or just shuffle and pick
        candidates = list(range(256, 0, -1))
        random.shuffle(candidates)
        
        # Bias towards larger numbers: 
        # We can sort by (random.random() / x) to favor small? No, we want large.
        # Sort by (random.random() * x) ?
        
        # Simpler: Just shuffle.
        
        current = []
        current_set = set()
        
        for cand in candidates:
            if time.time() > deadline:
                break
            # Check if adding cand is valid
            # Optimization: If len(current) is already >= len(best_final_set), 
            # we might want to be more selective? No, just try.
            
            new_set = try_add(current, cand)
            if new_set is not None:
                current = new_set
                current_set.add(cand)
        
        if len(current) > len(best_final_set):
            best_final_set = current
            # If we found a very large set, we might want to stop early?
            # The max possible is likely around 10-12.
            # If we find 12, we can try to find 13.
            # If we find 10, keep going.
            
            # Update deadline? No, keep going to see if we can do better.
            
            # If we find a set of size 12, we can spend more time on local search on that set.
            if len(best_final_set) >= 11:
                # Perform local search on this best set
                # Try to add any remaining number
                remaining = [x for x in range(1, 257) if x not in current_set]
                # Sort remaining by descending value
                remaining.sort(reverse=True)
                for cand in remaining:
                    if time.time() > deadline:
                        break
                    new_set = try_add(best_final_set, cand)
                    if new_set is not None:
                        best_final_set = new_set
                        current_set.add(cand)
                
                # Try swaps: remove one from best, add one from remaining
                # This is expensive. Let's just do a few random swaps.
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    if len(best_final_set) < 2:
                        continue
                    # Pick a random element to remove
                    idx = random.randint(0, len(best_final_set) - 1)
                    removed = best_final_set[idx]
                    candidate_set = best_final_set[:idx] + best_final_set[idx+1:]
                    
                    # Try to add any remaining element
                    # Sort remaining by descending value
                    remaining_for_swap = [x for x in range(256, 0, -1) if x not in set(candidate_set)]
                    for cand in remaining_for_swap:
                        new_set = try_add(candidate_set, cand)
                        if new_set is not None:
                            # If we found a valid set, accept it?
                            # Only if it's the same size or larger? 
                            # It's the same size (removed 1, added 1).
                            # So we don't improve size, but might allow further growth?
                            # This is a neutral move.
                            # Let's accept it to explore the neighborhood.
                            best_final_set = new_set
                            break
                    # We don't update best_final_set if we didn't find a valid swap?
                    # Actually, if we didn't find a valid swap, we should revert?
                    # The code above only updates best_final_set if a swap is found.
                    # If not found, best_final_set remains unchanged.
    
    # Final verification
    if not is_valid(best_final_set):
        # Fallback to a known valid small set
        # [31, 32, 34] was in the seed.
        # Let's find a valid set of size 4 or 5 quickly.
        # Try [100, 150, 200, 250]
        test_set = [100, 150, 200, 250]
        if is_valid(test_set):
            best_final_set = test_set
        else:
            best_final_set = [31, 32, 34]
            
    return [int(x) for x in best_final_set]
