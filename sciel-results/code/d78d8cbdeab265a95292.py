import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s buffer for 90s limit
    
    # Initial best known valid set
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)
    
    def compute_sums_by_cardinality(s):
        """
        Compute sums_by_k for a given set s using bitsets for speed.
        Returns a list of sets.
        """
        n = len(s)
        if n == 0:
            return [{0}]
        
        max_sum = sum(s)
        # Use list of bitsets (Python integers) where bit i is set if sum i is achievable
        # sums_by_k[k] is a bitset
        
        sums_by_k = [0] * (n + 1)
        sums_by_k[0] = 1  # bit 0 is set (sum 0)
        
        for v in s:
            # Update in reverse order
            for k in range(n, 0, -1):
                sums_by_k[k] |= (sums_by_k[k-1] << v)
        
        # Convert bitsets to sets for intersection counting
        result = [set() for _ in range(n + 1)]
        for k in range(n + 1):
            mask = sums_by_k[k]
            # Extract set bits
            # Efficient way: while mask: lsb = mask & -mask; idx = lsb.bit_length()-1; add idx; mask ^= lsb
            # But for large max_sum, this might be slow if many bits.
            # Alternative: just check each possible sum? No, bitset is faster for intersection.
            # Let's keep them as bitsets for collision counting.
            # Wait, find_collisions uses set intersection.
            # Let's return bitsets and modify find_collisions to use bitwise AND.
            
        return sums_by_k # Return bitsets

    def count_collisions_bitset(sums_by_k):
        """
        Count total cross-cardinality collisions using bitwise AND.
        Uses int.bit_count() for speed.
        """
        n = len(sums_by_k) - 1
        total = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                inter = sums_by_k[i] & sums_by_k[j]
                if inter:
                    total += inter.bit_count()
        return total

    def global_conflict_removal_bitset(initial_set, time_limit):
        """
        Start with initial_set, iteratively remove the element whose removal
        reduces the total number of cross-cardinality collisions the most.
        Uses bitset DP for speed.
        """
        current_set = list(initial_set)
        sums_by_k = compute_sums_by_cardinality(current_set)
        current_collisions = count_collisions_bitset(sums_by_k)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_idx = -1
            best_collisions = current_collisions
            best_new_set = None
            best_new_sums = None
            
            n = len(current_set)
            if n == 0:
                break
                
            # For each element, try removing it
            for idx in range(n):
                v = current_set[idx]
                new_set = current_set[:idx] + current_set[idx+1:]
                new_sums = compute_sums_by_cardinality(new_set)
                new_collisions = count_collisions_bitset(new_sums)
                
                if new_collisions < best_collisions:
                    best_collisions = new_collisions
                    best_idx = idx
                    best_new_set = new_set
                    best_new_sums = new_sums
                    improved = True
                    
            if improved:
                current_set = best_new_set
                sums_by_k = best_new_sums
                current_collisions = best_collisions
                
        return current_set, current_collisions

    def is_valid_set(s):
        """
        Check if the set is valid (no cross-cardinality sum collisions).
        """
        sums_by_k = compute_sums_by_cardinality(s)
        return count_collisions_bitset(sums_by_k) == 0

    def quick_validity_check(s):
        """
        Fast preliminary check: if any element is > 256 or set has duplicates, invalid.
        Then do full check.
        """
        if len(s) != len(set(s)):
            return False
        if any(v < 1 or v > 256 for v in s):
            return False
        return is_valid_set(s)

    # Strategy 1: Greedy construction
    # Start with empty set, add elements that maintain validity.
    # Order candidates by some heuristic (e.g., larger values first to spread sums).
    
    def greedy_construct():
        current = []
        candidates = list(range(1, 257))
        best_result = []
        
        # Try multiple orderings
        orderings = [
            sorted(candidates, reverse=True),
            sorted(candidates),
            list(range(1, 257, 2)),  # odds first
            list(range(2, 257, 2)),  # evens first
            sorted(candidates, key=lambda x: (x % 2, x)),
            sorted(candidates, key=lambda x: (-x % 3, x)),
        ]
        
        for order in orderings:
            if time.time() > deadline - 15:
                break
            curr = []
            for v in order:
                new_curr = curr + [v]
                if is_valid_set(new_curr):
                    curr = new_curr
                if time.time() > deadline - 10:
                    break
            if len(curr) > len(best_result):
                best_result = curr
        return best_result

    # Try greedy construction
    g_set = greedy_construct()
    if len(g_set) > best_size:
        best_set = g_set
        best_size = len(best_set)

    # Strategy 2: Global conflict removal from random large sets
    # This is often more effective for finding larger valid sets if they exist.
    
    random.seed(42)
    
    # Try different sizes and random samples
    # Start with large sets (e.g., 30-50) and remove conflicts.
    
    for attempt in range(80):
        if time.time() > deadline - 8:
            break
            
        size = random.randint(25, 45)
        if size > 256: size = 256
        random_set = random.sample(range(1, 257), size)
        
        # Run global conflict removal with a time limit per attempt
        time_per_attempt = max(0.3, (deadline - 8 - time.time()) / max(1, 80 - attempt))
        result_set, final_collisions = global_conflict_removal_bitset(random_set, time.time() + time_per_attempt)
        
        if final_collisions == 0:
            if len(result_set) > best_size:
                best_set = result_set
                best_size = len(result_set)
                
    # Strategy 3: Local search / Swap
    # If we have a valid set, try to improve it by swapping elements or adding new ones.
    
    def extend_set(current_set):
        """
        Try to add elements to current_set.
        """
        curr = list(current_set)
        candidates = [v for v in range(1, 257) if v not in curr]
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            best_new_set = None
            
            for v in candidates:
                if time.time() > deadline - 3:
                    break
                new_set = curr + [v]
                if is_valid_set(new_set):
                    best_new_set = new_set
                    improved = True
                    break
            
            if improved:
                curr = best_new_set
                candidates = [v for v in range(1, 257) if v not in curr]
        
        return curr

    # Strategy 4: Swap-based local search
    def swap_search(current_set):
        """
        Try to improve by removing one element and adding another (keeping size same or growing).
        """
        curr = list(current_set)
        in_set = set(curr)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            best_new_set = None
            
            # Try all (remove_idx, add_val) pairs
            for idx in range(len(curr)):
                if time.time() > deadline - 3:
                    break
                removed = curr[idx]
                base_without = curr[:idx] + curr[idx+1:]
                
                for v in range(1, 257):
                    if v in in_set or v == removed:
                        continue
                    new_set = base_without + [v]
                    if is_valid_set(new_set):
                        # Valid swap: same size, different composition
                        # Prefer swaps that allow future growth
                        best_new_set = new_set
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                curr = best_new_set
                in_set = set(curr)
        
        return curr

    # Try to extend the best set found
    if best_size > 0:
        extended = extend_set(best_set)
        if len(extended) > best_size:
            best_set = extended
            best_size = len(best_set)
        
        # Try swap search to find a better composition that allows further growth
        if time.time() < deadline - 5:
            swapped = swap_search(best_set)
            if len(swapped) > best_size:
                best_set = swapped
                best_size = len(best_set)
            else:
                # Even if same size, try extending from the swapped version
                if time.time() < deadline - 3:
                    ext2 = extend_set(swapped)
                    if len(ext2) > best_size:
                        best_set = ext2
                        best_size = len(best_set)

    # Ensure types are correct
    return [int(x) for x in best_set]

# Quick self-test if run directly
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    print(f"Result: {len(result)} elements, took {time.time()-t0:.2f}s")
    print(f"Set: {result}")
