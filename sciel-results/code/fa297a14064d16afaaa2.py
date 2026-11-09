import time
import random
import sys

def construct():
    deadline = time.time() + 140.0
    fixed = [1, 2, 5, 11]
    N = 284
    
    def get_energy(A):
        """Calculate the number of conflicting sums."""
        sums = {}
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    sums[s] += 1
                else:
                    sums[s] = 1
        conflicts = 0
        for count in sums.values():
            if count > 1:
                conflicts += count - 1
        return conflicts, sums

    def solve_sa(K, time_limit):
        """
        Try to find a Sidon set of size K containing 'fixed'.
        Returns the set if found, else None.
        """
        start_time = time.time()
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        max_steps = 1000000
        
        # Initialize
        # Fixed elements are 1, 2, 5, 11
        # We need K - 4 more elements
        var_count = K - 4
        if var_count < 0:
            return None
            
        # Random initialization
        current_set = set(fixed)
        while len(current_set) < K:
            x = random.randint(1, N)
            current_set.add(x)
        
        current_list = list(current_set)
        current_energy, current_sums = get_energy(current_list)
        best_energy = current_energy
        best_list = list(current_list)
        
        if current_energy == 0:
            return sorted(current_list)
            
        steps = 0
        while time.time() - start_time < time_limit:
            # Pick a random variable index (not fixed)
            # Fixed indices: 0, 1, 2, 3 in sorted list? No, list is unsorted.
            # We need to know which elements are fixed.
            # Let's keep the list and a set of fixed values.
            
            # Optimization: Keep track of which indices are fixed?
            # Simpler: Just pick a random element from the list.
            # If it's fixed, skip? Or allow moving fixed elements?
            # The task says "containing 1, 2, 5 and 11".
            # So they must remain in the set.
            # So we should NOT move fixed elements.
            
            # We need to identify fixed elements in current_list.
            # This is O(K).
            idx = random.randint(0, K - 1)
            val = current_list[idx]
            if val in fixed:
                continue
                
            # Choose a new value
            new_val = random.randint(1, N)
            if new_val in current_set:
                # Avoid duplicates in the set itself
                continue
            
            # Evaluate move
            # Remove val, add new_val
            
            # Update sums and energy incrementally
            # This is complex to code correctly and efficiently in Python without numpy for this small K.
            # Given K <= 20, O(K^2) recalculation is 400 ops.
            # 100,000 steps * 400 ops = 40,000,000 ops.
            # Python can do ~10M-50M simple ops/sec.
            # So ~1-4 seconds for 100k steps.
            # This is acceptable.
            
            new_list = current_list.copy()
            new_list[idx] = new_val
            new_set = set(new_list)
            
            # Check for duplicate elements (should be handled by new_val not in current_set)
            
            new_energy, _ = get_energy(new_list)
            
            # Acceptance
            delta_E = new_energy - current_energy
            if delta_E < 0 or random.random() < pow(2.718281828, -delta_E / T):
                current_list = new_list
                current_set = new_set
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_list = list(current_list)
                    if best_energy == 0:
                        return sorted(best_list)
                
            # Cooling
            T *= cooling
            if T < min_T:
                T = min_T
                # Restart if stuck?
                # Simple restart: re-initialize
                # For now, just keep going.
                
            steps += 1
            if steps > max_steps:
                break
                
        return None

    # Try sizes from 20 down to 4
    best_result = None
    
    # Prioritize 20
    # We will allocate time slices
    # 20: 30s
    # 19: 20s
    # 18: 10s
    # 17: 10s
    # ...
    
    sizes_to_try = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
    time_alloc = {
        20: 30,
        19: 20,
        18: 15,
        17: 10,
        16: 5,
        15: 5,
        14: 5,
        13: 5,
        12: 5,
        11: 5,
        10: 5,
        9: 2,
        8: 2,
        7: 1,
        6: 1,
        5: 1,
        4: 1
    }
    
    for K in sizes_to_try:
        if time.time() >= deadline - 5: # 5s margin
            break
            
        t_limit = time_alloc.get(K, 2)
        # Ensure we don't exceed deadline
        if time.time() + t_limit > deadline - 5:
            t_limit = max(0.1, deadline - 5 - time.time())
            
        result = solve_sa(K, t_limit)
        if result is not None:
            best_result = result
            # If we found a valid set, do we continue?
            # We want the MAX size.
            # Since we iterate 20, 19, ..., the first one found is the largest.
            # So we can return immediately?
            # Yes, because we start from the top.
            return result
            
    return best_result
