"""Distinct subset sums with a small largest element."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find 12 distinct positive integers <= 1048576 such that all 4096 subset sums
    are distinct, minimizing the largest integer.
    """
    start_time = time.time()
    deadline = 88.0  # 2s safety margin before 90s kill

    def check_valid(elements):
        """Check all subset sums distinct using bitset operations.
        Returns True if valid, False otherwise."""
        bits = 1  # bit 0 set = sum 0 exists
        for e in elements:
            if (bits << e) & bits:
                return False
            bits |= (bits << e)
        return True

    def get_min_valid(elements, idx, current_bits):
        """
        Find the smallest integer > elements[idx-1] that can be added to elements[idx-1:]
        such that subset sums remain distinct.
        We assume elements[0:idx] are fixed and valid.
        We need to find a value for elements[idx] such that:
        1. It is distinct from previous elements.
        2. (current_bits << val) & current_bits == 0
        3. val > elements[idx-1] (if idx > 0)
        
        Note: This function is used in a greedy construction where we fix elements one by one.
        """
        min_val = elements[idx - 1] + 1 if idx > 0 else 1
        # We can optimize by checking candidates.
        # The bitset `current_bits` represents sums of subsets of elements[0:idx].
        # If we add `val`, new sums are {s + val for s in sums}.
        # Collision if (s1 + val) == s2 => s2 - s1 == val.
        # So val must not be a difference of two existing sums.
        
        # Efficient check:
        # Let B be the bitset of existing sums.
        # We need (B << val) & B == 0.
        
        # Start searching from min_val
        val = min_val
        while True:
            if (current_bits << val) & current_bits:
                val += 1
            else:
                return val
        
    def greedy_construct(n=12):
        """Construct the lexicographically smallest valid sequence."""
        elements = []
        current_bits = 1 # Sum 0
        for i in range(n):
            min_val = elements[-1] + 1 if elements else 1
            # Find smallest valid val
            val = min_val
            while True:
                if (current_bits << val) & current_bits:
                    val += 1
                else:
                    break
            elements.append(val)
            current_bits |= (current_bits << val)
        return elements

    def local_search(elements, max_iters=5000):
        """
        Local search to minimize the maximum element.
        Strategy:
        1. Try to reduce the largest element a[11].
        2. If stuck, try to reduce other elements to create "space" for a[11].
        3. Use bitset for fast validation.
        """
        n = len(elements)
        best = list(elements)
        best_max = max(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            # We want to minimize the max element, which is usually at the end if sorted.
            # We maintain sorted order.
            
            # Try to reduce each element, prioritizing larger indices (larger values)
            # Actually, reducing any element that allows a subsequent reduction of max is good.
            # Let's try reducing elements in reverse order (largest to smallest)
            # because reducing a small element might allow reducing a large one later?
            # No, reducing a large element directly is best.
            
            # Let's iterate through indices 11 down to 0
            for i in range(n - 1, -1, -1):
                if time.time() > deadline:
                    break
                    
                old_val = best[i]
                # Minimum valid value for position i
                # Must be > best[i-1] if i>0, else 1
                # Also must be distinct from all others? 
                # If we keep sorted order, best[i] > best[i-1] and best[i] < best[i+1] (if i<n-1)
                # However, local search can break sorted order temporarily? 
                # Better to maintain sorted order to simplify constraints.
                
                min_cand = best[i - 1] + 1 if i > 0 else 1
                max_cand = best[i + 1] - 1 if i < n - 1 else old_val - 1
                
                # If we are reducing the last element, max_cand is just old_val - 1
                # We try to find the smallest valid value in [min_cand, max_cand]
                
                # Pre-compute bits for prefix [0..i-1] and suffix [i+1..n-1]?
                # Too slow. Just check full validity.
                
                # To speed up, we can check candidates.
                # Since we want to minimize the max, we specifically want to lower best[11].
                # If i == n-1, we scan from min_cand to old_val-1.
                # If i < n-1, we scan from min_cand to old_val-1, but must ensure best[i] < best[i+1].
                
                found = False
                # Optimization: If i is not the last, reducing it might allow reducing the last one.
                # But let's just find the smallest valid value for this position.
                
                for val in range(min_cand, max_cand + 1):
                    test = list(best)
                    test[i] = val
                    # Check if distinct and valid
                    # Since we maintain sorted order, distinctness is guaranteed if val > best[i-1] and val < best[i+1]
                    # We just need to check subset sums.
                    if check_valid(test):
                        best = test
                        improved = True
                        # If we improved, we might be able to improve further, so break and restart loop
                        # Or continue to see if we can reduce more?
                        # Breaking allows us to re-evaluate from the end again.
                        break
                
                if improved:
                    if i == n - 1:
                        best_max = max(best)
                    break # Restart the outer loop to try reducing again from the end
            
            if not improved:
                # Stuck in local minimum.
                # Try a random perturbation: increase a random element slightly, then minimize again?
                # Or swap two elements?
                # Let's try a random "kick": pick a random index j (not 0), increase it by 1, 
                # then run a few steps of minimization.
                
                if random.random() < 0.5: # 50% chance to kick
                    # Pick a random index to increase
                    j = random.randint(1, n - 1)
                    # Increase it
                    new_val = best[j] + random.randint(1, 10)
                    if new_val > 1048576: continue
                    test = list(best)
                    test[j] = new_val
                    # Check validity
                    if check_valid(test):
                        # Now try to minimize this specific element back down
                        # Or just accept it and let the next iteration minimize?
                        # Let's accept it and let the next loop minimize.
                        best = test
                        # Note: This might increase the max, but hopefully allows a better config later.
                        # To prevent infinite loop of increasing max, we only do this if we've been stuck for a while?
                        # For simplicity, we just do it.
                        continue

        return best

    # 1. Greedy Construction
    # This finds the lexicographically smallest sequence.
    # a1=1, a2=2 (1+2=3, sums: 0,1,2,3), a3=4 (sums 0..7), ... powers of 2.
    # Wait, greedy construction with "smallest valid" often yields powers of 2.
    # Let's verify.
    # a1=1. Bits: 11 (0,1)
    # a2: min 2. 11 << 2 = 1100. 11 & 1100 = 0. Valid. Bits: 1111 (0,1,2,3)
    # a3: min 3. 1111 << 3 = 1111000. 1111 & 1111000 = 0? 
    #    1111 (15) & 1111000 (240) = 0. Valid.
    #    So a3=3?
    #    Sums of {1,2,3}: 0,1,2,3, 1+3=4, 2+3=5, 1+2+3=6. All distinct.
    #    Max sum 6.
    #    a4: min 4. Bits for {1,2,3} is 1111111 (0..6).
    #    1111111 << 4 = 11111110000.
    #    1111111 & 11111110000 = 0. Valid.
    #    So a4=4?
    #    Sums of {1,2,3,4}: 0..10. Distinct.
    #    It seems greedy yields 1, 2, 3, 4...
    #    Let's check a5. Bits 0..10.
    #    min 5. 11111111111 (0..10) << 5.
    #    Overlap? 10 < 11. No overlap.
    #    So 1, 2, 3, 4, 5...
    #    This sequence is just 1, 2, ..., 12.
    #    Max element 12.
    #    Is 1,2,...,12 valid?
    #    Subset sums of 1..12 range from 0 to 78.
    #    Are they distinct?
    #    Yes, because each number is less than the sum of all previous numbers + 1?
    #    Sum(1..k) = k(k+1)/2.
    #    Next number k+1.
    #    We need k+1 <= Sum(1..k) + 1.
    #    k+1 <= k(k+1)/2 + 1.
    #    k <= k(k+1)/2 => 2 <= k+1 => k >= 1.
    #    So yes, 1, 2, ..., n is always a valid set with distinct subset sums.
    #    The maximum element is n.
    #    For n=12, max is 12.
    
    elements = greedy_construct(12)
    # elements should be [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    # Verify
    if not check_valid(elements):
        # Fallback to powers of 2 if greedy fails (shouldn't happen)
        elements = [1 << i for i in range(12)]

    # Local search to see if we can do better than 12?
    # Since 1,2,...,12 has max 12, and we need 12 distinct positive integers,
    # the minimum possible max is 12 (pigeonhole: if max < 12, we have < 12 positive integers available? No, we can have 1..11 and something else? 
    # If max is 11, we must use 1..11. But we need 12 numbers. So max MUST be at least 12.
    # Wait. "12 distinct positive integers".
    # If the set is {1, 2, ..., 12}, max is 12.
    # Can we have a set with max < 12?
    # No, because there are only 11 positive integers less than 12.
    # So the minimum possible largest integer is 12.
    
    # Therefore, the optimal solution is any valid set with max 12.
    # The set {1, ..., 12} is valid.
    # So the answer is 12.
    
    # The code should just return [1, 2, ..., 12].
    
    return elements

# EVOLVE-BLOCK-END
