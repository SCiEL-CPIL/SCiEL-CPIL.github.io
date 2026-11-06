import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    
    n = 512
    k = 4
    
    # 1. Base-3 Cantor Set Construction
    # Numbers with no digit '2' in base 3 form a 3-AP-free set.
    # For 4-AP-free, we can use a slightly denser variant or just start with this.
    # Actually, the standard Cantor set (no 2s) is 3-AP-free, which implies 4-AP-free?
    # No, 3-AP-free does NOT imply 4-AP-free. But it's a good start.
    # A better start for 4-AP-free is to take numbers whose base-4 digits are only 0,1?
    # Or use the known construction for r-AP-free sets.
    # For now, let's use the base-3 no-2 set as a strong seed.
    # Size of Cantor set in [1, 3^m] is 2^m.
    # 3^6 = 729 > 512. So m=6.
    # Size approx (2/3)^6 * 512 ~ 54. This is small.
    
    # Better strategy:
    # Use a greedy algorithm but with a smarter order or a different initialization.
    # The prompt suggests "base-3 digit-based method ... selecting numbers with no digit '2' in base 3".
    # Let's verify if that's 4-AP free.
    # If A is 3-AP free, is it 4-AP free?
    # A 4-AP a, a+d, a+2d, a+3d contains a 3-AP a, a+d, a+2d.
    # So YES, any 3-AP-free set is 4-AP-free.
    # The Cantor set (no 2s in base 3) is 3-AP-free.
    # However, its density is low (3^{-m} * 2^m = (2/3)^m).
    # For n=512, m=6, density (2/3)^6 ~ 0.087. Size ~ 45.
    # This is much smaller than the greedy seed which likely gets ~100+.
    
    # Let's try a different "structured" start:
    # Random greedy is actually quite good.
    # The key is to speed up the local search.
    # The current program uses set membership checks which are O(1) but have overhead.
    # Bitsets in NumPy can speed up validation significantly.
    
    # Let's implement a fast bitset-based local search starting from a greedy seed.
    
    def greedy_seed():
        """Fast greedy construction to get a valid starting set."""
        selected = set()
        for x in range(1, n + 1):
            valid = True
            # Check if adding x creates a 4-AP
            # x could be the 1st, 2nd, 3rd, or 4th term
            # Case 1: x is 1st term: x, x+d, x+2d, x+3d
            for d in range(1, (n - x) // 3 + 1):
                if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # Case 2: x is 2nd term: a, x, x+d, x+2d where a = x-d
            for d in range(1, (x - 1) // 3 + 1):
                a = x - d
                if a in selected and (x + d) in selected and (x + 2*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # Case 3: x is 3rd term: a, a+d, x, x+d where a = x-2d
            for d in range(1, (x - 1) // 2 + 1):
                a = x - 2*d
                if a in selected and (a + d) in selected and (x + d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # Case 4: x is 4th term: a, a+d, a+2d, x where a = x-3d
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3*d
                if a in selected and (a + d) in selected and (a + 2*d) in selected:
                    valid = False
                    break
            if valid:
                selected.add(x)
        return selected

    # Bitset representation for fast validation
    # We represent the set as a boolean numpy array of size n+1 (1-indexed)
    
    def is_valid_4ap_free_bitset(bitset):
        """
        Check if the set represented by bitset (numpy array of 0/1) contains no 4-AP.
        This is O(n^2) but with low constant factor due to vectorization? 
        Actually, checking all 4-APs is hard to vectorize directly for "any" AP.
        We can precompute all 4-APs in [1, n] and check if any are fully contained.
        """
        # Precomputation of all 4-APs is better done once outside, but we need to pass bitset.
        # Let's stick to a faster iterative check or precompute AP indices.
        pass

    # Let's precompute all 4-APs in [1, 512]
    # Total number of 4-APs is sum_{a=1}^{n} sum_{d=1}^{(n-a)/3} 1
    # This is roughly n^2 / 6 ~ 512^2 / 6 ~ 43k APs.
    # We can store them as a list of tuples (a, a+d, a+2d, a+3d).
    # Then validation is: for each AP, check if all 4 bits are 1.
    # Using numpy: mask = all_aps[:, 0] & all_aps[:, 1] ...
    
    all_aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            all_aps.append((a, a+d, a+2*d, a+3*d))
    
    # Convert to numpy array for vectorized checking
    # Shape: (num_aps, 4)
    all_aps_np = np.array(all_aps)
    
    def is_valid_bitset(bitset):
        """
        bitset: numpy array of shape (n+1,) with 1s at selected indices.
        Returns True if no 4-AP is fully contained.
        """
        # Extract bits for all AP terms
        # We need to check if any row has all 1s.
        # terms = bitset[all_aps_np] -> shape (num_aps, 4)
        # valid = not any row has sum 4.
        # To avoid creating the large array every time, we can use:
        # np.any(np.all(bitset[all_aps_np] == 1, axis=1))
        # This creates an array of size ~43k x 4 = 172k integers. Very fast.
        terms = bitset[all_aps_np]
        return not np.any(np.all(terms == 1, axis=1))

    # Initial greedy seed
    best_set = greedy_seed()
    best_size = len(best_set)
    
    # Convert to bitset
    best_bitset = np.zeros(n + 1, dtype=np.int8)
    for x in best_set:
        best_bitset[x] = 1
        
    # Local search
    unselected = set(range(1, n + 1)) - best_set
    random.seed(42)
    
    # Optimization: Precompute conflicts?
    # If we add x, which APs are violated?
    # We can maintain a count of "blocked" APs?
    # Simpler: just use the fast bitset validation.
    
    while time.time() < deadline:
        improved = False
        
        # Try adding one element
        if unselected:
            # Sort unselected by some heuristic? Or random?
            # Random is fine if validation is fast.
            # Let's try random sampling with a limit.
            candidates = list(unselected)
            random.shuffle(candidates)
            
            for x in candidates[:200]: # Try up to 200 random additions
                if time.time() > deadline - 2:
                    break
                
                # Test addition
                test_bitset = best_bitset.copy()
                test_bitset[x] = 1
                
                if is_valid_bitset(test_bitset):
                    # Accept
                    best_bitset = test_bitset
                    best_set.add(x)
                    unselected.remove(x)
                    best_size += 1
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try swaps: remove y, add x
        # This maintains size. Useful to escape local optima for future additions?
        # Or try remove y, add x1, x2 (net +1)
        
        if best_size > 0 and len(unselected) >= 1:
            # Try single swap to improve "potential" for additions?
            # Or just try 2-for-1 swaps directly to increase size.
            
            # 2-for-1 swap: remove y, add x1, x2
            if len(unselected) >= 2:
                candidates = list(unselected)
                random.shuffle(candidates)
                x1 = candidates[0]
                x2 = candidates[1]
                
                # Find a y to remove that allows adding x1, x2
                # We can iterate through a few random y's
                y_candidates = list(best_set)
                random.shuffle(y_candidates)
                
                for y in y_candidates[:50]:
                    if time.time() > deadline - 2:
                        break
                    
                    test_bitset = best_bitset.copy()
                    test_bitset[y] = 0
                    test_bitset[x1] = 1
                    test_bitset[x2] = 1
                    
                    if is_valid_bitset(test_bitset):
                        # Accept
                        best_bitset = test_bitset
                        best_set.remove(y)
                        best_set.add(x1)
                        best_set.add(x2)
                        unselected.remove(x1)
                        unselected.remove(x2)
                        unselected.add(y)
                        best_size += 1
                        improved = True
                        break
        
        if not improved:
            # Try more aggressive moves or restart?
            # For now, just continue loop. If stuck, we might need a different heuristic.
            # Given the fast validation, we can afford to try more random swaps.
            pass

    return sorted(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Quick check
    s = set(res)
    valid = True
    for a in s:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in s and (a+2*d) in s and (a+3*d) in s:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
