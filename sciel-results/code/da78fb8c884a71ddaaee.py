import random
import itertools
import time

def is_sidon(S):
    """Check if set S is a Sidon set (all unordered pair sums unique including diagonals)."""
    sums = set()
    for i, x in enumerate(S):
        for j, y in enumerate(S):
            if i <= j:
                s = x + y
                if s in sums:
                    return False
                sums.add(s)
    return True

def count_sidon_subsets_of_size_k(A, k):
    """Count the number of Sidon subsets of size k in A."""
    count = 0
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave margin for evaluator overhead
    
    # We want to minimize the largest Sidon subset size.
    # Baseline range(12) has max Sidon size 5.
    # Target: Find a set with 0 Sidon subsets of size 5 (max size <= 4).
    # Ideally, find a set with 0 Sidon subsets of size 4 (max size <= 3).
    
    best_set = list(range(12))
    best_score = 5  # Current best max Sidon size
    
    # We will use local search to minimize the count of Sidon subsets of size 5.
    # If we reach 0, we switch to minimizing Sidon subsets of size 4.
    
    # Initialize with a random set
    current_set = random.sample(range(129), 12)
    
    # Function to evaluate a set
    def evaluate_set(S):
        # Returns (max_sidon_size, count_k) where k is the largest size with non-zero count
        # We care most about large k.
        # Let's just compute max size directly for the final check, 
        # but for search, we minimize count of size 5, then 4, etc.
        pass

    # Local Search Strategy:
    # 1. Minimize count of Sidon-5.
    # 2. If count is 0, minimize count of Sidon-4.
    # 3. If count is 0, minimize count of Sidon-3 (unlikely to be 0 for 12 elements? 
    #    Max Sidon size 3 means no Sidon-4. Is it possible to have no Sidon-4 in 12 elements?
    #    Yes, if the set is very structured. But let's aim for 4 first).
    
    # We will run multiple restarts of local search.
    max_restarts = 100
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Restart with random set
        current_set = random.sample(range(129), 12)
        
        # Phase 1: Minimize Sidon-5 count
        current_count_5 = count_sidon_subsets_of_size_k(current_set, 5)
        
        # If we already have 0, great.
        # Local search loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all possible swaps or moves? 
            # With 12 elements, swapping 2 elements is 66 pairs.
            # Or moving 1 element to a new value.
            # Let's try swapping two elements in the set with two random values from [0,128].
            # Or simpler: pick 2 indices i, j. Replace A[j] with a new random value.
            
            # To be efficient, we try random perturbations.
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                i = random.randint(0, 11)
                j = random.randint(0, 11)
                if i == j:
                    continue
                
                old_val = current_set[j]
                new_val = random.randint(0, 128)
                
                if new_val in current_set:
                    continue
                
                current_set[j] = new_val
                new_count_5 = count_sidon_subsets_of_size_k(current_set, 5)
                
                if new_count_5 < current_count_5:
                    current_count_5 = new_count_5
                    improved = True
                    if current_count_5 == 0:
                        break
                else:
                    # Revert
                    current_set[j] = old_val
            
            if current_count_5 == 0:
                break
        
        # If we found a set with 0 Sidon-5, its max Sidon size is <= 4.
        # Let's verify the actual max size to be sure.
        if current_count_5 == 0:
            # Check if it has Sidon-4
            count_4 = count_sidon_subsets_of_size_k(current_set, 4)
            if count_4 == 0:
                # Max size <= 3
                # Check if it has Sidon-3
                count_3 = count_sidon_subsets_of_size_k(current_set, 3)
                if count_3 == 0:
                    max_size = 2
                else:
                    max_size = 3
            else:
                max_size = 4
            
            if max_size < best_score:
                best_score = max_size
                best_set = list(current_set)
                
                if best_score <= 3:
                    # We found a very good set. We can try to polish or just return.
                    # Score 5/3 = 1.67 is excellent.
                    # Let's keep searching a bit more in case we find 2? Unlikely.
                    # But if we found 4, we can try to improve to 3.
                    pass
        
        # If we haven't found 0 Sidon-5, we continue to next restart.
        # We could also keep the best set found so far (lowest count_5) to guide next restart?
        # For now, random restarts.

    return best_set
