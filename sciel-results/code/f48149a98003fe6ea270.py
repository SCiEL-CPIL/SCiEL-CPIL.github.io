# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 4096
    target_size = 80
    
    # We need a set S of distinct positive integers <= 4096
    # such that all unordered triples (with repetition) have unique sums.
    # This is a B_3[1] sequence.
    
    # Strategy: 
    # 1. Start with a good greedy construction to get a base set.
    # 2. Try to improve by local search / simulated annealing style swaps.
    
    def check_validity(S):
        """Check if set S has all unique triple sums."""
        sums = set()
        k = len(S)
        # For efficiency, we can use the fact that the number of triples is O(k^3)
        # For k=80, that's about 80^3/6 ~ 85k triples, which is manageable.
        # But we need to be careful about the sum range.
        # Max sum is 3*4096 = 12288.
        # We can use a boolean array of size 12289.
        max_sum = 3 * n
        seen = np.zeros(max_sum + 1, dtype=bool)
        
        S_sorted = sorted(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s = S_sorted[i] + S_sorted[j] + S_sorted[l]
                    if seen[s]:
                        return False
                    seen[s] = True
        return True
    
    def greedy_build():
        """Greedy construction: pick smallest possible next element."""
        S = []
        used_sums = set()
        
        # Precompute all possible triples for a given set? No, too slow.
        # Instead, for each candidate x, check if adding x creates a collision.
        # A collision occurs if:
        # 1. x+x+x is already in used_sums
        # 2. x+x+y is already in used_sums for some y in S
        # 3. x+y+y is already in used_sums for some y in S
        # 4. x+y+z is already in used_sums for some y,z in S (y<=z)
        
        # To make this efficient, we can maintain a set of forbidden values.
        # But for simplicity, let's just check directly for small sets.
        # Since we're only going up to ~80 elements, and the set is small initially,
        # direct checking is fine.
        
        for x in range(1, n + 1):
            # Check if x can be added
            valid = True
            S_sorted = sorted(S)
            k = len(S_sorted)
            
            # Generate all new triple sums involving x
            new_sums = set()
            # x, x, x
            new_sums.add(3 * x)
            # x, x, y for y in S
            for y in S_sorted:
                new_sums.add(2 * x + y)
            # x, y, y for y in S
            for y in S_sorted:
                new_sums.add(x + 2 * y)
            # x, y, z for y, z in S, y <= z
            for i in range(k):
                for j in range(i, k):
                    new_sums.add(x + S_sorted[i] + S_sorted[j])
            
            # Check for collisions
            if new_sums & used_sums:
                continue
            
            # No collision, add x
            S.append(x)
            used_sums.update(new_sums)
            
            if len(S) >= target_size:
                break
        
        return S
    
    # Step 1: Greedy build
    best_set = greedy_build()
    best_size = len(best_set)
    
    # Step 2: Try to improve with random restarts and local search
    # Since greedy is deterministic, let's try randomized greedy
    # and then local search.
    
    def randomized_greedy():
        """Randomized greedy: at each step, pick a random valid candidate."""
        S = []
        used_sums = set()
        
        # We'll try to pick elements in a somewhat spread out manner
        # to avoid clustering.
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S) >= target_size:
                break
            
            S_sorted = sorted(S)
            k = len(S_sorted)
            
            new_sums = set()
            new_sums.add(3 * x)
            for y in S_sorted:
                new_sums.add(2 * x + y)
            for y in S_sorted:
                new_sums.add(x + 2 * y)
            for i in range(k):
                for j in range(i, k):
                    new_sums.add(x + S_sorted[i] + S_sorted[j])
            
            if new_sums & used_sums:
                continue
            
            S.append(x)
            used_sums.update(new_sums)
        
        return S
    
    def local_search(S, max_iterations=1000):
        """Try to improve set S by swapping elements."""
        S = sorted(S)
        if not check_validity(S):
            return S
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Try to swap a random element with a random other element
            if len(S) < 2:
                break
            
            i = random.randint(0, len(S) - 1)
            j = random.randint(0, n - 1)
            if j == S[i]:
                continue
            
            # Try replacing S[i] with j
            new_S = S[:]
            new_S[i] = j
            new_S = sorted(set(new_S))  # Ensure distinct and sorted
            
            if len(new_S) != len(S):
                continue
            
            if check_validity(new_S):
                # Accept the swap if it's valid
                # We could also try to add more elements after a swap
                S = new_S
                # Try to extend
                for x in range(1, n + 1):
                    if len(S) >= target_size:
                        break
                    if x in S:
                        continue
                    S_sorted = sorted(S)
                    k = len(S_sorted)
                    
                    new_sums = set()
                    new_sums.add(3 * x)
                    for y in S_sorted:
                        new_sums.add(2 * x + y)
                    for y in S_sorted:
                        new_sums.add(x + 2 * y)
                    for idx in range(k):
                        for jdx in range(idx, k):
                            new_sums.add(x + S_sorted[idx] + S_sorted[jdx])
                    
                    if new_sums & used_sums_check(S):
                        continue
                    
                    S.append(x)
                    S = sorted(S)
                    break
        
        return S
    
    def used_sums_check(S):
        """Compute all triple sums for set S."""
        sums = set()
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(S_sorted[i] + S_sorted[j] + S_sorted[l])
        return sums
    
    # Try multiple randomized greedy runs
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        S = randomized_greedy()
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Try local search on the best set found
    if best_size < target_size and time.time() < deadline - 20:
        improved = local_search(best_set, max_iterations=500)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_size)
    
    # Final attempt: try to extend the best set
    if len(best_set) < target_size and time.time() < deadline - 5:
        S = sorted(best_set)
        used_sums = used_sums_check(S)
        for x in range(1, n + 1):
            if len(S) >= target_size:
                break
            if x in S:
                continue
            
            S_sorted = sorted(S)
            k = len(S_sorted)
            
            new_sums = set()
            new_sums.add(3 * x)
            for y in S_sorted:
                new_sums.add(2 * x + y)
            for y in S_sorted:
                new_sums.add(x + 2 * y)
            for i in range(k):
                for j in range(i, k):
                    new_sums.add(x + S_sorted[i] + S_sorted[j])
            
            if new_sums & used_sums:
                continue
            
            S.append(x)
            used_sums.update(new_sums)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    return best_set
# EVOLVE-BLOCK-END
