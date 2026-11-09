import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 90s kill
    
    def is_valid_set(A):
        """Verify that a set satisfies the cardinality-separated subset sum property."""
        if not A:
            return True
        n = len(A)
        # S[k] = set of sums of all k-element subsets
        S = [set() for _ in range(n + 1)]
        S[0] = {0}
        for x in A:
            for k in range(n, 0, -1):
                if S[k-1]:
                    S[k].update(s + x for s in S[k-1])
        # Check: for any i != j, S[i] and S[j] must be disjoint
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if S[i] & S[j]:
                    return False
        return True
    
    def run_greedy(order, max_size=60):
        """
        Greedily build a set from 'order' such that for any two subsets of different
        cardinalities, their sums are distinct.
        
        We maintain S[k] = set of sums of all k-element subsets of the current set.
        When adding x, we check if (S[j-1] + x) intersects S[i] for any i != j.
        """
        S = [set() for _ in range(max_size + 1)]
        S[0] = {0}
        A = []
        current_size = 0
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if current_size >= max_size:
                break
            
            valid = True
            # Check validity: for each j from 1 to current_size+1
            # new sums are S[j-1] + x
            # These must not be in S[i] for any i != j
            
            for j in range(1, current_size + 2):
                if not S[j-1]:
                    continue
                
                # For this j, we need (S[j-1] + x) to be disjoint from all S[i] where i != j
                # Precompute the union of all S[i] for i != j (within range 0..current_size)
                # But building that union each time is O(n * |S|) which is expensive.
                # Instead: for each s in S[j-1], check if s+x is in any S[i] for i != j.
                # We can optimize by checking the largest sets first (more likely to contain the value).
                
                for s in S[j-1]:
                    candidate = s + x
                    # Check all i != j
                    found = False
                    for i in range(current_size + 1):
                        if i == j:
                            continue
                        if candidate in S[i]:
                            found = True
                            break
                    if found:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                A.append(x)
                current_size += 1
                # Update S
                for k in range(current_size, 0, -1):
                    if S[k-1]:
                        S[k].update(s + x for s in S[k-1])
        
        return A
    
    def try_extend(A, S, current_size, available, max_size=60):
        """Try to extend the current set A by finding elements from available that can be added."""
        extended = list(A)
        S_copy = [set(s) for s in S]
        size = current_size
        added = False
        
        for x in available:
            if time.time() - start_time > deadline:
                break
            if size >= max_size:
                break
            
            valid = True
            for j in range(1, size + 2):
                if not S_copy[j-1]:
                    continue
                for s in S_copy[j-1]:
                    candidate = s + x
                    found = False
                    for i in range(size + 1):
                        if i == j:
                            continue
                        if candidate in S_copy[i]:
                            found = True
                            break
                    if found:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                extended.append(x)
                size += 1
                added = True
                for k in range(size, 0, -1):
                    if S_copy[k-1]:
                        S_copy[k].update(s + x for s in S_copy[k-1])
        
        return extended, S_copy, size, added
    
    # Try several deterministic orders first
    orders = [
        list(range(1, 257)),  # ascending
        list(range(256, 0, -1)),  # descending
        # Alternating high-low
        [256 - i if i % 2 == 0 else i + 1 for i in range(256)],
        # Mid-out
        [128 + i if i % 2 == 0 else 128 - i for i in range(256)],
    ]
    
    best_set = []
    for order in orders:
        if time.time() - start_time > deadline:
            break
        candidate = run_greedy(order)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Randomized search with various strategies
    all_numbers = list(range(1, 257))
    
    for iteration in range(500):
        if time.time() - start_time > deadline:
            break
        
        # Different random strategies
        r = random.random()
        if r < 0.15:
            order = sorted(all_numbers)
        elif r < 0.30:
            order = sorted(all_numbers, reverse=True)
        elif r < 0.45:
            # Bias toward larger numbers
            order = sorted(all_numbers, key=lambda x: random.random() ** 0.3)
        elif r < 0.60:
            # Bias toward smaller numbers
            order = sorted(all_numbers, key=lambda x: random.random() ** 3)
        elif r < 0.75:
            # Partially sorted with noise
            order = list(all_numbers)
            random.shuffle(order)
            # Sort a random subset
            start = random.randint(0, 200)
            end = min(256, start + random.randint(20, 80))
            order[start:end] = sorted(order[start:end])
        else:
            order = list(all_numbers)
            random.shuffle(order)
        
        candidate = run_greedy(order)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Local improvement: try to add more elements to the best set
    if best_set and len(best_set) < 60:
        used = set(best_set)
        available = [x for x in range(1, 257) if x not in used]
        
        # Build S for the current best set
        n = len(best_set)
        S = [set() for _ in range(n + 1)]
        S[0] = {0}
        for x in best_set:
            for k in range(n, 0, -1):
                if S[k-1]:
                    S[k].update(s + x for s in S[k-1])
        
        # Try adding elements one by one
        extended, S_ext, size, added = try_extend(best_set, S, n, available)
        if len(extended) > len(best_set):
            best_set = extended
    
    # Verify the best set is valid
    if best_set and not is_valid_set(best_set):
        # Fallback to a simple valid set
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return [int(x) for x in best_set]
