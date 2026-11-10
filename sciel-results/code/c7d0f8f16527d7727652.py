"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    def count_sum_collisions(arr):
        """Count number of sum collisions. Lower = more Sidon-like.
        We want to MAXIMIZE collisions to MINIMIZE max Sidon subset."""
        sums = {}
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                s = arr[i] + arr[j]
                if s in sums:
                    sums[s] += 1
                else:
                    sums[s] = 1
        # Count collisions: for each sum appearing k times, that's C(k,2) collision pairs
        collisions = 0
        for k in sums.values():
            if k > 1:
                collisions += k * (k - 1) // 2
        return collisions
    
    def is_sidon(subset):
        """Check if subset has all pair sums unique (including diagonals)."""
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_size(arr):
        """Find the largest Sidon subset size in arr using backtracking."""
        n = len(arr)
        best_size = 0
        arr_sorted = sorted(arr)
        
        def backtrack(idx, current, current_sums):
            nonlocal best_size
            if len(current) > best_size:
                best_size = len(current)
            
            # Pruning
            if len(current) + (n - idx) <= best_size:
                return

            for i in range(idx, n):
                new_elem = arr_sorted[i]
                # Check if adding new_elem keeps Sidon property
                # Need to check: new_elem + new_elem, and new_elem + current[k] for all k
                # against existing sums in current_sums, and among themselves
                new_sums = set(current_sums)
                ok = True
                # Check diagonal
                diag = new_elem + new_elem
                if diag in new_sums:
                    ok = False
                else:
                    new_sums.add(diag)
                # Check cross sums
                if ok:
                    for k in range(len(current)):
                        s = new_elem + current[k]
                        if s in new_sums:
                            ok = False
                            break
                        new_sums.add(s)
                
                if ok:
                    backtrack(i + 1, current + [new_elem], new_sums)

        backtrack(0, [], set())
        return best_size
    
    def score_candidate(arr):
        """Lower is better: the max Sidon subset size."""
        return max_sidon_size(arr)
    
    # Generate diverse initial candidates using AP-based and modular structures
    def generate_candidate(seed_val):
        random.seed(seed_val)
        # Strategy: use clusters to create sum collisions
        # Try different cluster sizes and spacings
        strategy = random.randint(0, 3)
        if strategy == 0:
            # Two APs with overlapping sum ranges
            a1 = random.randint(0, 20)
            d1 = random.randint(1, 3)
            n1 = random.randint(4, 7)
            a2 = random.randint(30, 60)
            d2 = random.randint(1, 3)
            n2 = 12 - n1
            cand = [a1 + i * d1 for i in range(n1)] + [a2 + i * d2 for i in range(n2)]
        elif strategy == 1:
            # One long AP
            a = random.randint(0, 30)
            d = random.randint(1, 2)
            cand = [a + i * d for i in range(12)]
        elif strategy == 2:
            # Three clusters
            a1 = random.randint(0, 15)
            a2 = random.randint(25, 45)
            a3 = random.randint(60, 90)
            d1 = random.randint(1, 2)
            d2 = random.randint(1, 2)
            d3 = random.randint(1, 2)
            n1 = random.randint(3, 5)
            n2 = random.randint(3, 5)
            n3 = 12 - n1 - n2
            cand = ([a1 + i * d1 for i in range(n1)] + 
                    [a2 + i * d2 for i in range(n2)] + 
                    [a3 + i * d3 for i in range(n3)])
        else:
            # Random with bias toward small values
            cand = [random.randint(0, 20) for _ in range(12)]
        
        # Ensure distinct and in range
        cand = sorted(cand)
        # Make distinct
        for i in range(1, 12):
            if cand[i] <= cand[i-1]:
                cand[i] = cand[i-1] + 1
        # Clamp to [0, 128]
        if cand[-1] > 128:
            shift = cand[-1] - 128
            cand = [max(0, x - shift) for x in cand]
            # Re-ensure distinct
            for i in range(1, 12):
                if cand[i] <= cand[i-1]:
                    cand[i] = cand[i-1] + 1
        if cand[-1] > 128:
            cand = list(range(12))
        return cand
    
    # Evaluate initial candidates
    candidates = []
    for seed_val in range(20):
        candidates.append(generate_candidate(seed_val))
    # Add some hand-crafted
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    candidates.append([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23])
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    candidates.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    
    best_candidate = None
    best_score = 12
    
    for cand in candidates:
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        s = score_candidate(cand)
        if s < best_score:
            best_score = s
            best_candidate = cand[:]
    
    if best_candidate is None:
        best_candidate = list(range(12))
        best_score = 12
    
    # Simulated annealing
    random.seed(42)
    current = best_candidate[:]
    current_collisions = count_sum_collisions(current)
    T = 10.0
    cooling = 0.9995
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Pick random index
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Generate new value
        new_val = random.randint(0, 128)
        if new_val == old_val or new_val in current:
            continue
        
        current[idx] = new_val
        new_collisions = count_sum_collisions(current)
        delta_coll = new_collisions - current_collisions  # We want to increase collisions
        
        # Accept if better (more collisions) or with probability
        if delta_coll >= 0:
            current_collisions = new_collisions
            # Check exact score
            s = score_candidate(current)
            if s < best_score:
                best_score = s
                best_candidate = current[:]
        else:
            # Accept with probability exp(-delta/T) where delta is negative
            # We want to accept moves that increase collisions
            prob = min(1.0, (2.718281828 ** (delta_coll / max(T, 0.01))))
            if random.random() < prob:
                current_collisions = new_collisions
            else:
                current[idx] = old_val
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Final polish: local search around best_candidate
    random.seed(123)
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        cand = best_candidate[:]
        idx = random.randint(0, 11)
        old_val = cand[idx]
        
        for _ in range(5):
            new_val = random.randint(0, 128)
            if new_val == old_val or new_val in cand:
                continue
            
            cand[idx] = new_val
            s = score_candidate(cand)
            
            if s < best_score:
                best_score = s
                best_candidate = cand[:]
                break
            else:
                cand[idx] = old_val
    
    return best_candidate
# EVOLVE-BLOCK-END
