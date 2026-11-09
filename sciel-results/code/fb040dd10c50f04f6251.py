import random
import time
import itertools

def is_sidon(subset):
    """Check if a subset is Sidon (all pair sums including diagonals are unique)."""
    n = len(subset)
    if n <= 1:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s_val = subset[i] + subset[j]
            if s_val in sums:
                return False
            sums.add(s_val)
    return True

def largest_sidon_subset_size(s):
    """Find the size of the largest Sidon subset of s."""
    s = sorted(s)
    n = len(s)
    # Try all subsets in descending order of size
    for size in range(n, 0, -1):
        for combo in itertools.combinations(range(n), size):
            subset = [s[i] for i in combo]
            if is_sidon(subset):
                return size
    return 0

def count_distinct_sums(s):
    """Count the number of distinct pair sums. Minimizing this maximizes collisions."""
    n = len(s)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(s[i] + s[j])
    return len(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    best_set = None
    best_score = float('inf')
    
    def update_best(s, score):
        nonlocal best_set, best_score
        if score < best_score:
            best_score = score
            best_set = list(s)
            if best_score <= 3:  # Can't do much better than 3
                return True
        return False

    random.seed(42)
    
    # Try structured initializations
    # APs are very collision-heavy
    candidates = []
    for start in range(0, 129, 5):
        for step in [1, 2, 3]:
            cand = [start + i*step for i in range(12)]
            if all(0 <= x <= 128 for x in cand) and len(set(cand)) == 12:
                candidates.append(cand)
    
    # Add some random structured sets
    for _ in range(10):
        base = random.randint(0, 50)
        step = random.randint(1, 3)
        cand = [base + i*step + random.randint(-2, 2) for i in range(12)]
        # Ensure distinct and in range
        cand = [max(0, min(128, c)) for c in cand]
        if len(set(cand)) == 12:
            candidates.append(cand)

    for cand in candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            score = largest_sidon_subset_size(cand)
            if update_best(cand, score):
                if best_score <= 3:
                    return best_set
    
    # Random search with local improvement using surrogate for speed
    # Surrogate: Minimize number of distinct pair sums (maximize collisions)
    
    for restart in range(300):
        if time.time() > deadline:
            break
            
        # Initialization: Start with a random set or a perturbed AP
        if random.random() < 0.5 and candidates:
            current_set = sorted(random.choice(candidates))
        else:
            current_set = sorted(random.sample(range(129), 12))
        
        current_score = largest_sidon_subset_size(current_set)
        current_surrogate = count_distinct_sums(current_set)
        
        if update_best(current_set, current_score):
            if best_score <= 3:
                return best_set
        
        # Local search
        T = 1.0
        cooling_rate = 0.99
        min_T = 0.001
        no_improve = 0
        
        while T > min_T and time.time() < deadline and no_improve < 100:
            # Generate neighbor
            new_set = current_set.copy()
            move_type = random.random()
            
            if move_type < 0.3:
                # Shift entire set
                shift = random.randint(-5, 5)
                new_set = [x + shift for x in new_set]
                # Clamp and fix duplicates if any
                new_set = [max(0, min(128, x)) for x in new_set]
                if len(set(new_set)) != 12:
                    no_improve += 1
                    T *= cooling_rate
                    continue
            elif move_type < 0.8:
                # Replace one element
                idx = random.randint(0, 11)
                old_val = new_set[idx]
                if random.random() < 0.5:
                    offset = random.randint(-10, 10)
                    new_val = max(0, min(128, old_val + offset))
                else:
                    new_val = random.randint(0, 128)
                
                if new_val != old_val and new_val not in new_set:
                    new_set[idx] = new_val
                else:
                    no_improve += 1
                    T *= cooling_rate
                    continue
            else:
                # Replace two elements
                idx1, idx2 = random.sample(range(12), 2)
                valid = True
                for idx in [idx1, idx2]:
                    old_val = new_set[idx]
                    if random.random() < 0.5:
                        offset = random.randint(-10, 10)
                        new_val = max(0, min(128, old_val + offset))
                    else:
                        new_val = random.randint(0, 128)
                    
                    if new_val == old_val or new_val in new_set:
                        valid = False
                        break
                    new_set[idx] = new_val
                
                if not valid:
                    no_improve += 1
                    T *= cooling_rate
                    continue
            
            new_set.sort()
            new_surrogate = count_distinct_sums(new_set)
            
            # Acceptance criterion based on surrogate
            if new_surrogate < current_surrogate:
                # Surrogate improved, check exact score
                new_score = largest_sidon_subset_size(new_set)
                if new_score < current_score:
                    current_set = new_set
                    current_score = new_score
                    current_surrogate = new_surrogate
                    no_improve = 0
                    
                    if update_best(current_set, current_score):
                        if best_score <= 3:
                            return best_set
                else:
                    # Surrogate improved but exact didn't, accept with prob
                    if random.random() < 0.5:
                        current_set = new_set
                        current_score = new_score
                        current_surrogate = new_surrogate
                        no_improve = 0
                    else:
                        no_improve += 1
            else:
                # Surrogate got worse or same
                delta_surrogate = new_surrogate - current_surrogate
                if random.random() < pow(2.718, -delta_surrogate / max(T, 1e-9)):
                    # Accept worse surrogate with low prob to escape local minima
                    new_score = largest_sidon_subset_size(new_set)
                    if new_score < current_score:
                        current_set = new_set
                        current_score = new_score
                        current_surrogate = new_surrogate
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            T *= cooling_rate
    
    return best_set
