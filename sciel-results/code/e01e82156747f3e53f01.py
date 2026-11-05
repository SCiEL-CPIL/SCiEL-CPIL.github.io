import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    n = 256
    
    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i+1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(x, S):
        """Check if adding x to set S maintains the property."""
        S_new = S | {x}
        S_sorted = sorted(S_new)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i+1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add_fast(x, S_set, S_sorted):
        """Faster check: only check triples involving x."""
        S_new = S_set | {x}
        S_sorted_new = sorted(S_new)
        m = len(S_sorted_new)
        # Find position of x
        x_idx = S_sorted_new.index(x)
        for i in range(m):
            a = S_sorted_new[i]
            for j in range(i+1, m):
                b = S_sorted_new[j]
                for k in range(j, m):
                    c = S_sorted_new[k]
                    # Only check if x is involved
                    if x in (a, b, c):
                        if (b + c) % a == 0:
                            return False
        return True
    
    # Start with upper tail
    best_set = set(range(192, 257))
    best_size = len(best_set)
    
    # Try to add elements from below
    candidates = list(range(1, 192))
    random.seed(42)
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline - 5:
            break
        if can_add(x, best_set):
            best_set.add(x)
            best_size = len(best_set)
    
    # Local search: try to swap out elements to add more
    all_elements = list(range(1, 257))
    random.shuffle(all_elements)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in all_elements:
            if time.time() > deadline - 5:
                break
            if x in best_set:
                continue
            # Try adding x
            if can_add(x, best_set):
                best_set.add(x)
                best_size = len(best_set)
                improved = True
            else:
                # Try swapping: remove one element y, add x
                for y in list(best_set):
                    if time.time() > deadline - 5:
                        break
                    if y >= x:
                        continue  # prefer to keep larger elements? Actually try all
                    test_set = (best_set - {y}) | {x}
                    if can_add(x, test_set):
                        best_set = test_set
                        improved = True
                        break
    
    # Try multiple random restarts with different initial seeds
    if time.time() < deadline - 10:
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            random.seed(seed * 1000 + 7)
            trial_set = set(range(192, 256))  # slightly smaller tail to allow more additions
            candidates2 = list(range(1, 192))
            random.shuffle(candidates2)
            for x in candidates2:
                if time.time() > deadline - 5:
                    break
                if can_add(x, trial_set):
                    trial_set.add(x)
            if len(trial_set) > best_size:
                best_set = trial_set
                best_size = len(trial_set)
    
    return sorted(best_set)
