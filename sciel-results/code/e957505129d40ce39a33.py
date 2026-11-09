import random
import time
import itertools

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_4_subsets(s):
    count = 0
    # Optimize: use a set for sums
    for combo in itertools.combinations(s, 4):
        sums = set()
        is_s = True
        for i in range(4):
            for j in range(i, 4):
                val = combo[i] + combo[j]
                if val in sums:
                    is_s = False
                    break
                sums.add(val)
            if not is_s:
                break
        if is_s:
            count += 1
    return count

def largest_sidon_subset_size(s):
    # We only care if it's 3 or 4 (or higher).
    # If count_sidon_4 is 0, max size is 3 (since any 3 non-AP is Sidon, and we have 12 elements, surely some 3 is Sidon? 
    # Actually, if all 3s are AP, then no Sidon 3? 
    # {0,1,2} is not Sidon. 
    # But we have 12 elements. It is impossible for ALL 3-subsets to be non-Sidon (AP).
    # So max size is at least 3.
    # If count_sidon_4 > 0, max size is at least 4.
    # We assume max size won't exceed 4 for our dense sets.
    
    c4 = count_sidon_4_subsets(s)
    if c4 == 0:
        return 3
    else:
        # Check for 5? 
        # If we have a Sidon 4, we might have a Sidon 5.
        # But for the purpose of minimizing, if we can get c4=0, we get 3.
        # If we can't, we are stuck with >= 4.
        # Let's just return 4 if c4 > 0 for simplicity in the search objective,
        # unless we find a 5, in which case it's worse.
        # But the search minimizes c4. Low c4 is better.
        return 4 

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Candidate 1: AP
    cand_ap = list(range(12))
    best_set = cand_ap
    best_score = count_sidon_4_subsets(cand_ap)
    
    # Candidate 2: Random Dense (clusters)
    # Try a few random dense sets
    for _ in range(5):
        # Generate a set with small gaps
        base = random.randint(0, 100)
        # Pick 12 numbers from a small range
        span = random.randint(10, 30)
        candidates = set()
        while len(candidates) < 12:
            candidates.add(base + random.randint(0, span))
        s = list(candidates)
        score = count_sidon_4_subsets(s)
        if score < best_score:
            best_score = score
            best_set = s
    
    # Local Search
    current_set = set(best_set)
    current_score = best_score
    
    # If we already found 0, we can stop or try to verify.
    if current_score == 0:
        return list(current_set)
        
    while time.time() < deadline:
        # Pick a random element to swap
        idx = random.randint(0, 11)
        old_val = list(current_set)[idx]
        
        # Pick a new value
        # Bias towards values close to existing ones to create collisions
        # Or just random
        if random.random() < 0.5:
            # Close to old_val
            new_val = old_val + random.choice([-2, -1, 1, 2])
        else:
            new_val = random.randint(0, 128)
            
        if new_val in current_set and new_val != old_val:
            continue
            
        new_set = list(current_set)
        new_set[idx] = new_val
        new_set = set(new_set)
        
        new_score = count_sidon_4_subsets(list(new_set))
        
        if new_score < current_score:
            current_score = new_score
            current_set = new_set
            if current_score < best_score:
                best_score = current_score
                best_set = list(current_set)
                
            if best_score == 0:
                break
        else:
            # Occasionally accept worse to escape local minima?
            # Or restart?
            if random.random() < 0.05:
                # Random restart with dense bias
                base = random.randint(0, 100)
                span = random.randint(10, 30)
                candidates = set()
                while len(candidates) < 12:
                    candidates.add(base + random.randint(0, span))
                current_set = candidates
                current_score = count_sidon_4_subsets(list(current_set))
                if current_score < best_score:
                    best_score = current_score
                    best_set = list(current_set)
                    
    return best_set
