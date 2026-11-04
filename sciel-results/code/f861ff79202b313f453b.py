import time
import random
import math
import numpy as np

def _score(U):
    """
    Compute the score for a given set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    if not U:
        return 0.0
    if 0 not in U:
        # Invalid set for the problem definition, but we ensure 0 is always present
        return -1.0
        
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 1.0 # Degenerate case, but max(U)=0 -> log(1)=0, undefined. 
                   # Actually if U={0}, max=0, denom=0. 
                   # Problem says containing 0. If U={0}, |U-U|=1, |U+U|=1, ratio=1.
                   # log(1)/log(1) is 0/0. 
                   # Let's assume U has at least 2 elements or handle max=0 separately.
                   # If max(U)=0, U={0}. Score is typically defined as limit or 1.
                   # The formula 1 + log(1)/log(1) is indeterminate.
                   # However, if U={0, 1}, max=1, denom=log(3).
                   # |U-U|={-1,0,1} size 3. |U+U|={0,1,2} size 3. Ratio 1. Score 1.
                   # So we return 1.0 for trivial cases.
    
    max_u = max(U)
    if max_u == 0:
        return 1.0

    u_arr = np.array(U, dtype=np.int64)
    
    # Compute sumset and diffset sizes
    # Sums: u_i + u_j
    sums = u_arr[:, None] + u_arr[None, :]
    # Diffs: u_i - u_j
    diffs = u_arr[:, None] - u_arr[None, :]
    
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    
    if sum_count == 0:
        return 1.0
        
    ratio = diff_count / sum_count
    
    if ratio <= 1.0:
        # If ratio <= 1, the term is <= 0, so score <= 1.
        # We want to maximize, so these are not good candidates for improvement over 1.
        # But we return the actual value to be precise.
        pass
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _generate_candidates(U, max_val_limit=10000):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    u_list = list(U)
    n = len(u_list)
    
    for i in range(n):
        for j in range(n):
            s = u_list[i] + u_list[j]
            d = abs(u_list[i] - u_list[j])
            
            if 0 < s <= max_val_limit and s not in U_set:
                candidates.add(s)
            if 0 < d <= max_val_limit and d not in U_set:
                candidates.add(d)
                
    return list(candidates)

def _local_search(start_set, deadline):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(start_set))
    if 0 not in U:
        U.append(0)
        U.sort()
        
    current_score = _score(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    # Set a reasonable limit for values to keep the search space manageable
    # The max value should not grow too large as it hurts the denominator
    max_val_limit = max(max(U) * 3 + 50, 200)
    
    candidates = _generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing, but cap it to avoid huge numbers
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2 + 100)
            candidates = _generate_candidates(U, new_limit)
            if len(candidates) > 1000:
                random.shuffle(candidates)
                candidates = candidates[:1000]

        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < 40 and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2+50))
                        no_improvement_count = 0
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max_val_limit)
                        no_improvement_count = 0
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2+50))
                        no_improvement_count = 0
                        break
        
        if improved:
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
            # If stuck, try a random kick
            if no_improvement_count % 10 == 0:
                # Add a random element to escape local optimum
                if len(U) < 40:
                    kick_val = random.randint(1, max_val_limit)
                    if kick_val not in U:
                        U = U + [kick_val]
                        U.sort()
                        current_score = _score(U)
                        candidates = _generate_candidates(U, max_val_limit)

    return best_local_set, best_local_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_set = [0, 1]
    best_score = _score(best_set)
    
    # Phase 1: Generate diverse starting points
    starts = []
    
    # 1. Small exhaustive search for size 3-4
    # This is fast and finds good local structures
    max_val_small = 30
    for n in range(3, 5):
        if time.time() > deadline - 30:
            break
        # Use itertools for combinations
        import itertools
        for combo in itertools.combinations(range(1, max_val_small + 1), n - 1):
            if time.time() > deadline - 30:
                break
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U
            # Keep top few for restarts
            starts.append(U)
            
    if len(starts) > 10:
        # Keep only the best 10 to avoid too many restarts
        starts.sort(key=lambda x: _score(x), reverse=True)
        starts = starts[:10]
    else:
        # If exhaustive didn't yield many, add some geometric ones
        starts.append([0, 1, 2, 4])
        starts.append([0, 1, 3, 9])
        starts.append([0, 1, 4, 16])
        starts.append([0, 1, 2, 5, 10])
        
    # Ensure best_set is in starts
    if best_set not in starts:
        starts.append(best_set)

    # Phase 2: Local search from each start
    for start in starts:
        if time.time() > deadline - 5:
            break
        
        s_set, s_score = _local_search(start, deadline)
        if s_score > best_score:
            best_score = s_score
            best_set = s_set

    return best_set

# Test/Validation
if __name__ == "__main__":
    U = construct()
    print(f"Result: {U}")
    print(f"Score: {_score(U)}")
