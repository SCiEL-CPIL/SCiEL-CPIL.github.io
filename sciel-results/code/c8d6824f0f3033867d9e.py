import math
import random
import time

def _score_set(U):
    """Compute the score for a given set U."""
    U = sorted(U)
    if not U or U[0] != 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute |U+U| (sumset)
    # U+U = {a+b | a in U, b in U}
    # Since U contains 0, min is 0, max is 2*max_u
    # Using sets for correctness
    sum_set = set()
    for i in range(len(U)):
        ui = U[i]
        for j in range(i, len(U)):
            sum_set.add(ui + U[j])
            
    # Compute |U-U| (difference set)
    # U-U = {a-b | a in U, b in U}
    diff_set = set()
    for i in range(len(U)):
        ui = U[i]
        for j in range(len(U)):
            diff_set.add(ui - U[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    # The formula: 1 + log(ratio) / log(2 max U + 1)
    # Note: if ratio < 1, log(ratio) is negative, score < 1.
    # We want to maximize this.
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    start_time = time.time()
    time_limit = 100  # Keep within 120s limit
    
    # Candidate sets to evaluate
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # 1. Deterministic Seeds based on known structures
    # The prompt mentions constructions of size 2003 and 54265 in the paper.
    # These are likely related to specific combinatorial designs.
    # Common structures for sum/difference sets:
    # - Powers of 2: {0, 1, 2, 4, ...} -> |U+U| ~ N^2/2, |U-U| ~ N^2/2. Ratio ~ 1.
    #   Score ~ 1 + 0 = 1.
    # - Sidon sets / Golomb rulers: |U+U| is large (close to N(N-1)/2 + N), |U-U| is large.
    #   Actually, for Sidon sets, sums are distinct for pairs i<=j. So |U+U| = N(N+1)/2.
    #   Differences: for i!=j, differences are distinct? No, but |U-U| is roughly N(N-1)+1.
    #   Ratio ~ (N^2) / (N^2/2) = 2.
    #   Score ~ 1 + log(2) / log(2 max U + 1).
    #   If max U is small (dense Sidon), score is high.
    #   If max U is large (sparse Sidon), score is low.
    #   We want dense sets with high difference-to-sum ratio.
    
    # Let's try to construct sets that are "dense" in [0, M] but have high difference variety.
    # Actually, if U is an interval {0, ..., M}, |U+U| = 2M+1, |U-U| = 2M+1. Ratio = 1. Score = 1.
    # If U is {0, 1, ..., M} \ {some elements}, we might increase |U-U| relative to |U+U|?
    # Removing elements from a dense set usually decreases both.
    
    # Consider U = {0, 1, 3, 7, ...} (Powers of 2 minus 1? No, 2^k - 1)
    # {0, 1, 3, 7, 15, 31}
    # Sums: distinct?
    # 0+15=15, 1+14(no), 3+12(no), 7+8(no).
    # Binary representation suggests sums of 2^k-1 are not all distinct.
    # 1+3 = 4, 7+?
    # Let's just generate random sets and hill climb.
    
    # Strategy:
    # 1. Start with a few good seeds.
    # 2. Perform local search (add, remove, swap elements).
    # 3. Use multiple random restarts.
    
    seeds = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 2, 5, 10],
        [0, 1, 4, 10, 20],
    ]
    
    # Add some random seeds
    random.seed(42)
    for _ in range(10):
        N = random.randint(3, 8)
        max_val = random.randint(10, 50)
        s = random.sample(range(1, max_val + 1), N - 1)
        seeds.append([0] + sorted(s))
    
    # Evaluate seeds
    for s in seeds:
        s = sorted(set(s)) # Ensure unique and sorted
        if 0 not in s:
            s.insert(0, 0)
        sc = _score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s[:]
            
    # Local Search / Hill Climbing
    # We will try to improve the current best set.
    # Moves:
    # - Add a random number x
    # - Remove a random number x (if len > 1)
    # - Replace a random number x with y
    
    max_iter = 5000
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 200
    
    current_set = best_set[:]
    current_score = best_score
    
    # We need to be careful with the time limit.
    while iter_count < max_iter and (time.time() - start_time) < time_limit and no_improve_count < max_no_improve:
        iter_count += 1
        
        # Choose a move type
        move_type = random.choice(['add', 'remove', 'replace'])
        
        new_set = current_set[:]
        
        if move_type == 'add':
            # Add a number
            max_val = max(current_set) if current_set else 0
            # Try adding a number near existing ones or a bit larger
            # Range: 1 to max_val * 2 or max_val + 10
            upper_bound = max(max_val * 2, max_val + 10, 20)
            x = random.randint(1, upper_bound)
            if x not in new_set:
                new_set.append(x)
                new_set = sorted(new_set)
            else:
                continue # Skip if duplicate
            
        elif move_type == 'remove':
            if len(current_set) <= 1:
                continue
            idx = random.randint(1, len(current_set) - 1) # Don't remove 0
            new_set.pop(idx)
            
        elif move_type == 'replace':
            if len(current_set) <= 1:
                continue
            idx = random.randint(1, len(current_set) - 1) # Don't replace 0
            old_val = new_set[idx]
            max_val = max(current_set)
            upper_bound = max(max_val * 2, max_val + 10, 20)
            new_val = random.randint(1, upper_bound)
            if new_val in new_set:
                continue
            new_set[idx] = new_val
            new_set = sorted(new_set)
            
        # Evaluate new set
        if not new_set or new_set[0] != 0:
            continue
            
        new_score = _score_set(new_set)
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            no_improve_count = 0
            
            # Update global best
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            no_improve_count += 1
            
            # Occasionally, if stuck, try a larger jump or reset to a random seed
            if no_improve_count % 50 == 0:
                # Try a random restart with a slightly different seed
                if random.random() < 0.1:
                    N = random.randint(3, 8)
                    max_val = random.randint(10, 100)
                    s = random.sample(range(1, max_val + 1), N - 1)
                    new_seed = [0] + sorted(s)
                    sc = _score_set(new_seed)
                    if sc > current_score:
                        current_set = new_seed
                        current_score = sc
                        no_improve_count = 0
                    else:
                        # Keep current, but maybe try a more aggressive move next time
                        pass

    # Final check: ensure best_set is valid
    if not best_set or 0 not in best_set:
        return [0, 1, 3]
        
    return best_set

# To allow quick testing or if the evaluator imports specific functions
if __name__ == "__main__":
    s = construct()
    print(s)
    print(_score_set(s))
