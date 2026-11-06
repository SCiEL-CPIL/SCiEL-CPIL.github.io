import time
import random
from itertools import combinations

def is_sidon(S):
    """Check if a list of integers is a Sidon set (unique pair sums including diagonals)."""
    sums = set()
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    # Check from largest possible down to smallest
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            if is_sidon(subset):
                return k
    return 2 # Any 2 distinct elements form a Sidon set

def count_conflicts(S):
    """Count the number of unordered pair-sum collisions in S.
    Used as a secondary metric to guide search when max_sidon_size is tied."""
    sums = {}
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    total = 0
    for c in sums.values():
        if c > 1:
            total += c * (c - 1) // 2
    return total

def count_four_sidon(ambient):
    """Count how many 4-element subsets ARE Sidon."""
    n = len(ambient)
    count = 0
    for subset in combinations(range(n), 4):
        vals = [ambient[i] for i in subset]
        if is_sidon(vals):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Initial candidates
    # Strategy: Start with dense sets in small ranges, as they tend to have smaller max Sidon sizes
    initial_sets = []
    
    # 1. Random dense sets in [0, 20]
    for _ in range(10):
        initial_sets.append(sorted(random.sample(range(0, 21), 12)))
    
    # 2. Arithmetic progressions with small step
    for d in [1, 2, 3]:
        for a in range(0, 128 - 11*d + 1, 5):
            initial_sets.append(sorted([a + i*d for i in range(12)]))
    
    # 3. Random sets in [0, 30]
    for _ in range(10):
        initial_sets.append(sorted(random.sample(range(0, 31), 12)))
    
    # 4. Structured candidates from inspiration 2: two tight clusters
    initial_sets.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    initial_sets.append([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
    initial_sets.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    
    # Evaluate initial candidates
    best_set = None
    best_score = 13 # Max possible is 12, so 13 is safe init
    best_conflicts = -1
    best_four_count = 999999
    
    for S0 in initial_sets:
        if time.time() > deadline:
            break
        score = max_sidon_size(S0)
        four_count = count_four_sidon(S0)
        if score < best_score:
            best_score = score
            best_set = S0[:]
            best_conflicts = count_conflicts(S0)
            best_four_count = four_count
        elif score == best_score:
            # Tie-break: prefer higher conflicts (harder set) and lower four-Sidon count
            c = count_conflicts(S0)
            if c > best_conflicts or (c == best_conflicts and four_count < best_four_count):
                best_conflicts = c
                best_set = S0[:]
                best_four_count = four_count

    if best_set is None:
        # Fallback
        best_set = sorted(random.sample(range(0, 21), 12))
        best_score = max_sidon_size(best_set)
        best_conflicts = count_conflicts(best_set)
        best_four_count = count_four_sidon(best_set)

    S = best_set[:]
    current_score = best_score
    current_conflicts = best_conflicts
    current_four_count = best_four_count
    
    universe = list(range(129))
    n = 12
    
    # Local Search
    no_improve_count = 0
    max_no_improve = 200 # Lowered to allow more restarts
    
    while time.time() < deadline:
        # Pick a random element to swap out
        idx = random.randrange(n)
        old_val = S[idx]
        
        # Pick a random replacement from universe not in S
        # Bias towards dense regions to maintain high collision density
        if random.random() < 0.7:
            new_val = random.randint(0, 30)
        else:
            new_val = random.choice(universe)
            
        if new_val in S:
            continue
            
        # Perform swap
        S[idx] = new_val
        new_S = sorted(S)
        
        # Evaluate
        new_score = max_sidon_size(new_S)
        
        if new_score < current_score:
            # Improvement in primary metric
            current_score = new_score
            current_conflicts = count_conflicts(new_S)
            current_four_count = count_four_sidon(new_S)
            best_set = new_S[:]
            best_score = current_score
            best_conflicts = current_conflicts
            best_four_count = current_four_count
            no_improve_count = 0
        elif new_score == current_score:
            # Tie in primary metric. Check secondary metrics
            new_conflicts = count_conflicts(new_S)
            new_four_count = count_four_sidon(new_S)
            
            # Prefer higher conflicts, then lower four-Sidon count
            if (new_conflicts > current_conflicts) or \
               (new_conflicts == current_conflicts and new_four_count < current_four_count):
                # Better secondary metric
                current_conflicts = new_conflicts
                current_four_count = new_four_count
                best_set = new_S[:]
                best_conflicts = current_conflicts
                best_four_count = current_four_count
                no_improve_count = 0
            else:
                # Revert if no improvement in either metric
                S[idx] = old_val
                no_improve_count += 1
        else:
            # Worse primary metric. Revert.
            S[idx] = old_val
            no_improve_count += 1
        
        # Restart logic
        if no_improve_count > max_no_improve:
            # Restart from a random dense set
            S = sorted(random.sample(range(0, 25), 12))
            current_score = max_sidon_size(S)
            current_conflicts = count_conflicts(S)
            current_four_count = count_four_sidon(S)
            
            if current_score < best_score:
                best_score = current_score
                best_set = S[:]
                best_conflicts = current_conflicts
                best_four_count = current_four_count
            elif current_score == best_score:
                if current_conflicts > best_conflicts or \
                   (current_conflicts == best_conflicts and current_four_count < best_four_count):
                    best_conflicts = current_conflicts
                    best_set = S[:]
                    best_four_count = current_four_count
            no_improve_count = 0

    return sorted(best_set)
