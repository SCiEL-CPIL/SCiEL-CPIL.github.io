import time
import math
import random
import numpy as np

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    n = len(U)
    
    # For small n, NumPy broadcasting is very fast
    # U[:, None] creates a column vector, U[None, :] creates a row vector
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    # Count unique elements
    # np.unique is O(N log N) on the flattened array of size N^2
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin
    
    # Helper to check time
    def time_left():
        return time.time() < deadline

    best_set = [0, 1, 3]
    best_score = fast_score(best_set)
    
    random.seed(42)

    # Phase 1: Generate diverse seeds
    seeds = []
    
    # Geometric progressions
    for m in range(2, 12):
        # b ranges from 1.1 to 10
        for b in [1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            U = [0]
            val = 1
            for k in range(1, m+1):
                val = int(round(b**k))
                if val in U:
                    continue
                U.append(val)
            if len(U) >= 2:
                seeds.append(U)
                
    # Powers of 2
    for m in range(2, 15):
        U = [1 << i for i in range(m)]
        seeds.append(U)
        
    # "Almost" geometric: add noise
    for base_set in seeds[:10]:
        U = list(base_set)
        # Add a random element
        if U:
            max_u = max(U)
            U.append(random.randint(1, max_u + 5))
        seeds.append(U)
        
    # New: "Binary Independence" style seeds
    # Select subsets of powers of 2 to create "sparse" structures
    # Or use a base-3 representation logic?
    # Let's try super-increasing sequences with random gaps
    for _ in range(20):
        U = [0]
        current = 1
        for _ in range(random.randint(3, 10)):
            # Randomly jump
            current += random.randint(1, 5)
            U.append(current)
        seeds.append(U)

    # Evaluate and sort seeds
    scored_seeds = []
    for U in seeds:
        U = sorted(set(U))
        if 0 not in U:
            U.insert(0, 0)
        s = fast_score(U)
        scored_seeds.append((s, U))
    
    scored_seeds.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 5 seeds
    top_seeds = [U for s, U in scored_seeds[:5]]
    
    # Update best
    if top_seeds:
        s, U = scored_seeds[0]
        if s > best_score:
            best_score = s
            best_set = U

    # Phase 2: Beam Search Construction
    # We will expand the best seeds by adding elements
    # Beam width = 10
    
    beam = [(best_score, best_set)]
    # Add other top seeds to beam
    for s, U in scored_seeds[1:5]:
        beam.append((s, U))
    
    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:10]
    
    max_size = 12 # Limit set size to keep computation fast
    
    for step in range(20): # 20 expansion steps
        if not time_left():
            break
            
        new_beam = []
        for score, U in beam:
            if len(U) >= max_size:
                new_beam.append((score, U))
                continue
                
            max_u = max(U) if U else 0
            # Candidate range: 1 to max_u + 10
            # To speed up, sample candidates or use a grid
            # Since we want to maximize ratio, we want to spread out.
            # Let's try all integers in a window? No, too many.
            # Try specific candidates:
            candidates = set()
            # Add small numbers
            for i in range(1, min(10, max_u + 1)):
                candidates.add(i)
            # Add numbers near max
            for i in range(max(1, max_u - 5), max_u + 11):
                candidates.add(i)
            # Add some randoms
            for _ in range(5):
                candidates.add(random.randint(1, max_u + 20))
            
            for c in candidates:
                if c in U:
                    continue
                new_U = U + [c]
                new_U.sort()
                s = fast_score(new_U)
                if s > -1:
                    new_beam.append((s, new_U))
        
        if not new_beam:
            break
            
        # Sort and prune
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:10]
        
        # Update global best
        if beam[0][0] > best_score:
            best_score = beam[0][0]
            best_set = beam[0][1]

    # Phase 3: Local Search (First Improvement)
    # Apply to the best set found
    U = list(best_set)
    current_score = best_score
    
    improved = True
    while improved and time_left():
        improved = False
        n = len(U)
        max_u = max(U)
        
        # 1. Try modifying existing elements
        for i in range(1, n): # Don't modify 0
            old_val = U[i]
            # Try nearby values
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 0:
                    continue
                if new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                s = fast_score(new_U)
                if s > current_score:
                    current_score = s
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        
        # 2. Try adding a new element
        if not improved and len(U) < max_size:
            # Search for best new element
            best_add_score = -1
            best_add_val = -1
            # Sample candidates
            candidates = set()
            for i in range(1, max_u + 15):
                candidates.add(i)
            for _ in range(10):
                candidates.add(random.randint(1, max_u + 50))
                
            for c in candidates:
                if c in U:
                    continue
                new_U = U + [c]
                new_U.sort()
                s = fast_score(new_U)
                if s > best_add_score:
                    best_add_score = s
                    best_add_val = c
            
            if best_add_score > current_score:
                current_score = best_add_score
                U.append(best_add_val)
                U.sort()
                improved = True
                
        # 3. Try removing an element
        if not improved and len(U) > 3:
            best_rem_score = -1
            best_rem_idx = -1
            for i in range(1, n): # Don't remove 0
                new_U = U[:i] + U[i+1:]
                s = fast_score(new_U)
                if s > best_rem_score:
                    best_rem_score = s
                    best_rem_idx = i
            
            if best_rem_score > current_score:
                current_score = best_rem_score
                U = U[:best_rem_idx] + U[best_rem_idx+1:]
                improved = True

        if current_score > best_score:
            best_score = current_score
            best_set = U[:]

    # Phase 4: Simulated Annealing
    U = np.array(best_set, dtype=np.int64)
    current_score = best_score
    
    T = 0.02
    T_min = 0.001
    cooling = 0.999
    
    for iteration in range(50000):
        if not time_left():
            break
            
        if T < T_min:
            T = T_min
            
        n = len(U)
        move_type = random.random()
        
        if move_type < 0.5 and n > 2:
            # Modify
            idx = random.randint(1, n-1)
            old_val = U[idx]
            # Propose new value
            # Gaussian noise
            new_val = max(0, int(old_val + random.gauss(0, max(1, old_val*0.1))))
            if new_val == old_val:
                continue
            new_U = U.copy()
            new_U[idx] = new_val
            if len(set(new_U)) != len(new_U):
                continue
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        elif move_type < 0.8 and n < max_size:
            # Add
            max_u = int(U.max())
            new_val = random.randint(1, max_u + 20)
            if new_val in U:
                continue
            new_U = np.append(U, new_val)
            new_U = np.sort(new_U)
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        else:
            # Remove
            if n > 3:
                idx = random.randint(1, n-1)
                new_U = np.delete(U, idx)
                s = fast_score(new_U)
                delta = s - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    U = new_U
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = sorted(U.tolist())
                        
        T *= cooling

    # Final Polish
    U = list(best_set)
    improved = True
    while improved and time_left():
        improved = False
        n = len(U)
        for i in range(1, n):
            old_val = U[i]
            for delta in range(-2, 3):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 0 or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        if improved:
            best_set = U[:]

    return best_set

if __name__ == "__main__":
    res = construct()
    print(compute_score(res))
