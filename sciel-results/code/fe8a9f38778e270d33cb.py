import time
import math
import random
import numpy as np

def compute_score(U):
    """Exact score calculation for verification."""
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
    """
    Fast scoring using numpy.
    U: list or numpy array of nonnegative integers.
    """
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    n = len(U)
    
    # Vectorized sum and diff
    # For very small n, this is very fast.
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    # Count unique elements
    # np.unique is O(N log N) where N = n^2
    n_sums = len(np.unique(sums.ravel()))
    n_diffs = len(np.unique(diffs.ravel()))
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def generate_sidon_like(n, max_val):
    """
    Greedily generate a Sidon-like set of size n containing 0.
    Sidon property: all pairwise sums are distinct (ignoring order).
    This minimizes |U+U| for a given set of values.
    """
    U = [0]
    used_sums = {0}
    for _ in range(n - 1):
        best_candidate = None
        best_min_sum = -1
        # Try candidates from 1 upwards, or random range
        # To keep max_val bounded, we pick from a range
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for c in candidates:
            if c in U:
                continue
            # Check if adding c creates any duplicate sums
            valid = True
            for u in U:
                s = u + c
                if s in used_sums:
                    valid = False
                    break
                used_sums.add(s)
            if valid:
                # Also check sums among existing elements are already in used_sums
                # We just need to ensure new sums (u+c) are not in used_sums
                # And that we don't add duplicates within the new set (handled by u loop)
                # Note: this greedy approach ensures all sums are distinct.
                best_candidate = c
                break # Take the first valid one found in random order to diversify
            
            # If invalid, we don't add to used_sums, so we don't need to undo
            # But wait, the loop above adds to used_sums. We need to be careful.
            # Let's re-implement check without side effects or with undo.
            
    # The above implementation is flawed because it modifies used_sums.
    # Let's do a cleaner version.
    
    U = [0]
    all_sums = {0}
    for _ in range(n - 1):
        found = False
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        for c in candidates:
            if c in U:
                continue
            new_sums = {u + c for u in U}
            # Check intersection with existing sums
            if new_sums & all_sums:
                continue
            # Also check if new_sums have internal duplicates?
            # u1 + c = u2 + c implies u1 = u2, which is impossible since U is a set.
            # So new_sums are distinct from each other.
            U.append(c)
            all_sums |= new_sums
            found = True
            break
        if not found:
            break
    return sorted(U)

def local_search(U, deadline, max_steps=200):
    """
    Local search with diverse moves: move, swap, add, remove.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.insert(0, 0)
        U = sorted(U)
    
    current_score = fast_score(U)
    if current_score <= 0:
        return U, current_score

    n = len(U)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        improved = False
        # Try a few random moves
        for _ in range(20):
            if time.time() > deadline:
                break
                
            move_type = random.random()
            
            if move_type < 0.4 and n > 2:
                # Move an element
                i = random.randint(1, n-1)
                old_val = U[i]
                # Try small moves
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val < 0 or new_val in U:
                        continue
                    new_U = U[:i] + [new_val] + U[i+1:]
                    new_U.sort()
                    s = fast_score(new_U)
                    if s > current_score:
                        U = new_U
                        current_score = s
                        n = len(U)
                        improved = True
                        break
                if improved:
                    break
                    
            elif move_type < 0.6 and n > 3:
                # Swap two elements
                i, j = random.sample(range(1, n), 2)
                new_U = U[:]
                new_U[i], new_U[j] = new_U[j], new_U[i]
                # Ensure sorted and unique (swap of distinct elements in a set keeps them distinct)
                # But we need to ensure 0 is still 0? No, 0 is just an element.
                # If we swap 0 with something, 0 moves. That's allowed.
                # But the problem says "containing 0". It doesn't say 0 must be at index 0.
                # However, our score function checks `if 0 not in U`.
                # So swapping is fine.
                s = fast_score(new_U)
                if s > current_score:
                    U = sorted(new_U)
                    current_score = s
                    improved = True
                    break
                    
            elif move_type < 0.8 and n < 12:
                # Add an element
                max_u = max(U)
                # Try values near existing ones or new ranges
                candidates = set()
                for u in U:
                    candidates.add(u + 1)
                    candidates.add(u + 2)
                    candidates.add(u * 2)
                candidates.add(max_u + 1)
                candidates.add(max_u + 2)
                candidates.add(random.randint(1, max_u + 10))
                
                for c in candidates:
                    if c in U or c < 0:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    s = fast_score(new_U)
                    if s > current_score:
                        U = new_U
                        current_score = s
                        n = len(U)
                        improved = True
                        break
                        
            else:
                # Remove an element
                if n > 3:
                    i = random.randint(1, n-1)
                    new_U = U[:i] + U[i+1:]
                    s = fast_score(new_U)
                    if s > current_score:
                        U = new_U
                        current_score = s
                        n = len(U)
                        improved = True
                        break
                        
        if not improved:
            # If no improvement, try a larger jump
            if n > 2:
                i = random.randint(1, n-1)
                old_val = U[i]
                # Large jump
                new_val = max(0, old_val + random.randint(-20, 20))
                if new_val != old_val and new_val not in U:
                    new_U = U[:i] + [new_val] + U[i+1:]
                    new_U.sort()
                    s = fast_score(new_U)
                    if s > current_score:
                        U = new_U
                        current_score = s
                        n = len(U)

    return U, current_score

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_set = [0, 1, 3, 7, 12] # A known good Golomb ruler
    best_score = fast_score(best_set)
    
    random.seed(42)
    
    # Phase 1: Generate diverse starting points
    starts = []
    
    # 1. Small Sidon-like sets
    for n in range(3, 10):
        for _ in range(5):
            U = generate_sidon_like(n, n*n*2)
            if len(U) >= 3:
                starts.append(U)
                
    # 2. Random sets
    for n in range(3, 10):
        for _ in range(10):
            max_val = random.randint(20, 100)
            U = [0]
            while len(U) < n:
                x = random.randint(1, max_val)
                if x not in U:
                    U.append(x)
            starts.append(sorted(U))
            
    # 3. Geometric
    for n in range(3, 8):
        for b in [1.5, 2.0, 2.5, 3.0]:
            U = [0, 1]
            cur = 1
            for _ in range(n-2):
                cur = int(cur * b)
                if cur not in U:
                    U.append(cur)
            starts.append(sorted(U))
            
    # 4. Known Golomb Rulers
    starts.append([0, 1, 4, 10, 18, 23, 29, 36])
    starts.append([0, 1, 4, 10, 18, 23, 29, 36, 44])
    starts.append([0, 1, 3, 7, 12, 20, 29, 36, 47, 56])
    
    # Phase 2: Local Search on each start
    for U in starts:
        if time.time() > deadline - 5:
            break
        U_opt, score = local_search(U, deadline)
        if score > best_score:
            best_score = score
            best_set = U_opt[:]
            
    # Phase 3: Simulated Annealing on the best set
    U = np.array(best_set, dtype=np.int64)
    current_score = best_score
    
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    
    for iteration in range(20000):
        if time.time() > deadline - 2:
            break
        
        if T < T_min:
            T = T_min
            
        n = len(U)
        move_type = random.random()
        
        if move_type < 0.5 and n > 2:
            idx = random.randint(0, n-1)
            old_val = U[idx]
            sigma = max(1, old_val * 0.2)
            new_val = max(0, int(old_val + random.gauss(0, sigma)))
            if new_val == old_val:
                continue
            new_U = U.copy()
            new_U[idx] = new_val
            if len(set(new_U)) != len(new_U):
                continue
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        elif move_type < 0.7 and n < 15:
            max_u = int(U.max())
            new_val = random.randint(1, max_u + 10)
            if new_val in U:
                continue
            new_U = np.append(U, new_val)
            new_U = np.sort(new_U)
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        elif move_type < 0.9 and n > 3:
            idx = random.randint(0, n-1)
            new_U = np.delete(U, idx)
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        else:
            # Swap
            if n > 3:
                i, j = random.sample(range(n), 2)
                new_U = U.copy()
                new_U[i], new_U[j] = new_U[j], new_U[i]
                s = fast_score(new_U)
                delta = s - current_score
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    U = new_U
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = sorted(U.tolist())

        T *= cooling

    # Final Polish
    U = list(best_set)
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        n = len(U)
        for i in range(n):
            old_val = U[i]
            for delta in range(-3, 4):
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
