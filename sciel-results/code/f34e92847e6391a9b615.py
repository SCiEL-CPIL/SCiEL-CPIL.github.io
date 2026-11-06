import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def is_sidon(vals):
        """Check if vals is a Sidon set (all pair sums a_i+a_j for i<=j are unique)."""
        sums = set()
        n = len(vals)
        for i in range(n):
            vi = vals[i]
            for j in range(i, n):
                s = vi + vals[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def surrogate_score(ambient):
        """
        Returns a tuple to be minimized.
        (has_6, count_5, count_4)
        """
        n = len(ambient)
        c6 = 0
        c5 = 0
        c4 = 0
        
        # Check 6
        for subset in itertools.combinations(range(n), 6):
            if is_sidon([ambient[i] for i in subset]):
                c6 += 1
                if c6 >= 5: # Early exit, it's bad
                    return (1, c6, 0)
        if c6 > 0:
            return (1, c6, 0)
            
        # Check 5
        for subset in itertools.combinations(range(n), 5):
            if is_sidon([ambient[i] for i in subset]):
                c5 += 1
                if c5 >= 50: # Cap
                    break
        if c5 > 0:
            return (0, c5, 0)
            
        # Check 4
        for subset in itertools.combinations(range(n), 4):
            if is_sidon([ambient[i] for i in subset]):
                c4 += 1
                if c4 >= 100: # Cap
                    break
        return (0, 0, c4)

    def true_max_sidon(ambient):
        """Returns the actual max Sidon size."""
        n = len(ambient)
        for k in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), k):
                if is_sidon([ambient[i] for i in subset]):
                    found = True
                    break
            if found:
                return k
        return 0

    # Initial Candidates
    rng = random.Random(42)
    candidates = []
    
    # Dense sets
    candidates.append(list(range(12)))
    candidates.append(list(range(0, 12)))
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Two clusters
    candidates.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Random
    for _ in range(100):
        candidates.append(sorted(rng.sample(range(129), 12)))
        
    best_score = None
    best_ambient = None
    
    for cand in candidates:
        if time.time() > deadline:
            break
        score = surrogate_score(cand)
        if best_score is None or score < best_score:
            best_score = score
            best_ambient = list(cand)
            
    # Local Search
    if best_ambient is not None:
        current = list(best_ambient)
        current_score = best_score
        
        # We will iterate until deadline
        while time.time() < deadline:
            # Pick a random index to change
            idx = rng.randrange(12)
            old_val = current[idx]
            
            # Pick a new value
            new_val = rng.randrange(129)
            if new_val in current:
                continue
                
            current[idx] = new_val
            new_score = surrogate_score(current)
            
            if new_score < current_score:
                current_score = new_score
                best_score = new_score
                best_ambient = list(current)
            else:
                # Revert
                current[idx] = old_val
                
            # Occasionally try a "swap" of two elements? 
            if rng.random() < 0.1:
                i1 = rng.randrange(12)
                i2 = rng.randrange(12)
                if i1 != i2:
                    v1 = current[i1]
                    v2 = current[i2]
                    current[i1] = rng.randrange(129)
                    while current[i1] in current:
                        current[i1] = rng.randrange(129)
                    current[i2] = rng.randrange(129)
                    while current[i2] in current:
                        current[i2] = rng.randrange(129)
                        
                    new_score = surrogate_score(current)
                    if new_score < current_score:
                        current_score = new_score
                        best_score = new_score
                        best_ambient = list(current)
                    else:
                        current[i1] = v1
                        current[i2] = v2

    # Final Verification and Polish
    if best_ambient is None:
        return list(range(12))
        
    # Check if we can do a final polish to ensure we return a valid, distinct set
    # The search ensures distinctness by construction.
    
    # Let's verify the true max sidon size for the best candidate
    # And maybe try a few more local steps specifically optimizing the true metric
    # if time permits.
    
    if time.time() < deadline - 5.0:
        # Polish: try to reduce true max sidon size
        best_true = true_max_sidon(best_ambient)
        polished = list(best_ambient)
        
        for _ in range(500): # Limited polish steps
            if time.time() > deadline - 5.0:
                break
            idx = rng.randrange(12)
            old_val = polished[idx]
            new_val = rng.randrange(129)
            if new_val in polished:
                continue
            polished[idx] = new_val
            true_val = true_max_sidon(polished)
            if true_val < best_true:
                best_true = true_val
                best_ambient = list(polished)
            else:
                polished[idx] = old_val
                
    return best_ambient
