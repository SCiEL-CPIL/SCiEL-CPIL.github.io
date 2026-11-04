import time
import random
import math

def _score_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0.0
    max_u = max(U_set)
    # Compute sumset and diffset efficiently
    sums = set()
    diffs = set()
    diffs.add(0)
    U_list = sorted(U_set)
    for i in range(n):
        ui = U_list[i]
        for j in range(i, n):
            sums.add(ui + U_list[j])
        for j in range(i + 1, n):
            d = U_list[j] - ui
            diffs.add(d)
            diffs.add(-d)
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max U + 1)
    if len(sums) == 0:
        return 0.0
    ratio = len(diffs) / len(sums)
    if ratio <= 1.0:
        return 1.0
    if 2 * max_u + 1 <= 1:
        return 1.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    U_list = sorted(U_set)
    n = len(U_list)
    max_val = max(U_list) if U_list else 0
    for i in range(n):
        a = U_list[i]
        for j in range(i, n):
            b = U_list[j]
            s = a + b
            d = b - a
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
    
    # Geometric neighbors
    if max_val > 0:
        for mult in [1.2, 1.5, 2, 2.5, 3]:
            val = int(max_val * mult)
            if 0 < val <= limit_val and val not in U_set:
                candidates.add(val)
    
    # Small integers
    for val in range(1, min(50, limit_val + 1)):
        if val not in U_set:
            candidates.add(val)

    return [c for c in candidates if 0 < c <= limit_val]

def _local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    current_score = _score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = _generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            current_max = max(U)
            new_limit = max(max_val_limit, int(current_max * 2.5))
            candidates = _generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = _generate_candidates(U, max(max_val_limit, int(max(U)*2.5)))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 2 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = _score_set(new_U)
                    if new_score > current_score + 1e-10:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, int(max(U)*2.5)))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def _geometric_set(b, m):
    """Generate U = {0, floor(b^k) for k=1..m}"""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = int(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def _modular_greedy_construct(p, n_elements, seed, deadline):
    """
    Greedy construction restricted to multiples of prime p.
    """
    random.seed(seed)
    
    # Start with V = {0, 1}
    V = [0, 1]
    best_score = _score_set([0, p])
    
    for _ in range(n_elements - 2):
        if time.time() > deadline - 5:
            break
            
        # Generate candidate multipliers
        max_existing = max(V) if V else 0
        
        candidates = set()
        # Geometric-ish expansion
        for base in [1.5, 2, 2.5, 3]:
            for k in range(1, 8):
                val = int(max_existing * (base ** k)) if max_existing > 0 else k
                if val > 0:
                    candidates.add(val)
        
        # Random candidates
        for _ in range(50):
            val = random.randint(1, max_existing * 3 + 10)
            if val > 0:
                candidates.add(val)
        
        # Small integers
        for val in range(1, min(50, max_existing + 1)):
            candidates.add(val)
        
        candidates = [c for c in candidates if c not in V]
        
        best_candidate = None
        best_local_score = best_score
        
        # Try a subset of candidates (to save time)
        random.shuffle(candidates)
        for c in candidates[:30]:
            new_V = V + [c]
            new_V.sort()
            # Convert to actual set U
            U = [v * p for v in new_V]
            s = _score_set(U)
            if s > best_local_score:
                best_local_score = s
                best_candidate = c
        
        if best_candidate is not None:
            V.append(best_candidate)
            V.sort()
            best_score = best_local_score
        else:
            # No improvement, try a random small addition
            if random.random() < 0.5:
                V.append(1 if 1 not in V else 2)
                V.sort()
            break
    
    # Final polish: convert V back to U and run local search
    U = [v * p for v in V]
    return U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    random.seed(42)

    # Initial seeds
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Structured seeds
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 4, 6, 10, 15, 21, 28], # Triangular + 1?
        [0, 1, 3, 6, 10, 15, 21, 28], # Triangular
    ]
    
    for cand in structured_candidates:
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Geometric sets
    for b in [1.5, 1.8, 2, 2.5, 3]:
        for m in range(6, 12):
            cand = _geometric_set(b, m)
            s = _score_set(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)

    # Modular greedy construction
    for p in [2, 3, 5, 7]:
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            U, s = _modular_greedy_construct(p, 15, seed, deadline)
            if s > best_score:
                best_score = s
                best_set = list(U)

    # Dynamic Local Search with multiple restarts
    seeds = [best_set]
    # Add a few other good structured seeds for diversity
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = _local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts? 
            # To keep it simple, we just update the primary seed list
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # Final Polish: High-precision local search on the best set
    # Use a tighter limit and more iterations if time permits
    if time.time() < deadline - 2:
        final_score, final_set = _local_search_dynamic(best_set, deadline, max_val_limit=2000, max_n_limit=50)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set

    return best_set
