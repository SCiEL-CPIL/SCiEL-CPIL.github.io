import time
import random
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0

def _compute_set_sizes_fft(u):
    """Compute |U+U| and |U-U| using FFT for exact counts."""
    if not u:
        return 0, 0
    m = max(u)
    if m == 0:
        return 1, 1
    size = 2 * m + 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    vec = np.zeros(fft_size, dtype=np.float64)
    for x in u:
        vec[x] = 1.0
    
    fft_vec = np.fft.fft(vec)
    sum_fft = fft_vec * fft_vec
    sum_conv = np.fft.ifft(sum_fft).real
    n_sums = int(np.sum(sum_conv[:size] > 0.5))
    
    diff_fft = fft_vec * np.conj(fft_vec)
    diff_conv = np.fft.ifft(diff_fft).real
    n_diffs = int(np.sum(diff_conv[:m+1] > 0.5))
    if m > 0:
        n_diffs += int(np.sum(diff_conv[fft_size-m:fft_size] > 0.5))
    
    return n_sums, n_diffs

def _score(u):
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    n_sums, n_diffs = _compute_set_sizes_fft(u)
    
    if n_sums == 0:
        return 0.0
        
    ratio = n_diffs / float(n_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _score_brute(u_list):
    """Score using brute force for small sets."""
    if not u_list or len(u_list) < 2:
        return 0.0
    m = max(u_list)
    if m < 1:
        return 0.0
    if m <= 300:
        sums = set()
        diffs = set()
        for a in u_list:
            for b in u_list:
                sums.add(a + b)
                diffs.add(a - b)
        n_sums = len(sums)
        n_diffs = len(diffs)
        if n_sums == 0:
            return 0.0
        ratio = n_diffs / float(n_sums)
        if ratio <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / math.log(2 * m + 1)
    else:
        return _score(u_list)

def _is_sidon(u):
    """Check if u is a Sidon set (all pairwise sums are distinct)."""
    sums = {}
    n = len(u)
    for i in range(n):
        for j in range(i, n):
            s = u[i] + u[j]
            if s in sums:
                return False
            sums[s] = (i, j)
    return True

def _is_sidon_fast(u):
    """Check if u is a Sidon set using difference set approach."""
    diffs = set()
    n = len(u)
    for i in range(n):
        for j in range(i + 1, n):
            d = u[j] - u[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _greedy_sidon(n, max_attempts=100):
    """
    Greedy construction of a Sidon set of size n.
    Try to keep the set as compact as possible.
    """
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    for attempt in range(max_attempts):
        u = [0]
        diffs = set()
        max_val = 0
        
        for i in range(1, n):
            placed = False
            # Try values from current max+1 upward
            for pos in range(max_val + 1, max_val + i * i * 2 + 50):
                valid = True
                for a in u:
                    d = pos - a
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    for a in u:
                        diffs.add(pos - a)
                    u.append(pos)
                    max_val = pos
                    placed = True
                    break
            
            if not placed:
                # Give up on this attempt
                break
        
        if placed or len(u) == n:
            if len(u) == n:
                return sorted(u)
    
    # Fallback: use known small Sidon sets or a simple construction
    # For n up to 15, use the classic construction
    if n <= 15:
        # Use the Bose-Chowla construction for small n
        # Or just use a simple quadratic construction
        u = [0]
        for i in range(1, n):
            u.append(i * i)
        return sorted(u)
    
    # Last resort: spread out
    u = [0]
    for i in range(1, n):
        u.append(i * n)
    return sorted(u)

def _error_repair(u, target_n, timeout=3.0):
    """
    Error repair: start from a near-Sidon set and repair violations.
    A Sidon set has |U-U| = n(n-1)+1 (maximal for the size).
    We try to find a set of size target_n that is close to Sidon
    but has a smaller max element.
    """
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score_brute(u)
    
    best_u = u[:]
    best_score = _score_brute(u)
    start_time = time.time()
    n = len(u)
    
    # First, check if it's already Sidon
    if _is_sidon_fast(u):
        return u, best_score
    
    # Repair loop: find conflicts and resolve them
    for iteration in range(500):
        if time.time() > start_time + timeout or time.time() > _DEADLINE - 5.0:
            break
        
        # Find all pairs with duplicate differences
        diff_count = {}
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d not in diff_count:
                    diff_count[d] = []
                diff_count[d].append((i, j))
        
        # Find a duplicate difference
        duplicate_found = False
        for d, pairs in diff_count.items():
            if len(pairs) > 1:
                duplicate_found = True
                # Move one of the elements in the first pair
                idx_i, idx_j = pairs[0]
                old_val = u[idx_j]
                
                # Try moving u[idx_j] to a new position
                max_val = max(u)
                # Try positions near the current value
                candidates = set()
                for delta in range(-10, 11):
                    candidates.add(old_val + delta)
                for _ in range(20):
                    candidates.add(random.randint(0, max_val + 20))
                
                best_new_val = None
                best_new_score = best_score
                
                for new_val in sorted(candidates):
                    if new_val < 0:
                        continue
                    if new_val in u and new_val != old_val:
                        continue
                    
                    new_u = u[:]
                    new_u[idx_j] = new_val
                    new_u = sorted(set(new_u))
                    if len(new_u) != len(u):
                        continue
                    
                    # Quick check: is it closer to Sidon?
                    new_diff_count = {}
                    for i2 in range(len(new_u)):
                        for j2 in range(i2 + 1, len(new_u)):
                            d2 = new_u[j2] - new_u[i2]
                            if d2 not in new_diff_count:
                                new_diff_count[d2] = 0
                            new_diff_count[d2] += 1
                    
                    num_duplicates = sum(1 for v in new_diff_count.values() if v > 1)
                    old_num_duplicates = sum(1 for v in diff_count.values() if v > 1)
                    
                    if num_duplicates < old_num_duplicates:
                        s = _score_brute(new_u)
                        if s > best_new_score:
                            best_new_score = s
                            best_new_val = new_val
                            best_u = new_u[:]
                            best_score = s
                
                if best_new_val is not None:
                    u = best_u[:]
                    break
        
        if not duplicate_found:
            break
    
    return best_u, best_score

def _local_search_sidon(u, timeout=5.0):
    """
    Local search specifically optimized for near-Sidon sets.
    Objective: maximize score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    
    For a Sidon set of size n:
    |U-U| = n(n-1) + 1
    |U+U| = n(n+1)/2
    Score = 1 + log((n(n-1)+1)/(n(n+1)/2)) / log(2*max(U)+1)
           = 1 + log(2(n-1)/(n+1) + 2/(n(n+1))) / log(2*max(U)+1)
    
    To maximize: minimize max(U) for a given n, or increase n.
    """
    u = sorted(set(u))
    if len(u) < 2:
        return u, _score_brute(u)
    
    best_u = u[:]
    best_score = _score_brute(u)
    start_time = time.time()
    n = len(u)
    
    while time.time() < start_time + timeout and time.time() < _DEADLINE - 3.0:
        improved = False
        max_val = max(u)
        
        for i in range(n):
            if time.time() > start_time + timeout or time.time() > _DEADLINE - 3.0:
                break
            
            old_val = u[i]
            
            # Try to move element i to reduce max(U) while maintaining Sidon property
            if i == n - 1 and old_val == max_val:
                # Try to reduce the maximum element
                for new_val in range(old_val - 1, -1, -1):
                    if new_val in u and new_val != old_val:
                        continue
                    
                    new_u = u[:]
                    new_u[i] = new_val
                    new_u = sorted(set(new_u))
                    if len(new_u) != n:
                        continue
                    
                    if _is_sidon_fast(new_u):
                        s = _score_brute(new_u)
                        if s > best_score:
                            u = new_u
                            best_u = u[:]
                            best_score = s
                            max_val = max(u)
                            improved = True
                        break  # Only try the first valid lower position
                    elif random.random() < 0.1:
                        # Occasionally accept non-Sidon if score is better
                        s = _score_brute(new_u)
                        if s > best_score * 0.99:
                            u = new_u
                            best_u = u[:]
                            best_score = s
                            max_val = max(u)
                            improved = True
                        break
            else:
                # Try small perturbations
                step = max(1, max_val // 20)
                for delta in [-step, step, -2*step, 2*step]:
                    new_val = old_val + delta
                    if new_val < 0:
                        continue
                    if new_val in u and new_val != old_val:
                        continue
                    
                    new_u = u[:]
                    new_u[i] = new_val
                    new_u = sorted(set(new_u))
                    if len(new_u) != n:
                        continue
                    
                    s = _score_brute(new_u)
                    if s > best_score:
                        u = new_u
                        best_u = u[:]
                        best_score = s
                        max_val = max(u)
                        improved = True
                        break
        
        if not improved:
            # Try adding a new element
            if len(u) < 30:
                max_val = max(u)
                for new_val in range(max_val + 1, max_val + max_val + 10):
                    new_u = u + [new_val]
                    new_u = sorted(set(new_u))
                    if len(new_u) != len(u) + 1:
                        continue
                    
                    if _is_sidon_fast(new_u):
                        s = _score_brute(new_u)
                        if s > best_score:
                            u = new_u
                            best_u = u[:]
                            best_score = s
                            improved = True
                            break
                if improved:
                    n = len(u)
        
        if not improved and random.random() < 0.1:
            # Random restart perturbation
            idx = random.randint(0, len(u) - 1)
            u[idx] = random.randint(0, max(u) + 20)
            u = sorted(set(u))
            if len(u) < n:
                # Restore missing element
                u.append(random.randint(0, max(u) + 30))
                u = sorted(set(u))
    
    return best_u, best_score

def _evolve_population(pop, timeout=10.0):
    """
    Evolutionary approach: maintain a population of candidate sets.
    Selection, crossover, mutation, and error repair.
    """
    pop = sorted(pop, key=lambda x: -x[1])  # Sort by score descending
    best_u, best_s = pop[0]
    start_time = time.time()
    
    while time.time() < start_time + timeout and time.time() < _DEADLINE - 5.0:
        new_pop = [pop[0]]  # Elitism
        
        while len(new_pop) < len(pop) and time.time() < start_time + timeout and time.time() < _DEADLINE - 5.0:
            # Selection: tournament
            candidates = random.sample(pop, min(3, len(pop)))
            parent1 = max(candidates, key=lambda x: x[1])
            parent2 = max(random.sample(pop, min(3, len(pop))), key=lambda x: x[1])
            
            # Crossover: combine elements from both parents
            u1, s1 = parent1
            u2, s2 = parent2
            
            # Try different crossover strategies
            strategy = random.random()
            if strategy < 0.3:
                # Uniform crossover
                child = []
                for x in u1:
                    if random.random() < 0.5:
                        child.append(x)
                for x in u2:
                    if random.random() < 0.5 and x not in child:
                        child.append(x)
                child = sorted(set(child))
                # Ensure 0 is in the set
                if 0 not in child:
                    child.append(0)
                    child = sorted(child)
                # Limit size
                if len(child) > max(len(u1), len(u2)):
                    # Keep the smallest elements to control max
                    child = sorted(child)[:max(len(u1), len(u2))]
            elif strategy < 0.6:
                # Order-based: take prefix from one, suffix from other
                split = random.randint(1, min(len(u1), len(u2)) - 1)
                child = list(u1[:split]) + list(u2[split:])
                child = sorted(set(child))
                if 0 not in child:
                    child.append(0)
                    child = sorted(child)
            else:
                # Just copy parent1 with mutation
                child = u1[:]
            
            # Ensure valid set
            child = sorted(set(child))
            if len(child) < 2:
                continue
            if 0 not in child:
                child.append(0)
                child = sorted(child)
            
            # Mutation: random perturbation
            if random.random() < 0.7:
                idx = random.randint(0, len(child) - 1)
                max_val = max(child)
                step = max(1, max_val // 10)
                child[idx] = max(0, child[idx] + random.randint(-step, step))
                child = sorted(set(child))
                if len(child) < 2:
                    continue
            
            # Error repair
            child_u, child_s = _error_repair(child, len(child), timeout=0.5)
            
            new_pop.append((child_u, child_s))
        
        # Update population
        pop = sorted(new_pop, key=lambda x: -x[1])[:len(pop)]
        if pop[0][1] > best_s:
            best_s = pop[0][1]
            best_u = pop[0][0]
    
    return best_u, best_s

def construct():
    global _DEADLINE
    random.seed(42)
    
    best_u = [0, 1]
    best_s = _score_brute(best_u)
    
    # Phase 1: Quick scan of small Sidon sets with local search
    for n in range(3, 16):
        if time.time() > _DEADLINE - 30.0:
            break
        
        # Generate a Sidon set
        u_init = _greedy_sidon(n, max_attempts=20)
        if len(u_init) != n:
            continue
        
        # Local search to improve
        u_opt, s_opt = _local_search_sidon(u_init, timeout=3.0)
        
        if s_opt > best_s:
            best_s = s_opt
            best_u = u_opt[:]
    
    # Phase 2: Evolutionary search with error repair
    if time.time() < _DEADLINE - 20.0:
        # Build initial population
        pop = []
        for n in range(5, 18):
            if time.time() > _DEADLINE - 15.0:
                break
            u_init = _greedy_sidon(n, max_attempts=10)
            if len(u_init) == n:
                s = _score_brute(u_init)
                pop.append((u_init, s))
        
        # Add some random sets
        for _ in range(5):
            n = random.randint(4, 15)
            max_span = n * n * 3
            u_init = [0]
            for _ in range(n - 1):
                u_init.append(random.randint(1, max_span))
            u_init = sorted(set(u_init))
            while len(u_init) < n:
                u_init.append(random.randint(1, max_span))
                u_init = sorted(set(u_init))
            s = _score_brute(u_init)
            pop.append((u_init, s))
        
        if pop:
            # Include best so far
            pop.append((best_u[:], best_s))
            
            time_budget = min(15.0, _DEADLINE - time.time() - 5.0)
            if time_budget > 3.0:
                u_evo, s_evo = _evolve_population(pop, timeout=time_budget)
                if s_evo > best_s:
                    best_s = s_evo
                    best_u = u_evo[:]
    
    # Phase 3: Final polish with error repair on the best solution
    if time.time() < _DEADLINE - 5.0:
        u_repaired, s_repaired = _error_repair(best_u, len(best_u), timeout=5.0)
        if s_repaired > best_s:
            best_u = u_repaired
            best_s = s_repaired
        
        # One more round of local search
        u_final, s_final = _local_search_sidon(best_u, timeout=3.0)
        if s_final > best_s:
            best_u = u_final
            best_s = s_final
    
    return best_u
