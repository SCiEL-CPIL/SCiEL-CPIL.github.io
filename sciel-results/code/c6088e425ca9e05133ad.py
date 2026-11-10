"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def _score(A):
    """Return max(|A+A|, |A*A|) for sorted list A of length 20."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return max(len(ss), len(ps))

def _score_bounded(A, bound):
    """Return max(|A+A|, |A*A|) but early-exit if either set reaches bound."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            ss.add(s)
            if len(ss) >= bound:
                return bound
            p = ai * A[j]
            ps.add(p)
            if len(ps) >= bound:
                return bound
    return max(len(ss), len(ps))

def _valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 or x > maximum for x in A):
        return False
    return True

def _gen_smooth_numbers(max_val=10000, primes=(2, 3, 5)):
    """Generate all p-smooth numbers up to max_val."""
    smooth = set()
    def gen(idx, current):
        if current > max_val:
            return
        if idx == len(primes):
            if current > 0:
                smooth.add(current)
            return
        p = primes[idx]
        val = current
        while val <= max_val:
            gen(idx + 1, val)
            if val > max_val // p:
                break
            val *= p
    gen(0, 1)
    return sorted(smooth)

def _construct_candidates(maximum=10000, n=20):
    """Generate diverse initial candidate sets, prioritizing AP-based starts."""
    candidates = []
    smooth = _gen_smooth_numbers(maximum)
    
    # 1. Arithmetic progressions (prioritized - minimizes sumset to ~39)
    for a in range(1, 50):
        for b in range(1, 50):
            A = [a * t + b for t in range(n)]
            if all(0 < x <= maximum for x in A) and len(set(A)) == n:
                candidates.append(A)
    
    # 2. First N smooth numbers
    if len(smooth) >= n:
        candidates.append(sorted(smooth[:n]))
    
    # 3. Powers of 2
    p2 = [2**i for i in range(n)]
    if all(x <= maximum for x in p2):
        candidates.append(p2)
    
    # 4. Powers of 2 + some small odds
    p2_10 = [2**i for i in range(10)]
    for extra in [[3,5,7,9,11,13], [3,5,7,15,21,25], [1,3,5,7,9,11], [3,5,7,9,11,15]]:
        A = sorted(set(p2_10 + extra))
        if len(A) == n and all(x <= maximum for x in A):
            candidates.append(A)
    
    # 5. Mixed geometric
    for base in [2, 3]:
        A = [base**i for i in range(n)]
        if all(x <= maximum for x in A):
            candidates.append(sorted(A))
    
    # 6. Quadratic sequences
    for c2 in range(1, 30):
        for c1 in range(-10, 11):
            for c0 in range(1, 100):
                A = []
                valid = True
                for t in range(n):
                    val = c2 * t * t + c1 * t + c0
                    if val <= 0 or val > maximum:
                        valid = False
                        break
                    A.append(val)
                if valid and len(set(A)) == n:
                    candidates.append(sorted(A))
    
    # 7. Smooth number subsets with gaps
    if len(smooth) >= n * 2:
        candidates.append(sorted(smooth[::2][:n]))
        candidates.append(sorted(smooth[:n//2] + smooth[n//2:n]))
    
    # 8. Random samples from smooth numbers
    rng = random.Random(42)
    for _ in range(5):
        if len(smooth) >= n:
            A = sorted(rng.sample(smooth, n))
            candidates.append(A)
    
    # 9. Hybrid: powers of 2 with some smooth replacements
    p2_base = [2**i for i in range(n)]
    for _ in range(5):
        A = list(p2_base)
        for _ in range(3):
            idx = rng.randint(0, n-1)
            A[idx] = rng.choice(smooth[:50])
        A = sorted(set(A))
        if len(A) == n and all(x <= maximum for x in A):
            candidates.append(A)
    
    return candidates

def _local_search(A, max_val=10000, n=20, deadline=None, smooth_list=None):
    """Local search with incremental scoring and bounded evaluation."""
    if deadline is None:
        deadline = time.time() + 60
    
    best_A = list(A)
    best_score = _score(best_A)
    A_set = set(best_A)
    rng = random.Random(42)
    
    # Build candidate pool - focus on values that create product collisions
    pool = set()
    pool.update(best_A)
    
    # Add smooth numbers (high density for product collisions)
    if smooth_list is None:
        smooth_list = _gen_smooth_numbers(max_val)
    pool.update(smooth_list[:300])
    
    # Add neighbors of current elements
    for x in best_A:
        for d in range(-10, 11):
            if x + d > 0 and x + d <= max_val:
                pool.add(x + d)
        for mult in [2, 3, 4, 5]:
            if x * mult <= max_val:
                pool.add(x * mult)
            if x % mult == 0:
                pool.add(x // mult)
        # Add values that share factors with existing elements
        for y in best_A:
            if x != y and y * x <= max_val:
                pool.add(y * x)
    
    # Add small integers (high collision potential)
    for v in range(1, 200):
        pool.add(v)
    
    pool = sorted(pool)
    
    # Limit pool size for speed
    if len(pool) > 2000:
        # Keep small values (high collision) and values near existing elements
        small_pool = [v for v in pool if v <= 500]
        large_pool = [v for v in pool if v > 500]
        pool = small_pool[:500] + large_pool[:1500]
        pool = sorted(set(pool))
    
    improved = True
    stall_count = 0
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        
        # Determine pool size based on stall count (increase exploration when stuck)
        if stall_count <= 3:
            search_pool = pool[:500]
        elif stall_count <= 6:
            search_pool = pool[:1000]
        else:
            search_pool = pool
        
        for i in range(n):
            old_val = best_A[i]
            for new_val in search_pool:
                if new_val == old_val or new_val in A_set:
                    continue
                
                best_A[i] = new_val
                A_set.discard(old_val)
                A_set.add(new_val)
                
                sc = _score_bounded(best_A, best_score)
                
                if sc < best_score:
                    best_score = sc
                    improved = True
                    stall_count = 0
                    break
                else:
                    best_A[i] = old_val
                    A_set.discard(new_val)
                    A_set.add(old_val)
            
            if improved:
                break
        
        if not improved:
            # Random perturbations with larger jumps when stuck
            n_perturb = min(200, 50 + stall_count * 30)
            for _ in range(n_perturb):
                if time.time() > deadline:
                    break
                i = rng.randint(0, n - 1)
                old_val = best_A[i]
                new_val = rng.choice(search_pool)
                if new_val == old_val or new_val in A_set:
                    continue
                
                best_A[i] = new_val
                A_set.discard(old_val)
                A_set.add(new_val)
                
                sc = _score_bounded(best_A, best_score)
                
                if sc < best_score:
                    best_score = sc
                    improved = True
                    stall_count = 0
                    break
                else:
                    best_A[i] = old_val
                    A_set.discard(new_val)
                    A_set.add(old_val)
            
            if not improved:
                # Try a bigger jump: replace 2 elements simultaneously
                if time.time() < deadline - 5:
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        i = rng.randint(0, n - 1)
                        j = rng.randint(0, n - 1)
                        if i == j:
                            continue
                        old_i, old_j = best_A[i], best_A[j]
                        new_i = rng.choice(search_pool)
                        new_j = rng.choice(search_pool)
                        if new_i == old_i or new_i in A_set or new_j == old_j or new_j in A_set or new_i == new_j:
                            continue
                        
                        best_A[i] = new_i
                        best_A[j] = new_j
                        A_set.discard(old_i)
                        A_set.discard(old_j)
                        A_set.add(new_i)
                        A_set.add(new_j)
                        
                        sc = _score_bounded(best_A, best_score)
                        
                        if sc < best_score:
                            best_score = sc
                            improved = True
                            stall_count = 0
                            break
                        else:
                            best_A[i] = old_i
                            best_A[j] = old_j
                            A_set.discard(new_i)
                            A_set.discard(new_j)
                            A_set.add(old_i)
                            A_set.add(old_j)
                
                if not improved:
                    break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    
    n = 20
    maximum = 10000
    
    # Generate smooth numbers once
    smooth_list = _gen_smooth_numbers(maximum)
    
    # Generate initial candidates
    candidates = _construct_candidates(maximum, n)
    
    # Evaluate all candidates with bounded scoring
    best_A = None
    best_score = float('inf')
    
    for A in candidates:
        if time.time() > deadline - 30:  # Reserve 30s for local search
            break
        if _valid(A, n, maximum):
            sc = _score_bounded(A, best_score if best_score < float('inf') else 10000)
            if sc < best_score:
                best_score = sc
                best_A = list(A)
    
    if best_A is None:
        best_A = list(range(1, n + 1))
        best_score = _score(best_A)
    
    # Local search from best candidate
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_A, best_score = _local_search(best_A, maximum, n, deadline, smooth_list)
    
    # Final validation
    if not _valid(best_A, n, maximum):
        best_A = list(range(1, n + 1))
    
    return best_A
# EVOLVE-BLOCK-END
