import time
import random
import math

def _calc_score(A):
    """
    Compute max(|A+A|, |A*A|) for a set of distinct positive integers.
    A is a list of n distinct positive integers.
    """
    n = len(A)
    sumset = set()
    prodset = set()
    
    # Optimization: Use local variables for speed
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
            
    return max(len(sumset), len(prodset))

def _simulated_annealing(A, max_iters=10000, start_temp=50.0, end_temp=0.1, timeout=50.0):
    """
    Simulated Annealing to minimize max(|A+A|, |A*A|).
    """
    A = sorted(A)
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    n = len(A)
    
    # Precompute some useful data for move generation?
    # Maybe a list of "good" candidate values to try?
    # We can generate random values, but biasing towards multiples of existing elements might help.
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Cooling schedule
        T = start_temp * (end_temp / start_temp) ** (i / max_iters)
        
        # Pick a random index to modify
        idx = random.randint(0, n - 1)
        old_val = A[idx]
        
        # Generate new value
        # Strategy: Mix of random, local perturbation, and structured moves
        r = random.random()
        
        if r < 0.3:
            # Local perturbation
            delta = random.randint(-10, 10)
            new_val = old_val + delta
        elif r < 0.6:
            # Multiplicative perturbation (try to create collisions)
            # Multiply or divide by a small factor
            factor = random.choice([2, 3, 4, 5, 0.5, 1/3, 1/2])
            new_val = int(round(old_val * factor))
            if new_val == old_val:
                new_val = old_val + random.randint(1, 5)
        elif r < 0.8:
            # Try a multiple of another element in the set
            if n > 1:
                other_idx = random.randint(0, n - 1)
                if other_idx != idx:
                    base = A[other_idx]
                    mult = random.randint(1, 10)
                    new_val = base * mult
                else:
                    new_val = random.randint(1, 10000)
            else:
                new_val = random.randint(1, 10000)
        else:
            # Global random jump
            new_val = random.randint(1, 10000)
            
        # Validate new value
        if new_val < 1 or new_val > 10000:
            continue
        if new_val in A and new_val != old_val:
            continue
            
        # Apply move
        A[idx] = new_val
        new_score = _calc_score(A)
        
        delta_e = new_score - current_score
        
        # Accept if better or with probability exp(-delta_e / T)
        if delta_e < 0 or (T > 0 and random.random() < math.exp(-delta_e / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = list(A)
        else:
            # Revert
            A[idx] = old_val
            
    return sorted(best_A), best_score

def _local_search(A, timeout=5.0):
    """
    Hill climbing local search.
    """
    A = sorted(A)
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    n = len(A)
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for idx in range(n):
            if time.time() - start_time > timeout:
                break
            old_val = A[idx]
            # Try small deltas
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
                new_val = old_val + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A and new_val != old_val:
                    continue
                A[idx] = new_val
                new_score = _calc_score(A)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_A = list(A)
                    improved = True
                    break
            if not improved:
                A[idx] = old_val
                
    return sorted(best_A), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    candidates = []
    
    # 1. Arithmetic Progression (Minimizes Sumset to 2n-1)
    # Small APs keep values low, which is good for product set too (smaller numbers -> smaller products -> more collisions?)
    candidates.append(list(range(1, 21)))
    candidates.append(list(range(2, 22)))
    candidates.append(list(range(10, 30)))
    
    # 2. Smooth Numbers (Minimizes Product Set)
    # Generate 2-smooth numbers (2^a * 3^b)
    smooth2 = set()
    a = 0
    while 2**a <= 10000:
        b = 0
        v = 2**a
        while v <= 10000:
            smooth2.add(v)
            b += 1
            v = 2**a * (3**b)
        a += 1
    smooth2 = sorted(smooth2)
    if len(smooth2) >= 20:
        candidates.append(smooth2[:20])
        # Mix small and large smooth numbers
        candidates.append(smooth2[:10] + smooth2[-10:])
        
    # Generate 3-smooth numbers (2^a * 3^b * 5^c)
    smooth3 = set()
    a = 0
    while 2**a <= 10000:
        b = 0
        v2 = 2**a
        while v2 <= 10000:
            c = 0
            v = v2
            while v <= 10000:
                smooth3.add(v)
                c += 1
                v = v2 * (5**c)
            b += 1
            v2 = 2**a * (3**b)
        a += 1
    smooth3 = sorted(smooth3)
    if len(smooth3) >= 20:
        candidates.append(smooth3[:20])
        
    # 3. Random Sets
    random.seed(42)
    for _ in range(5):
        candidates.append(sorted(random.sample(range(1, 10001), 20)))
        
    # 4. Hybrid: Start with AP, then replace some with smooth numbers?
    # Let's just let SA do the mixing.
    
    best_A = None
    best_score = float('inf')
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        cand = sorted(set(x for x in cand if 1 <= x <= 10000))
        if len(cand) < 20:
            continue
        score = _calc_score(cand)
        if score < best_score:
            best_score = score
            best_A = list(cand)
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = _calc_score(best_A)
        
    # Refine the best candidate with Simulated Annealing
    # Allocate most of the time to SA
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        # Run SA for a significant portion of the time
        sa_timeout = time_remaining - 2.0
        best_A, best_score = _simulated_annealing(best_A, max_iters=100000, timeout=sa_timeout)
        
    # Final polish with local search if time permits
    if time.time() < deadline - 1.0:
        best_A, best_score = _local_search(best_A, timeout=deadline - time.time() - 0.5)
        
    return best_A

# EVOLVE-BLOCK-END
