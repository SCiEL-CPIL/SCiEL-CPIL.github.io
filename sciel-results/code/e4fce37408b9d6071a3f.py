import time
import random
import math

def _sumset_size(A):
    """Compute |A+A| with equal summands allowed"""
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai + A[j])
    return len(s)

def _prodset_size(A):
    """Compute |A*A| with equal factors allowed"""
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai * A[j])
    return len(s)

def _objective(A):
    """Return max(|A+A|, |A*A|)"""
    ss = _sumset_size(A)
    ps = _prodset_size(A)
    return max(ss, ps)

def _is_valid(A, n=20, maximum=10000):
    """Check if A is valid: n distinct positive integers <= maximum"""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x < 1 or x > maximum for x in A):
        return False
    return True

def _generate_ap(start=1, diff=1, n=20):
    """Generate an Arithmetic Progression"""
    A = [start + i * diff for i in range(n)]
    if max(A) > 10000:
        return None
    return A

def _generate_gp(start=1, ratio=2.0, n=20):
    """Generate a Geometric Progression (rounded)"""
    A = []
    val = float(start)
    for i in range(n):
        A.append(int(round(val)))
        val *= ratio
    # Ensure distinct and within bounds
    A = sorted(set(A))
    if len(A) < n:
        # Fill gaps if rounding caused collisions
        last = max(A) if A else 1
        while len(A) < n:
            last += 1
            if last > 10000:
                return None
            if last not in A:
                A.append(last)
    return sorted(A[:n])

def _generate_rational_gp(start=1, p=3, q=2, n=20):
    """Generate GP with rational ratio p/q"""
    A = []
    val = float(start)
    ratio = p / q
    for i in range(n):
        A.append(int(round(val)))
        val *= ratio
    A = sorted(set(A))
    if len(A) < n:
        last = max(A) if A else 1
        while len(A) < n:
            last += 1
            if last > 10000:
                return None
            if last not in A:
                A.append(last)
    return sorted(A[:n])

def _local_search(A, time_limit=5.0):
    """
    Local search with steepest descent and random restarts within the time limit.
    Uses a mix of local perturbations and global jumps.
    """
    start_time = time.time()
    current_A = sorted(A[:])
    current_obj = _objective(current_A)
    best_A = current_A[:]
    best_obj = current_obj
    
    n = len(current_A)
    max_val = 10000
    
    # Parameters for search
    step_size = 10
    
    while (time.time() - start_time) < time_limit:
        improved = False
        
        # 1. Try local moves: change one element by small delta
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            
            # Try deltas
            for delta in range(-step_size, step_size + 1):
                if delta == 0:
                    continue
                new_val = current_A[i] + delta
                if new_val < 1 or new_val > max_val:
                    continue
                if new_val in current_A:
                    continue
                
                trial = current_A[:]
                trial[i] = new_val
                if len(set(trial)) != n:
                    continue
                
                obj = _objective(trial)
                if obj < current_obj:
                    current_obj = obj
                    current_A = trial
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_A = current_A[:]
                    improved = True
                    break # Restart the i-loop with new state? Or just break delta loop?
                    # Breaking delta loop, continuing i loop with updated current_A is fine.
        
        # 2. If no improvement, try a random jump to escape local minimum
        if not improved:
            # Randomly pick an element and change it significantly
            i = random.randint(0, n - 1)
            new_val = random.randint(1, max_val)
            if new_val in current_A:
                continue
            trial = current_A[:]
            trial[i] = new_val
            if len(set(trial)) != n:
                continue
            obj = _objective(trial)
            
            # Accept with some probability if worse (Simulated Annealing lite)
            # Or just accept if it's not too much worse?
            # For now, accept if it's better or equal, or randomly if slightly worse
            if obj <= current_obj or random.random() < 0.1:
                current_A = trial
                current_obj = obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_A = current_A[:]
        
        # Occasionally, try to swap two elements
        if random.random() < 0.2:
            i, j = random.sample(range(n), 2)
            # Swap values? No, that doesn't change the set.
            # Replace two elements with two new random values?
            # This is expensive. Stick to single element changes.
            
    return best_A, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_A = None
    best_obj = float('inf')
    
    # Generate diverse starting points
    starts = []
    
    # 1. Arithmetic Progressions
    for start in [1, 2, 3, 5, 10]:
        for diff in [1, 2, 3]:
            A = _generate_ap(start=start, diff=diff, n=20)
            if A and _is_valid(A):
                starts.append(A)
                
    # 2. Geometric Progressions (Integer ratios)
    for ratio in [2, 3, 4]:
        A = _generate_gp(start=1, ratio=ratio, n=20)
        if A and _is_valid(A):
            starts.append(A)
            
    # 3. Rational Geometric Progressions (p/q)
    # Focus on ratios close to 1 to keep sumset small, and larger to structure product set
    for p in range(2, 10):
        for q in range(1, p):
            if math.gcd(p, q) != 1:
                continue
            ratio = p / q
            if ratio > 3: # Too sparse
                continue
            A = _generate_rational_gp(start=1, p=p, q=q, n=20)
            if A and _is_valid(A):
                starts.append(A)
                
    # 4. Random starts
    for _ in range(10):
        A = sorted(random.sample(range(1, 10001), 20))
        starts.append(A)
        
    # 5. Specific "Hybrid" starts: Low numbers + High numbers
    A_hybrid = list(range(1, 11)) + [100 + i for i in range(10)]
    starts.append(A_hybrid)
    
    # Evaluate all starts and keep the best few for local search
    # Sort by objective to prioritize good starts
    evaluated_starts = []
    for A in starts:
        obj = _objective(A)
        evaluated_starts.append((obj, A))
        
    evaluated_starts.sort(key=lambda x: x[0])
    
    # Run local search on the top 5 starts
    # Allocate time based on remaining budget
    remaining_time = deadline - (time.time() - start_time)
    time_per_start = max(1.0, remaining_time / 5)
    
    for i, (obj, A) in enumerate(evaluated_starts[:5]):
        if (time.time() - start_time) > deadline - 2:
            break
            
        # Run local search
        A_ls, obj_ls = _local_search(A, time_limit=time_per_start)
        
        if obj_ls < best_obj:
            best_obj = obj_ls
            best_A = A_ls[:]
            
        # If we found a very good solution, we can stop early or refine
        # For now, continue to try others if time permits
        
    # Final polish on the best found
    if best_A is not None and (time.time() - start_time) < deadline - 1:
        A_final, obj_final = _local_search(best_A, time_limit=min(5.0, deadline - (time.time() - start_time)))
        if obj_final < best_obj:
            best_obj = obj_final
            best_A = A_final[:]
            
    if best_A is None:
        best_A = list(range(1, 21))
        
    return sorted(best_A)
