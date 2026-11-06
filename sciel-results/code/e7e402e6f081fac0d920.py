"""Optimized program for integer sum-product extremal sets."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for sorted list A."""
        n = len(A)
        sumset = set()
        prodset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(A[i] + A[j])
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    def local_search(A, max_iters=5000):
        """Local search: try swapping elements or adjusting them."""
        A = sorted(A)
        current_score = evaluate(A)
        best_local = list(A)
        best_local_score = current_score
        n = len(A)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random index to modify
            idx = random.randint(0, n - 1)
            old_val = A[idx]
            
            # Try different new values
            best_new_val = old_val
            best_new_score = current_score
            
            # Candidate values: nearby values, random values
            candidates = set()
            # Nearby values
            for delta in range(1, 50):
                candidates.add(old_val + delta)
                candidates.add(max(1, old_val - delta))
            # Random values
            for _ in range(10):
                candidates.add(random.randint(1, 10000))
            # Geometric neighbors
            if old_val > 1:
                candidates.add(max(1, old_val // 2))
            candidates.add(old_val * 2)
            candidates.add(old_val * 3)
            
            for new_val in candidates:
                if new_val < 1 or new_val > 10000:
                    continue
                # Check distinctness
                if any(new_val == A[k] for k in range(n) if k != idx):
                    continue
                
                A[idx] = new_val
                A_sorted = sorted(A)
                score = evaluate(A_sorted)
                A[idx] = old_val
                
                if score < best_new_score:
                    best_new_score = score
                    best_new_val = new_val
            
            if best_new_val != old_val:
                A[idx] = best_new_val
                A = sorted(A)
                current_score = best_new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local = list(A)
            
            # Early stop if we've hit a good score
            if current_score <= 210:
                break
        
        return best_local, best_local_score
    
    # Generate initial candidates
    def make_geometric(base, ratio, n=20):
        """Generate a geometric-like progression."""
        A = []
        val = base
        for i in range(n):
            A.append(val)
            val = int(val * ratio)
            if val > 10000:
                # Fill remaining with distinct values
                for j in range(len(A), n):
                    val += random.randint(1, 100)
                    if val > 10000:
                        val = 10000 - (n - j)
                    A.append(val)
                break
        # Ensure distinct and valid
        A = sorted(set(A))
        while len(A) < n:
            A.append(random.randint(1, 10000))
        A = sorted(set(A))[:n]
        return A
    
    def make_arithmetic(start, diff, n=20):
        """Generate an arithmetic progression."""
        A = [start + i * diff for i in range(n)]
        if max(A) > 10000:
            diff = 10000 // n
            A = [start + i * diff for i in range(n)]
        A = sorted(set(A))
        while len(A) < n:
            A.append(random.randint(1, 10000))
        A = sorted(set(A))[:n]
        return A
    
    def make_random(n=20):
        """Generate a random set."""
        A = random.sample(range(1, 10001), n)
        return sorted(A)
    
    # Try multiple initial configurations
    initial_sets = []
    
    # Geometric progressions with various bases and ratios
    for base in [2, 3, 4, 5, 10]:
        for ratio in [1.5, 2, 2.5, 3]:
            initial_sets.append(make_geometric(base, ratio))
    
    # Arithmetic progressions
    for start in [1, 5, 10, 100]:
        for diff in [1, 2, 5, 10, 50]:
            initial_sets.append(make_arithmetic(start, diff))
    
    # Random sets
    for _ in range(10):
        initial_sets.append(make_random())
    
    # Also try a mix: start with small numbers, grow geometrically
    for _ in range(5):
        A = [1]
        for i in range(19):
            last = A[-1]
            new_val = int(last * random.uniform(1.2, 2.0))
            if new_val > 10000:
                new_val = random.randint(last + 1, 10000)
            if new_val not in A:
                A.append(new_val)
            else:
                A.append(last + 1)
        A = sorted(set(A))
        while len(A) < 20:
            A.append(random.randint(1, 10000))
        A = sorted(set(A))[:20]
        initial_sets.append(A)
    
    # Evaluate and run local search on each
    for A_init in initial_sets:
        if time.time() > deadline - 10.0:
            break
        
        # Deduplicate
        A_init = sorted(set(A_init))
        if len(A_init) != 20 or max(A_init) > 10000:
            continue
        
        score = evaluate(A_init)
        if score < best_score:
            best_score = score
            best_set = list(A_init)
        
        # Run local search
        A_local, local_score = local_search(A_init, max_iters=3000)
        if local_score < best_score:
            best_score = local_score
            best_set = list(A_local)
        
        # Early exit if we've found a very good solution
        if best_score <= 215:
            break
    
    # Final local search refinement on the best set
    if best_set is not None and time.time() < deadline - 5.0:
        A_refined, refined_score = local_search(best_set, max_iters=5000)
        if refined_score < best_score:
            best_score = refined_score
            best_set = list(A_refined)
    
    # Ensure validity
    if best_set is None:
        best_set = list(range(1, 21))
    
    best_set = sorted(set(best_set))
    # Pad or trim to exactly 20
    while len(best_set) < 20:
        best_set.append(random.randint(1, 10000))
    best_set = sorted(set(best_set))[:20]
    
    # Verify
    assert len(best_set) == 20
    assert all(isinstance(x, int) and x >= 1 and x <= 10000 for x in best_set)
    assert len(set(best_set)) == 20
    
    return best_set
# EVOLVE-BLOCK-END
