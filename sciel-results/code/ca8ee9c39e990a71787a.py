import time
import random
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    sums = set()
    prods = set()
    # Optimization: Use lists and set conversion, or direct addition.
    # Direct addition is usually fine for n=20.
    for i in range(n):
        ai = A[i]
        for j in range(n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def _generate_gp_candidate(r, start=1, n=20, max_val=10000):
    """
    Generate a candidate set based on a geometric progression with ratio r.
    If the GP exceeds max_val or has duplicates, we truncate and fill/perturb.
    """
    A = []
    current = float(start)
    for _ in range(n):
        val = int(round(current))
        if val <= 0:
            val = 1
        if val > max_val:
            break
        if val not in A:
            A.append(val)
        else:
            # If duplicate, nudge it
            val += 1
            if val > max_val:
                val = max_val - 1
            if val not in A:
                A.append(val)
            else:
                # Hard collision, skip this step in GP generation
                pass
        current *= r
        
    # If we didn't get 20 elements, fill the rest with random distinct values
    # that are larger than the last GP element to maintain "spread"
    if len(A) < n:
        existing = set(A)
        # Start filling from the last element + 1, or random high values
        # To minimize product set, we might want to continue the GP logic loosely
        # or just pick random large numbers that don't create too many collisions
        last_val = A[-1] if A else 1
        # Try to continue the GP for the remaining slots
        for _ in range(n - len(A)):
            # Pick a value that is roughly last_val * r^(1/n) or similar
            # Or just random in [last_val+1, 10000]
            # Let's try a mix: some structured, some random
            if len(A) > 0 and A[-1] < 10000:
                # Estimate next GP step
                if len(A) >= 2:
                    r_est = A[-1] / A[-2] if A[-2] != 0 else 1
                    next_est = int(A[-1] * r_est)
                    if next_est > 10000:
                        next_est = 10000
                    if next_est not in existing and next_est > A[-1]:
                        A.append(next_est)
                        existing.add(next_est)
                        continue
            # Random fill
            for _ in range(100):
                val = random.randint(last_val + 1, 10000)
                if val not in existing:
                    A.append(val)
                    existing.add(val)
                    break
            else:
                # Fallback: increment from last
                val = A[-1] + 1 if A else 1
                while val in existing and val <= 10000:
                    val += 1
                if val <= 10000:
                    A.append(val)
                    existing.add(val)
    
    # If we have more than n, take first n (sorted)
    A = sorted(A[:n])
    
    # Ensure distinctness and bounds
    if len(set(A)) != n or any(x < 1 or x > 10000 for x in A):
        return None
        
    return A

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    rng = random.Random(42)
    
    best_A = None
    best_score = float('inf')
    
    # Phase 1: Multi-start over Geometric Ratios
    # The product set size is heavily influenced by the multiplicative structure.
    # GPs with small integer-like ratios tend to have small product sets.
    # We search over the ratio r in log space.
    
    # Log range for r: 
    # Min r: close to 1 (e.g., 1.01) -> dense set, large sumset, small productset? 
    # Max r: close to (10000)^(1/19) ~ 1.85 (to fit 20 terms in 1..10000)
    # Actually, if r is too large, we run out of room.
    # Let's sample r in [1.05, 1.8]
    
    r_samples = np.linspace(1.05, 1.8, 100)
    # Also add some specific "nice" ratios
    nice_ratios = [1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    all_r = list(r_samples) + nice_ratios
    
    for r in all_r:
        if time.time() > deadline:
            break
            
        # Generate candidate
        A = _generate_gp_candidate(r, start=1, n=20, max_val=10000)
        if A is None:
            continue
            
        # Local refinement around this GP structure
        # We will perturb individual elements slightly to break symmetries that might increase set sizes
        # and to optimize the specific sum/product collisions.
        
        current_A = A[:]
        current_score = _score(current_A)
        
        # Simple local search: try changing each element by small deltas
        # Since the structure is GP, we want to stay close to the GP values.
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(20))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2:
                    break
                    
                old_val = current_A[i]
                # Try small perturbations
                # Range of perturbation: +/- 5% of the value or +/- 10
                delta_range = max(10, int(old_val * 0.05))
                
                # Sample a few deltas
                for _ in range(5):
                    delta = rng.randint(-delta_range, delta_range)
                    new_val = old_val + delta
                    
                    if new_val <= 0 or new_val > 10000:
                        continue
                    if new_val in current_A and new_val != old_val:
                        continue
                        
                    new_A = current_A[:]
                    new_A[i] = new_val
                    new_A.sort()
                    
                    # Quick check: if new_val is close to old_val, score change is local
                    s = _score(new_A)
                    
                    if s < current_score:
                        current_A = new_A
                        current_score = s
                        improved = True
                        break # Move to next index
                        # Note: We break out of delta loop, not index loop
                        # But we need to re-evaluate 'improved' for the while condition
                        # Actually, standard hill climbing: if we found an improvement, 
                        # we restart the scan or continue. Let's continue scanning 
                        # but mark improved.
                
                # If we found an improvement in this index, we break the inner delta loop.
                # We do NOT break the index loop immediately unless we want to restart.
                # For efficiency, we just continue to next index.
                
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
            
    # Phase 2: If no GP candidate was good enough, or to explore non-GP structures,
    # run a few random restarts with SA, but constrained to "log-convex" or similar?
    # For now, let's just do a few pure random starts if time permits.
    
    if time.time() < deadline - 10:
        for _ in range(5):
            # Random start
            A = rng.sample(range(1, 10001), 20)
            current_A = A[:]
            current_score = _score(current_A)
            
            # Quick SA
            T = 500
            for step in range(2000):
                if time.time() > deadline - 2:
                    break
                i = rng.randint(0, 19)
                old_val = current_A[i]
                new_val = rng.randint(1, 10000)
                if new_val in current_A and new_val != old_val:
                    continue
                new_A = current_A[:]
                new_A[i] = new_val
                new_A.sort()
                s = _score(new_A)
                delta = s - current_score
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_A = new_A
                    current_score = s
                T *= 0.999
                
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
                
    # Final fallback if best_A is still None (should not happen)
    if best_A is None:
        best_A = list(range(1, 21))
        
    return sorted(best_A)
