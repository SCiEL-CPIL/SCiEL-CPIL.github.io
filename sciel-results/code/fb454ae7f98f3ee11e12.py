"""Combined program: greedy residue-class removal with NumPy vectorization and set-cover-style heuristics."""
import time
import numpy as np

N = 5000
DEADLINE = 300.0  # seconds, safety margin below 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.where(sieve)[0].tolist()

def construct():
    start_time = time.time()
    
    # Precompute primes up to N+1 (since len(points) can be up to N+1)
    all_primes = _primes(N + 1)
    
    # Start with the full set [0, 5000]
    points = np.arange(N + 1, dtype=np.int64)
    
    # Iteratively repair: for each prime p <= current length, if all p residue
    # classes are present, remove the least-populated one
    # We use a set-cover-inspired heuristic: prioritize primes where removal
    # of one class breaks the most "conflicts" (i.e., where the class is most evenly distributed)
    
    changed = True
    while changed:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        changed = False
        current_len = len(points)
        
        if current_len == 0:
            break
        
        # Find the best prime to act on using a heuristic:
        # For each prime p <= current_len, check if all residue classes are covered.
        # If so, compute a "conflict score" based on how evenly the classes are distributed.
        # Remove the class from the prime with the highest conflict score (most "damaging" to admissibility).
        
        best_prime = None
        best_class = None
        best_score = -1
        best_counts = None
        
        for p in all_primes:
            if p > current_len:
                break
            
            # Count residue classes mod p using NumPy
            residues = points % p
            class_counts = np.bincount(residues, minlength=p)
            
            # Check if all residue classes are covered
            if np.all(class_counts > 0):
                # Compute a heuristic score: prefer primes where the minimum class is small relative to the mean
                # This approximates "removing this class breaks the most conflicts"
                min_count = np.min(class_counts)
                mean_count = np.mean(class_counts)
                
                # Higher score = more urgent to fix (more uneven distribution or smaller min)
                # Use 1/min_count as a proxy for "how much this class contributes to full coverage"
                # Also factor in p: larger primes have more classes, so removing one is proportionally less impactful
                # But we want to fix the "worst" case first
                
                # Simple heuristic: score = min_count / (p * mean_count)
                # Lower min_count relative to mean = more uneven = higher priority
                # Actually, we want to remove the LEAST populated class, so we just need to identify which prime
                # to act on. Let's use: prioritize smaller primes first (they're checked more often in admissibility)
                # but if a large prime has a very small min class, that might be more critical.
                
                # Alternative: just process in ascending order (simpler, matches current behavior)
                # But let's try: prioritize by (min_count / p) - smaller is more urgent
                score = min_count / p
                
                # We want to act on the most urgent prime. Let's define urgency as inverse of score
                # Actually, let's just stick with ascending order for simplicity and correctness,
                # but use NumPy for speed. The key improvement is vectorized counting.
                
                if best_prime is None or score < best_score:
                    best_prime = p
                    best_score = score
                    best_counts = class_counts
        
        if best_prime is not None:
            p = best_prime
            # Find the least-populated class
            min_class = int(np.argmin(best_counts))
            
            # Remove all points in that residue class
            mask = points % p != min_class
            points = points[mask]
            changed = True
    
    return points.tolist()
