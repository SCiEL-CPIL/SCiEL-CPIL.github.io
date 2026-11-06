"""Program for integer sum-product extremal sets using geometric progression structure."""
import itertools
import math
from functools import lru_cache

def construct():
    """
    Construct 20 distinct positive integers <= 10000 minimizing max(|A+A|, |A*A|).
    
    Strategy: Use a truncated geometric progression with rational ratio p/q.
    Geometric progressions create multiplicative collisions (a_i * a_j = a_k * a_l 
    when i+j = k+l), which minimizes the product set size. We search over small
    integers p, q and initial terms a to find the best configuration.
    """
    n = 20
    max_val = 10000
    
    best_score = float('inf')
    best_set = None
    
    # For geometric progression with ratio r = p/q:
    # a_i = a * (p/q)^i for i = 0, ..., n-1
    # We need all a_i to be distinct positive integers <= max_val
    # a_i = a * p^i / q^i must be integer, so q^i must divide a * p^i
    # Since gcd(p,q) = 1, q^i must divide a
    # So a must be a multiple of q^(n-1)
    
    # Search over small p, q with gcd(p,q)=1, p > q >= 1
    # and appropriate initial term a
    
    for q in range(1, 12):
        for p in range(q + 1, q + 12):
            if math.gcd(p, q) != 1:
                continue
            
            # a must be a multiple of q^(n-1) to make all terms integer
            # a = k * q^(n-1) for some positive integer k
            q_pow = q ** (n - 1)
            
            # The largest term is a * p^(n-1) / q^(n-1) = k * p^(n-1)
            # We need k * p^(n-1) <= max_val
            # So k <= max_val // p^(n-1)
            
            p_pow = p ** (n - 1)
            if p_pow > max_val:
                continue
            
            max_k = max_val // p_pow
            if max_k < 1:
                continue
            
            # Try different values of k
            # For small q, q^(n-1) can be large, limiting k
            for k in range(1, min(max_k + 1, 50)):
                a = k * q_pow
                
                # Generate the set
                A = []
                valid = True
                for i in range(n):
                    term = a * (p ** i) // (q ** i)
                    # Verify it's exactly integer (should be by construction)
                    if a * (p ** i) % (q ** i) != 0:
                        valid = False
                        break
                    if term > max_val:
                        valid = False
                        break
                    A.append(term)
                
                if not valid:
                    continue
                
                # Check distinctness
                if len(set(A)) != n:
                    continue
                
                # Compute sumset and productset sizes
                A_set = set(A)
                
                # Sumset: A + A with equal summands allowed
                sumset = set()
                for x in A:
                    for y in A:
                        sumset.add(x + y)
                
                # Productset: A * A with equal factors allowed
                prodset = set()
                for x in A:
                    for y in A:
                        prodset.add(x * y)
                
                score = max(len(sumset), len(prodset))
                
                if score < best_score:
                    best_score = score
                    best_set = A[:]
    
    # If no geometric progression worked well, try other structured sets
    if best_set is None:
        # Fallback: use powers of 2 (which is a geometric progression with p=2, q=1)
        best_set = [2**i for i in range(n)]
        # Check if all <= max_val
        if best_set[-1] > max_val:
            best_set = list(range(1, n + 1))
    
    # Additional local improvement: try small perturbations
    # Actually, let's also try some other structured approaches
    
    # Try sets based on powers of small integers
    candidates = []
    
    # Powers of 2
    p2 = [2**i for i in range(n)]
    if p2[-1] <= max_val:
        candidates.append(p2)
    
    # Powers of 3
    p3 = [3**i for i in range(n)]
    if p3[-1] <= max_val:
        candidates.append(p3)
    
    # Mixed: combine geometric with small offsets
    # Try a * 2^i + b for small b
    for a_base in [1, 2, 3]:
        for b in range(0, 3):
            cand = [a_base * (2**i) + b for i in range(n)]
            if len(set(cand)) == n and all(1 <= x <= max_val for x in cand):
                candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        A_set = set(cand)
        sumset = set(x + y for x in cand for y in cand)
        prodset = set(x * y for x in cand for y in cand)
        score = max(len(sumset), len(prodset))
        if score < best_score:
            best_score = score
            best_set = cand[:]
    
    # Also try the original simple approach as baseline
    simple = list(range(1, n + 1))
    sumset_s = set(x + y for x in simple for y in simple)
    prodset_s = set(x * y for x in simple for y in simple)
    score_s = max(len(sumset_s), len(prodset_s))
    if score_s < best_score:
        best_score = score_s
        best_set = simple[:]
    
    return best_set
