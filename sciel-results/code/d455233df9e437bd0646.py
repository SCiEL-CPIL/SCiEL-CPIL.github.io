"""Rewritten program: search for A,B maximizing |A+B| with pairwise-coprime distinct sums."""
import time
import itertools
import math

_DEADLINE = 85.0  # safety margin before 90s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime."""
    vals = list(set(values))
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if math.gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _sumset(A, B):
    return set(a + b for a in A for b in B)

def _score(A, B):
    """Return (cardinality, valid) for given A, B."""
    s = _sumset(A, B)
    if _pairwise_coprime(s):
        return len(s), True
    return len(s), False

def _search_max():
    """
    Search over all triples A and B in [1,128] to maximize |A+B|
    subject to pairwise-coprime distinct sums.
    
    Strategy: 
    - Precompute all valid pairs (a,b) and their sums.
    - Use a smarter search: for each candidate sumset S (pairwise coprime),
      check if it can be covered by A x B with |A|=|B|=3.
    
    But that's expensive. Instead, do a direct search with pruning.
    
    Key insight: the maximum possible |A+B| is 9 (3x3 grid, all distinct).
    We need 9 pairwise coprime integers in [2, 256].
    
    Let's enumerate all pairwise-coprime 9-element subsets of [2,256] 
    and check if each can be expressed as A+B for some A,B with 3 elements each.
    """
    start_time = time.time()
    
    # First, let's try to find a solution with 9 elements directly.
    # A+B has 9 elements iff all 9 sums are distinct.
    # The 9 sums must be pairwise coprime.
    
    # Approach: iterate over all A (combinations of 3 from 1..128) is too many.
    # Instead, use the structure: if A = {a1,a2,a3}, B = {b1,b2,b3},
    # then A+B = {ai+bj : i,j in {1,2,3}}.
    
    # Let's fix B and search for A. Or better, use a heuristic search.
    
    # Precompute: for a given set of 9 target sums S, can we find A,B?
    # This is a system of 9 equations. Hard to solve directly.
    
    # Alternative: random search with local improvement.
    import random
    rng = random.Random(42)
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Verify the seed
    s_seed = _sumset(best_A, best_B)
    if _pairwise_coprime(s_seed):
        best_score = len(s_seed)
    
    # Try to find 9-element sumsets.
    # Key constraint: all 9 sums distinct and pairwise coprime.
    
    # Let's enumerate possible B values and for each, try to find A.
    # For fixed B = {b1,b2,b3}, A+B = {a+b1, a+b2, a+b3 for a in A}.
    # For this to have 9 distinct elements, we need a_i + b_j != a_k + b_l 
    # unless (i,j)=(k,l). This means the differences a_i - a_k must not equal 
    # b_l - b_j for any (i,k) != (j,l) pair... actually it's more subtle.
    
    # 9 distinct sums iff: for i != k and j != l, a_i + b_j != a_k + b_l.
    # i.e., a_i - a_k != b_l - b_j for all i!=k, j!=l.
    
    # Let's do a focused search:
    # 1. Generate candidate B sets.
    # 2. For each B, generate candidate A sets that give 9 distinct sums.
    # 3. Check pairwise coprimality.
    
    # To keep it fast, limit the search space.
    
    # Actually, let's just do a broader random search with many iterations.
    
    def random_triple():
        return sorted(rng.sample(range(1, 129), 3))
    
    # Phase 1: Random search
    iterations = 0
    while time.time() - start_time < _DEADLINE - 10:
        A = random_triple()
        B = random_triple()
        s = _sumset(A, B)
        if len(s) == 9 and _pairwise_coprime(s):
            return A, B
        if _pairwise_coprime(s) and len(s) > best_score:
            best_score = len(s)
            best_A, best_B = A, B
        iterations += 1
        if iterations % 50000 == 0 and time.time() - start_time > _DEADLINE - 15:
            break
    
    # Phase 2: Local search from best found
    if best_score >= 8:
        while time.time() - start_time < _DEADLINE - 5:
            A = list(best_A)
            B = list(best_B)
            # Perturb one element
            if rng.random() < 0.5:
                idx = rng.randrange(3)
                A[idx] = rng.randint(1, 128)
                A = sorted(set(A))
                if len(A) < 3:
                    continue
            else:
                idx = rng.randrange(3)
                B[idx] = rng.randint(1, 128)
                B = sorted(set(B))
                if len(B) < 3:
                    continue
            s = _sumset(A, B)
            if _pairwise_coprime(s):
                if len(s) > best_score:
                    best_score = len(s)
                    best_A, best_B = A, B
                    if best_score == 9:
                        return best_A, best_B
    
    return best_A, best_B

def construct():
    """Return two lists A,B each with exactly 3 distinct integers in [1,128].
    All distinct values in A+B must be pairwise coprime. Maximize |A+B|."""
    A, B = _search_max()
    # Ensure validity
    s = _sumset(A, B)
    if not _pairwise_coprime(s):
        # Fallback to known valid solution
        return [1, 7, 13], [4, 10, 16]
    return A, B
