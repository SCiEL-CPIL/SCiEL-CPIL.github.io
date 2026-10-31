"""Program for the cap set task; implements a priority-based greedy construction."""
# EVOLVE-BLOCK-START
import time
import itertools

# Global deadline guard: 300s wall-clock limit, use 280s for safety
_DEADLINE = time.time() + 280.0

def _is_cap_set(vectors):
    """Check if a set of vectors in {0,1,2}^n is a cap set (no 3-term AP).
    A 3-term AP in {0,1,2}^n means there exist distinct a, b, c in the set
    such that a + c = 2*b (coordinate-wise). Equivalently, for any two distinct
    elements a, b in the set, the midpoint m = (a+b)/2 must NOT be in the set,
    AND (a+b) must be even in every coordinate (i.e., a[i] and b[i] have the same parity).
    Actually, more precisely: a, b, c form a 3-AP if a + c = 2b.
    So for any pair (a, b) with a != b, if (a[i] + b[i]) is even for all i,
    then c = 2*b - a must not be in the set.
    """
    vec_set = set(vectors)
    n = len(next(iter(vectors))) if vectors else 0
    for i, a in enumerate(vectors):
        for b in vectors[i+1:]:
            # Check if a + b is even in all coordinates (necessary for midpoint to be in {0,1,2}^n)
            # Actually, midpoint m_j = (a_j + b_j) / 2 must be in {0, 1, 2}
            # This requires a_j + b_j to be even for all j, i.e., a_j and b_j have same parity
            # But actually we need c = 2*b - a to be in {0,1,2}^n and in the set
            # Let's just compute c = 2*b - a and check if it's valid and in the set
            c = tuple(2 * b[j] - a[j] for j in range(n))
            # c must be in {0,1,2}^n
            if all(0 <= c[j] <= 2 for j in range(n)):
                if c in vec_set and c != a and c != b:
                    return False
                # Also check if c == a or c == b, that would mean a, b, c not all distinct
                # But we already require distinct a, b, c
    return True

def _is_cap_set_fast(vec_set, new_vec, n):
    """Check if adding new_vec to vec_set keeps it a cap set.
    Only need to check pairs involving new_vec.
    """
    for a in vec_set:
        # Check if new_vec, a, and some c form a 3-AP
        # Case 1: new_vec is the middle: 2*new_vec = a + c => c = 2*new_vec - a
        c1 = tuple(2 * new_vec[j] - a[j] for j in range(n))
        if all(0 <= c1[j] <= 2 for j in range(n)):
            if c1 in vec_set and c1 != new_vec and c1 != a:
                return False
        
        # Case 2: a is the middle: 2*a = new_vec + c => c = 2*a - new_vec
        c2 = tuple(2 * a[j] - new_vec[j] for j in range(n))
        if all(0 <= c2[j] <= 2 for j in range(n)):
            if c2 in vec_set and c2 != new_vec and c2 != a:
                return False
        
        # Case 3: c is the middle: 2*c = new_vec + a => c = (new_vec + a) / 2
        # This requires new_vec[j] + a[j] to be even for all j
        valid = True
        c3 = []
        for j in range(n):
            s = new_vec[j] + a[j]
            if s % 2 != 0:
                valid = False
                break
            c3.append(s // 2)
            if c3[-1] < 0 or c3[-1] > 2:
                valid = False
                break
        if valid:
            c3 = tuple(c3)
            if c3 in vec_set and c3 != new_vec and c3 != a:
                return False
    return True

def priority(vector, n):
    """
    Greedy cap set construction using a priority ordering.
    
    The strategy: order vectors by a hash-like function that creates a
    non-trivial ordering, then greedily add vectors that maintain the cap set property.
    
    We use a multi-criteria priority:
    1. Weighted sum with fixed weights to create a specific ordering
    2. The actual greedy construction happens externally, but priority() is called
       for each vector to determine its rank.
    
    Since the evaluator calls priority() for each vector and then processes them
    in descending order of priority, we need to return a scalar that induces a
    good ordering for cap set construction.
    
    Strategy: Use a pseudo-random but deterministic ordering based on the vector's
    coordinates weighted by position-specific weights. This breaks the symmetry
    of the ternary grid.
    """
    # Fixed weight vectors that create a non-trivial ordering
    # Use multiple "random" weight vectors and combine them
    weights1 = [1, 3, 2, 5, 1, 4, 2, 8, 3, 7, 2, 6, 1, 5, 3, 9, 2, 4, 7, 1]
    weights2 = [7, 2, 5, 1, 3, 8, 4, 6, 9, 1, 5, 3, 7, 2, 8, 4, 6, 9, 1, 5]
    
    score1 = 0.0
    score2 = 0.0
    for j in range(n):
        w1 = weights1[j % len(weights1)]
        w2 = weights2[j % len(weights2)]
        score1 += w1 * vector[j]
        score2 += w2 * vector[j]
    
    # Combine with a non-linear transformation to create more distinct priorities
    # Use a hash-like combination
    combined = score1 * 1.0 + score2 * 0.7 + (score1 * score2) * 0.01
    
    # Add a small perturbation based on the vector's "position" in lex order
    # to break ties
    lex_score = 0.0
    multiplier = 1.0
    for j in range(n):
        lex_score += vector[j] * multiplier
        multiplier *= 3.0
    
    return combined + lex_score * 1e-10

# EVOLVE-BLOCK-END
