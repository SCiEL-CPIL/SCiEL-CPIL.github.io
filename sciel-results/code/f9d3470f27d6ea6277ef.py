import time
import math

_START = time.time()
_DEADLINE = _START + 280.0

_cache = {}

def _conflict_degree(vector, n):
    """
    Structural heuristic: estimate the number of conflict pairs for this vector.
    
    A vector v is in conflict with a pair (a, b) if a + b + v ≡ 0 (mod 3).
    This means v ≡ -(a+b) (mod 3), or equivalently, a + b ≡ -v (mod 3).
    
    We approximate the conflict degree by counting how many pairs (a, b) 
    in a small neighborhood could form such a triple. Since we can't enumerate
    all 3^(2n) pairs, we use a structural proxy:
    
    For each coordinate i, if v[i] = 0, then a[i] + b[i] must be 0, meaning
    (a[i], b[i]) ∈ {(0,0), (1,2), (2,1)} — 3 options.
    If v[i] = 1, then a[i] + b[i] must be 2, meaning
    (a[i], b[i]) ∈ {(0,2), (1,1), (2,0)} — 3 options.
    If v[i] = 2, then a[i] + b[i] must be 1, meaning
    (a[i], b[i]) ∈ {(0,1), (1,0), (2,2)} — 3 options.
    
    So every coordinate gives exactly 3 options for the pair. The total number
    of pairs (a,b) with a+b ≡ -v is always 3^n, regardless of v.
    
    This means the raw conflict count is uniform! We need a different structural measure.
    
    Better: consider that a and b must be distinct from v and from each other.
    The number of *valid* conflicts (where a, b, v are all distinct) varies.
    
    For a given v, the number of pairs (a,b) with a+b ≡ -v, a≠b, a≠v, b≠v:
    Total pairs with a+b ≡ -v: 3^n
    Subtract pairs where a = b: need 2a ≡ -v (mod 3), so a ≡ -v/2 (mod 3) per coord.
    Since 2^{-1} ≡ 2 (mod 3), a[i] ≡ 2*v[i] (mod 3). So there's exactly 1 such a.
    Subtract pairs where a = v: then b ≡ -2v (mod 3), so b[i] ≡ v[i] (mod 3) since -2≡1.
    So b = v, giving 1 pair (v, v) but we need a≠b so this is already excluded.
    Subtract pairs where b = v: similarly 1 pair.
    
    So valid conflicts ≈ 3^n - 1 (for a=b case) - 2 (for a=v or b=v cases, but these 
    overlap with a=b when v ≡ 2v, i.e., v=0).
    
    This is still nearly uniform. The structural difference must come from the 
    *greedy selection* process: which vectors get selected early affects which 
    conflicts matter.
    
    New structural approach: prioritize vectors that are "far" in Hamming distance
    from the origin and from each other. Use a space-filling curve or locality-sensitive
    ordering to spread selections.
    """
    # Structural measure: Hamming distance from all-zeros and all-ones vectors
    # Vectors far from structured patterns are less likely to be in low-dim subspaces
    
    # Count coordinate diversity
    counts = [0, 0, 0]
    for x in vector:
        counts[x] += 1
    
    # Entropy of coordinate distribution - higher entropy means more "generic"
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p)
    
    # Also penalize vectors that are "aligned" (all same value)
    max_count = max(counts)
    alignment_penalty = max_count / n  # 1.0 if all same, ~1/3 if balanced
    
    # Structural priority: high entropy, low alignment
    return entropy * 10.0 - alignment_penalty * 5.0

def priority(vector, n):
    """
    Return a float priority for a vector in {0,1,2}^n.
    Prioritize vectors with high coordinate diversity (low conflict potential).
    """
    # Deadline check
    if time.time() - _START > _DEADLINE:
        return 0.0

    return _conflict_degree(vector, n)
