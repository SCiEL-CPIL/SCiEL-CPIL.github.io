"""Cap set construction via greedy selection with anchor-based spreading priorities."""
import time
import hashlib

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = 300.0
_ANCHOR_CACHE = {}

def _get_anchors(n):
    """Generate a set of anchor vectors for distance calculation."""
    if n in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[n]
    
    anchors = []
    # Constant vectors
    anchors.append(tuple([0] * n))
    anchors.append(tuple([1] * n))
    anchors.append(tuple([2] * n))
    
    # Basis vectors (e_i)
    for i in range(n):
        v = [0] * n
        v[i] = 1
        anchors.append(tuple(v))
        
    # Anti-basis vectors (2*e_i)
    for i in range(n):
        v = [0] * n
        v[i] = 2
        anchors.append(tuple(v))
        
    _ANCHOR_CACHE[n] = anchors
    return anchors

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Uses a combination of random hashing and distance to fixed anchors
    to encourage spatial spreading (approximating Farthest-Point Sampling).
    """
    # Deadline guard: If we are very close to the limit, return a neutral value
    # to avoid any potential overhead in complex calculations (though this is light).
    if time.time() - _START_TIME > _DEADLINE - 1.0:
        return 0.5

    # 1. High-quality random hash for symmetry breaking
    # Use SHA-256 for robust mixing
    h = int.from_bytes(hashlib.sha256(vector).digest()[:8], 'big')
    rand_score = h / (1 << 64)
    
    # 2. Distance to Anchors (Spreading Heuristic)
    # We want vectors that are far from the "center" or "common" structures.
    # Using a small set of anchors approximates this without O(n^2) cost.
    anchors = _get_anchors(n)
    # Limit to first 10 anchors for speed
    min_dist = n
    for a in anchors[:10]:
        # Hamming distance
        dist = 0
        for i in range(n):
            if vector[i] != a[i]:
                dist += 1
        if dist < min_dist:
            min_dist = dist
            
    dist_score = min_dist / n
    
    # 3. Structural Balance
    # Prefer vectors with balanced 1s and 2s
    n1 = 0
    n2 = 0
    for v in vector:
        if v == 1:
            n1 += 1
        elif v == 2:
            n2 += 1
            
    balance = 1.0 - abs(n1 - n2) / max(1, n)
    
    # Combine scores
    # Weighting: Spreading is primary, Randomness is secondary, Balance is tertiary
    return 0.5 * dist_score + 0.3 * rand_score + 0.2 * balance

# EVOLVE-BLOCK-END
