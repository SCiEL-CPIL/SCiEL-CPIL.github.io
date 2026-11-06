import time
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def _can_add(vec, vec_set, n):
    """Check if vec can be added to vec_set while maintaining cap set property."""
    for a in vec_set:
        for b in vec_set:
            if a == b:
                continue
            ok = True
            for k in range(n):
                if (a[k] + b[k] + vec[k]) % 3 != 0:
                    ok = False
                    break
            if ok:
                return False
    for a in vec_set:
        ok = True
        for k in range(n):
            if (2 * a[k] + vec[k]) % 3 != 0:
                ok = False
                break
        if ok:
            return False
    return True

def _greedy_cap_set(n, order):
    """Build cap set by greedily adding vectors in given order."""
    cap_set = []
    vec_set = set()
    for vec in order:
        if _can_add(vec, vec_set, n):
            cap_set.append(vec)
            vec_set.add(vec)
    return cap_set

def _hill_climb(cap_set, n, max_iter=20000):
    """Improve cap set via local search: try replacing vectors."""
    vec_set = set(cap_set)
    best_cap_set = list(cap_set)
    best_size = len(cap_set)
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        if len(cap_set) < 1:
            break
        
        idx = random.randint(0, len(cap_set) - 1)
        removed = cap_set[idx]
        new_vec_set = vec_set - {removed}
        
        # Try random vectors to replace
        found = False
        for _ in range(200):
            vec = tuple(random.randint(0, 2) for _ in range(n))
            if vec == removed or vec in new_vec_set:
                continue
            if _can_add(vec, new_vec_set, n):
                cap_set[idx] = vec
                new_vec_set.add(vec)
                found = True
                if len(new_vec_set) > best_size:
                    best_size = len(new_vec_set)
                    best_cap_set = list(new_vec_set)
                break
        
        if not found:
            vec_set.add(removed)
    
    return best_cap_set

def priority(vector, n):
    # Ensure deterministic, distinct scores for all vectors
    # Use a multi-component hash-based score with structural features
    
    c0 = sum(1 for v in vector if v == 0)
    c1 = sum(1 for v in vector if v == 1)
    c2 = sum(1 for v in vector if v == 2)
    
    # Structural features that correlate with cap set membership
    # Vectors with balanced 1/2 counts and few zeros tend to be in cap sets
    imbalance = abs(c1 - c2)
    
    # Primary: high-quality hash for uniqueness
    h = _hash_vec(vector)
    
    # Secondary: structural preference
    # Prefer vectors with moderate zero count, balanced 1/2
    structural = -5.0 * c0 - 2.0 * imbalance + 1.0 * min(c1, c2)
    
    # Combine: hash dominates for uniqueness, structural for ordering
    # Scale hash to [0, 1) and add structural boost
    result = (h / 4294967296.0) * 1000.0 + structural
    
    return result
