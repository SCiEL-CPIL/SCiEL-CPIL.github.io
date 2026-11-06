import time
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Safety margin before 330s kill

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def _is_cap_set(vectors, n):
    """Check if a set of vectors forms a cap set (no three distinct vectors sum to 0 mod 3)."""
    vec_set = set(vectors)
    for i, a in enumerate(vectors):
        for j in range(i+1, len(vectors)):
            b = vectors[j]
            # Compute c = -a - b mod 3
            c = tuple((-a[k] - b[k]) % 3 for k in range(n))
            if c in vec_set and c != a and c != b:
                return False
    return True

def _can_add(vec, vec_set, n):
    """Check if vec can be added to vec_set while maintaining cap set property."""
    for a in vec_set:
        for b in vec_set:
            if a == b:
                continue
            # Check if a + b + vec == 0 mod 3
            ok = True
            for k in range(n):
                if (a[k] + b[k] + vec[k]) % 3 != 0:
                    ok = False
                    break
            if ok:
                return False
    # Also check pairs involving vec
    for a in vec_set:
        # Check if a + a + vec == 0 mod 3 (i.e., 2a + vec == 0)
        ok = True
        for k in range(n):
            if (2 * a[k] + vec[k]) % 3 != 0:
                ok = False
                break
        if ok:
            return False
    return True

def _build_initial_cap_set(n):
    """Build an initial cap set using greedy approach with balanced 1/2 counts."""
    cap_set = []
    vec_set = set()
    
    # Generate candidate vectors, prioritizing balanced 1/2 counts
    candidates = []
    for mask in range(1 << n):
        vec = []
        c1 = 0
        c2 = 0
        for i in range(n):
            if (mask >> i) & 1:
                vec.append(2)
                c2 += 1
            else:
                vec.append(1)
                c1 += 1
        vec = tuple(vec)
        imbalance = abs(c1 - c2)
        candidates.append((imbalance, _hash_vec(vec), vec))
    
    candidates.sort(key=lambda x: (x[0], -x[1]))
    
    for _, _, vec in candidates:
        if _can_add(vec, vec_set, n):
            cap_set.append(vec)
            vec_set.add(vec)
    
    return cap_set, vec_set

def _hill_climb(cap_set, n, max_iter=5000):
    """Improve cap set via local search: try replacing vectors to increase size."""
    vec_set = set(cap_set)
    current_size = len(cap_set)
    best_cap_set = list(cap_set)
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Try removing a random vector and adding a different one
        if current_size < 1:
            break
        
        idx = random.randint(0, current_size - 1)
        removed = cap_set[idx]
        new_vec_set = vec_set - {removed}
        
        # Try to add a better vector
        added = False
        for mask in range(1 << n):
            vec = []
            for i in range(n):
                if (mask >> i) & 1:
                    vec.append(2)
                else:
                    vec.append(1)
            vec = tuple(vec)
            if vec == removed:
                continue
            if vec in new_vec_set:
                continue
            if _can_add(vec, new_vec_set, n):
                cap_set[idx] = vec
                new_vec_set.add(vec)
                added = True
                break
        
        if added:
            vec_set = new_vec_set
            if len(vec_set) > len(best_cap_set):
                best_cap_set = list(vec_set)
        else:
            # Restore
            vec_set.add(removed)
    
    return best_cap_set

def priority(vector, n):
    # Fast path: if we've already done optimization, use cached result
    if not hasattr(priority, '_cache'):
        priority._cache = {}
    
    cache_key = (n, tuple(vector))
    if cache_key in priority._cache:
        return priority._cache[cache_key]
    
    # Build initial cap set
    if time.time() > _DEADLINE - 30.0:
        # Not enough time, use simple heuristic
        c0 = sum(1 for v in vector if v == 0)
        c1 = sum(1 for v in vector if v == 1)
        c2 = sum(1 for v in vector if v == 2)
        h = _hash_vec(vector)
        imbalance = abs(c1 - c2)
        result = h - 50.0 * c0 - 10.0 * imbalance
        priority._cache[cache_key] = result
        return result
    
    # Build initial cap set and optimize
    cap_set, vec_set = _build_initial_cap_set(n)
    
    # Hill climbing
    if time.time() < _DEADLINE - 20.0:
        improved = _hill_climb(cap_set, n, max_iter=10000)
        if len(improved) > len(cap_set):
            cap_set = improved
    
    # Score based on cap set size
    result = float(len(cap_set))
    priority._cache[cache_key] = result
    return result
