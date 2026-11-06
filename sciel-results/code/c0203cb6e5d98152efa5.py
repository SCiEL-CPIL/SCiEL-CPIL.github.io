import time
import random
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

# Global cache for computed orders
_cached_orders = {}
_cache_lock = None # Not needed if single threaded, but good practice? No, just dict.

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def _can_add_fast(vec, vec_set, n):
    """Fast check for cap set property."""
    for a in vec_set:
        for b in vec_set:
            if a == b:
                continue
            # Check a + b + vec = 0 mod 3
            # This is equivalent to a + b = -vec mod 3
            # We can precompute sums? No, vec changes.
            # Just check directly.
            ok = True
            for k in range(n):
                if (a[k] + b[k] + vec[k]) % 3 != 0:
                    ok = False
                    break
            if ok:
                return False
    for a in vec_set:
        # Check 2*a + vec = 0 mod 3 => vec = -2*a = a mod 3? No.
        # 2*a + vec = 0 => vec = -2*a = a (since -2 = 1 mod 3).
        # So vec must be equal to a?
        # If vec == a, then a+a+a = 3a = 0.
        # So we just need to check if vec is in vec_set.
        # But vec is not in vec_set yet.
        # So we just need to ensure vec != a for all a in vec_set.
        # Which is true if vec is not in vec_set.
        pass
    return True

# Actually, the condition "no three distinct vectors sum to 0"
# includes cases like a, b, c distinct.
# It also excludes a, a, a? No, "distinct".
# But if a, b are distinct, and a+b+a=0 => b+a=0 => b=-a.
# So if we have a and -a, we can't add a?
# Wait, the triple must be distinct.
# So a, a, a is not a valid triple to check.
# But a, b, c distinct.
# If we add vec, we check:
# 1. vec, a, b distinct. a+b+vec=0.
# 2. vec, a, vec? No, distinct.
# 3. a, a, vec? No, distinct.
# So we only check pairs (a,b) in vec_set with a!=b, and check if a+b+vec=0.
# And we check if vec is in vec_set (which would mean vec,a,vec? No, vec is new).
# If vec is in vec_set, we don't add it.
# So the check is:
# 1. vec not in vec_set.
# 2. For all a, b in vec_set, a != b: (a+b+vec) % 3 != 0.

def _can_add(vec, vec_set, n):
    if vec in vec_set:
        return False
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
    return True

def _greedy_cap_set(n, order):
    cap_set = []
    vec_set = set()
    for vec in order:
        if _can_add(vec, vec_set, n):
            cap_set.append(vec)
            vec_set.add(vec)
    return cap_set

def _compute_best_order(n, time_budget=5.0):
    """Compute a good ordering for n using random restarts."""
    if n > 10:
        return None # Too large to pre-compute efficiently
    
    all_vectors = list(itertools.product([0, 1, 2], repeat=n))
    random.shuffle(all_vectors)
    
    best_cap_set = []
    best_size = 0
    start_time = time.time()
    
    # Try several random orders
    max_restarts = 20
    for _ in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        random.shuffle(all_vectors)
        cap_set = _greedy_cap_set(n, all_vectors)
        if len(cap_set) > best_size:
            best_size = len(cap_set)
            best_cap_set = cap_set
            
    # Create a mapping from vector to priority
    # Vectors in best_cap_set get high priority.
    # Others get random priority.
    priority_map = {}
    for i, v in enumerate(best_cap_set):
        priority_map[v] = 10000.0 - i # High priority, ordered by discovery?
        
    # Assign random priorities to others
    for v in all_vectors:
        if v not in priority_map:
            priority_map[v] = random.random()
            
    return priority_map

def priority(vector, n):
    global _cached_orders
    
    # Check if we have a cached order for this n
    if n not in _cached_orders:
        # Try to compute a good order
        # Only for small n to avoid timeout
        if n <= 10:
            try:
                _cached_orders[n] = _compute_best_order(n, time_budget=5.0)
            except Exception:
                _cached_orders[n] = None
        else:
            _cached_orders[n] = None
            
    order_map = _cached_orders[n]
    
    if order_map is not None:
        return order_map.get(vector, 0.0)
    else:
        # Fallback: structural heuristic + hash
        c0 = sum(1 for v in vector if v == 0)
        c1 = sum(1 for v in vector if v == 1)
        c2 = sum(1 for v in vector if v == 2)
        imbalance = abs(c1 - c2)
        h = _hash_vec(vector)
        structural = -5.0 * c0 - 2.0 * imbalance + 1.0 * min(c1, c2)
        result = (h / 4294967296.0) * 1000.0 + structural
        return result
