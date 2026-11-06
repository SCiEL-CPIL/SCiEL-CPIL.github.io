import time
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0  # 30s safety margin before 300s limit

# Global cache for reference cap set
_REF_CAP_SET = None
_REF_BUILD_DONE = False

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def _is_cap_set(vecs, n):
    """Check if a set of vectors forms a cap set."""
    m = len(vecs)
    for i in range(m):
        for j in range(i + 1, m):
            # Check if there exists k != i,j such that vecs[i] + vecs[j] + vecs[k] == 0 mod 3
            # For each coordinate, the forbidden value for the third vector is determined
            # We check against all other vectors
            for k in range(m):
                if k == i or k == j:
                    continue
                ok = True
                for coord in range(n):
                    if (vecs[i][coord] + vecs[j][coord] + vecs[k][coord]) % 3 == 0:
                        ok = False
                        break
                if not ok:
                    return False
    return True

def _build_reference_cap_set(n, time_budget):
    """
    Build a reference cap set using greedy heuristic within time budget.
    Returns a set of vectors that form a cap set.
    """
    global _REF_CAP_SET
    start = time.time()
    
    # Generate all vectors
    all_vecs = list(itertools.product([0, 1, 2], repeat=n))
    
    # Sort by a heuristic: prefer vectors with more non-zeros, then balanced 1/2
    def heuristic(v):
        c0 = sum(1 for x in v if x == 0)
        c1 = sum(1 for x in v if x == 1)
        c2 = sum(1 for x in v if x == 2)
        balance = abs(c1 - c2)
        return (-c0, balance, _hash_vec(v))
    
    all_vecs.sort(key=heuristic)
    
    cap_set = []
    cap_set_set = set()
    
    for v in all_vecs:
        if time.time() - start > time_budget:
            break
        
        # Check if adding v keeps the set a cap set
        can_add = True
        for existing in cap_set:
            # Check if v conflicts with any pair in cap_set
            # v conflicts with (a, b) if for some coord, v_i + a_i + b_i == 0 mod 3
            # We need to check all pairs, which is O(m^2)
            # Optimization: for each existing vector a, the forbidden values for v
            # given a and b are determined. But we need to check all pairs (a, b).
            # Alternative: for each pair (a, b) in cap_set, check if v is forbidden
            pass
        
        # More efficient: for each pair (a, b) in cap_set, compute the forbidden
        # vector f where f_i = -(a_i + b_i) mod 3. If v == f, conflict.
        # But we also need to check v against each single vector? No, the condition
        # is no THREE distinct vectors sum to zero. So v conflicts with pair (a,b)
        # if v_i = -(a_i+b_i) mod 3 for ALL i.
        
        # Precompute forbidden vectors from pairs
        if len(cap_set) >= 2:
            for idx_a in range(len(cap_set)):
                for idx_b in range(idx_a + 1, len(cap_set)):
                    a = cap_set[idx_a]
                    b = cap_set[idx_b]
                    # Check if v is the forbidden completion
                    conflict = True
                    for coord in range(n):
                        if (v[coord] + a[coord] + b[coord]) % 3 != 0:
                            conflict = False
                            break
                    if conflict:
                        can_add = False
                        break
                if not can_add:
                    break
        
        if can_add:
            cap_set.append(v)
            cap_set_set.add(v)
    
    _REF_CAP_SET = cap_set_set
    return cap_set_set

def priority(vector, n):
    global _REF_CAP_SET, _REF_BUILD_DONE
    
    # Check deadline
    if time.time() > _DEADLINE:
        h = _hash_vec(vector)
        return h - 30.0
    
    # On first call, build reference cap set with a time budget
    if not _REF_BUILD_DONE:
        _REF_BUILD_DONE = True
        # Use 5 seconds to build reference set
        _build_reference_cap_set(n, 5.0)
    
    h = _hash_vec(vector)
    
    # Score: vectors in reference cap set get high priority
    if _REF_CAP_SET is not None and vector in _REF_CAP_SET:
        return 100000.0 + h
    else:
        # For vectors not in reference set, use structural score
        c0 = sum(1 for x in vector if x == 0)
        c1 = sum(1 for x in vector if x == 1)
        c2 = sum(1 for x in vector if x == 2)
        balance = abs(c1 - c2)
        return h - 45.0 * c0 - 2.0 * balance

