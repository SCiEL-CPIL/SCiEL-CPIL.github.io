"""Pack distinct multiples of a fixed integer set - minimize the enclosing span."""
import time

# EVOLVE-BLOCK-START
_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_N = len(_DIVISORS)
# LCM of all divisors: 2*3*5*7*11*13 = 30030
_PERIOD = 30030
_DEADLINE = time.time() + 85  # safety margin vs 90 s wall-clock
_BEST = list(_DIVISORS)  # always-valid fallback
_BEST_SPAN = max(_BEST) - min(_BEST)


def _check_window(L, S):
    """Check if we can assign distinct multiples of each divisor in [L, L+S].
    Returns list of representatives if possible, else None.
    """
    U = L + S
    # For each divisor, find all multiples in [L, U]
    cands = []
    for d in _DIVISORS:
        k = (L + d - 1) // d
        multiples = []
        m = k * d
        while m <= U:
            multiples.append(m)
            m += d
        cands.append(multiples)
    
    # Check if any divisor has no candidates
    for c in cands:
        if not c:
            return None
    
    # Use backtracking with MRV (minimum remaining values) heuristic
    # Order divisors by number of candidates (ascending)
    order = sorted(range(_N), key=lambda i: len(cands[i]))
    
    used = set()
    reps = [0] * _N
    
    def bt(pos):
        if pos == _N:
            return True
        i = order[pos]
        for val in cands[i]:
            if val not in used:
                used.add(val)
                reps[i] = val
                if bt(pos + 1):
                    return True
                used.remove(val)
        return False
    
    if bt(0):
        return reps
    return None


def construct():
    global _BEST, _BEST_SPAN
    start = time.time()

    # Search spans from 1 upward. For each span, check L values.
    # Due to periodicity with period 30030, we only need L in [1, 30030].
    # But the values must be <= 1000000, so L+S <= 1000000.
    
    for S in range(1, _BEST_SPAN):
        if time.time() > _DEADLINE:
            break
        
        # L must satisfy: L >= 1 and L + S <= 1000000
        # Also, due to periodicity, we can restrict L to [1, 30030]
        # But we need to ensure that for some representative k, k*d_i is in [L, L+S]
        # and the overall pattern repeats, so checking L in [1, 30030] should suffice
        # for finding the minimum span.
        
        max_L = min(30030, _MAXV - S)
        
        found = False
        for L in range(1, max_L + 1):
            if time.time() > _DEADLINE:
                break
            
            res = _check_window(L, S)
            if res is not None:
                # Validate
                if all(1 <= x <= _MAXV for x in res) and len(set(res)) == _N:
                    if S < _BEST_SPAN:
                        _BEST = list(res)
                        _BEST_SPAN = S
                found = True
                break
        
        if found:
            break
    
    # Final validation + type coercion
    out = [int(x) for x in _BEST]
    assert len(out) == _N
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in out)
    assert len(set(out)) == _N
    assert all(1 <= x <= _MAXV for x in out)
    assert all(out[i] % _DIVISORS[i] == 0 for i in range(_N))
    return out
# EVOLVE-BLOCK-END
