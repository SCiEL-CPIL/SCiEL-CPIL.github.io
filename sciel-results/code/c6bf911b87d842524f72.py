import time
import random
import numpy as np

_START = time.time()
_DEADLINE = 85.0  # Safety margin

def _sq_nonzero_count(coefs):
    """Count nonzero terms in the exact square using numpy convolution."""
    # Ensure input is int64 for numpy
    arr = np.array(coefs, dtype=np.int64)
    sq = np.convolve(arr, arr)
    return int(np.count_nonzero(sq))

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _construct_factorized(rng):
    """
    Search for P = A * B where A has 3 terms and B has 4 terms.
    A = a0*x^u0 + a1*x^u1 + a2*x^u2
    B = b0*x^v0 + b1*x^v1 + b2*x^v2 + b3*x^v3
    We want P to have 12 terms (distinct sums u_i+v_j).
    Then optimize coefficients to minimize nonzeros in P^2.
    """
    best_score = 10**9
    best_coefs = None
    
    # Initial random start
    # Exponents
    u = sorted(rng.sample(range(0, 15), 3))
    v = sorted(rng.sample(range(0, 15), 4))
    # Coefficients
    a = [rng.choice([-2, -1, 1, 2]) for _ in range(3)]
    b = [rng.choice([-2, -1, 1, 2]) for _ in range(4)]
    
    def _get_P(u, v, a, b):
        coefs = [0] * 41
        valid = True
        for i in range(3):
            for j in range(4):
                deg = u[i] + v[j]
                if deg > 40:
                    valid = False
                    break
                coefs[deg] += a[i] * b[j]
        if not valid:
            return None
        # Check if exactly 12 nonzero terms
        nz = sum(1 for c in coefs if c != 0)
        if nz != 12:
            return None
        if coefs[40] == 0:
             # Shift or accept? Task says nonzero final coefficient.
             # If coefs[40] is 0, we can shift the whole polynomial?
             # No, shifting changes the polynomial.
             # We must ensure degree is 40 and coeff is nonzero.
             # If max degree < 40, we can multiply by x^(40-max_deg).
             # But that adds terms? No, just shifts.
             # However, the problem asks for length <= 41.
             # If max_deg < 40, we can pad with zeros at the end?
             # "nonzero final coefficient" implies the last element of the list is nonzero.
             # So if max_deg < 40, we can return a list of length max_deg+1?
             # "length<=41".
             # But usually "polynomial of degree D" implies list of D+1.
             # Let's assume we can return shorter lists.
             # But "nonzero final coefficient" is key.
             # If we return coefs[:max_deg+1], and coefs[max_deg] != 0, it's valid.
             # Let's check the constraint: "length<=41, nonzero final coefficient".
             # So if max_deg is 30, we return length 31.
             # This is allowed.
             pass
        return coefs
    
    P = _get_P(u, v, a, b)
    if P is None:
        # Try to find a valid configuration
        for _ in range(1000):
            if time.time() > _DEADLINE: break
            u = sorted(rng.sample(range(0, 20), 3))
            v = sorted(rng.sample(range(0, 20), 4))
            a = [rng.choice([-2, -1, 1, 2]) for _ in range(3)]
            b = [rng.choice([-2, -1, 1, 2]) for _ in range(4)]
            P = _get_P(u, v, a, b)
            if P is not None:
                break
    
    if P is None:
        return [10**9, None]
    
    current_score = _sq_nonzero_count(P)
    best_score = current_score
    best_coefs = P[:]
    
    T = 5.0
    cooling = 0.99
    
    while time.time() < _DEADLINE:
        # Propose move
        move = rng.random()
        if move < 0.3:
            # Change exponent in A
            idx = rng.randrange(3)
            old_u = u[idx]
            new_u = old_u + rng.choice([-2, -1, 1, 2])
            if 0 <= new_u <= 40:
                u[idx] = new_u
                u = sorted(u)
                P_new = _get_P(u, v, a, b)
                if P_new is not None:
                    new_score = _sq_nonzero_count(P_new)
                    delta = new_score - current_score
                    if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                        P = P_new
                        if new_score < best_score:
                            best_score = new_score
                            best_coefs = P_new[:]
                    else:
                        u[idx] = old_u
                        u = sorted(u)
                else:
                    u[idx] = old_u
                    u = sorted(u)
        elif move < 0.6:
            # Change exponent in B
            idx = rng.randrange(4)
            old_v = v[idx]
            new_v = old_v + rng.choice([-2, -1, 1, 2])
            if 0 <= new_v <= 40:
                v[idx] = new_v
                v = sorted(v)
                P_new = _get_P(u, v, a, b)
                if P_new is not None:
                    new_score = _sq_nonzero_count(P_new)
                    delta = new_score - current_score
                    if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                        P = P_new
                        if new_score < best_score:
                            best_score = new_score
                            best_coefs = P_new[:]
                    else:
                        v[idx] = old_v
                        v = sorted(v)
                else:
                    v[idx] = old_v
                    v = sorted(v)
        elif move < 0.85:
            # Change coefficient in A
            idx = rng.randrange(3)
            old_a = a[idx]
            new_a = old_a + rng.choice([-2, -1, 1, 2])
            if -64 <= new_a <= 64 and new_a != 0:
                a[idx] = new_a
                P_new = _get_P(u, v, a, b)
                if P_new is not None:
                    new_score = _sq_nonzero_count(P_new)
                    delta = new_score - current_score
                    if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                        P = P_new
                        if new_score < best_score:
                            best_score = new_score
                            best_coefs = P_new[:]
                    else:
                        a[idx] = old_a
                else:
                    a[idx] = old_a
        else:
            # Change coefficient in B
            idx = rng.randrange(4)
            old_b = b[idx]
            new_b = old_b + rng.choice([-2, -1, 1, 2])
            if -64 <= new_b <= 64 and new_b != 0:
                b[idx] = new_b
                P_new = _get_P(u, v, a, b)
                if P_new is not None:
                    new_score = _sq_nonzero_count(P_new)
                    delta = new_score - current_score
                    if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                        P = P_new
                        if new_score < best_score:
                            best_score = new_score
                            best_coefs = P_new[:]
                    else:
                        b[idx] = old_b
                else:
                    b[idx] = old_b
        
        T *= cooling
    
    # Trim trailing zeros
    if best_coefs is not None:
        while len(best_coefs) > 1 and best_coefs[-1] == 0:
            best_coefs.pop()
        if best_coefs[-1] == 0:
             return [10**9, None]
        if len(best_coefs) > 41:
             return [10**9, None]
    
    return [best_score, best_coefs]

def _construct_direct_sa(rng):
    """Direct simulated annealing on 12-term polynomial."""
    best_score = 10**9
    best_coefs = None
    
    # Init
    positions = sorted(rng.sample(range(0, 40), 11) + [40])
    coefs = [0] * 41
    for p in positions:
        coefs[p] = rng.choice([-3, -2, -1, 1, 2, 3])
    if coefs[40] == 0:
        coefs[40] = 1
    
    current_score = _sq_nonzero_count(coefs)
    best_score = current_score
    best_coefs = coefs[:]
    
    T = 5.0
    cooling = 0.995
    
    while time.time() < _DEADLINE:
        move = rng.random()
        if move < 0.5:
            # Change value
            nz = [i for i in range(41) if coefs[i] != 0]
            idx = rng.choice(nz)
            old = coefs[idx]
            new = old + rng.choice([-3, -2, -1, 1, 2, 3])
            if -64 <= new <= 64 and new != 0:
                coefs[idx] = new
                if coefs[40] == 0:
                    coefs[idx] = old
                    continue
                new_score = _sq_nonzero_count(coefs)
                delta = new_score - current_score
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_coefs = coefs[:]
                else:
                    coefs[idx] = old
        elif move < 0.8:
            # Move position
            nz = [i for i in range(41) if coefs[i] != 0]
            idx = rng.choice(nz)
            if idx != 40:
                new_pos = rng.randrange(0, 40)
                if coefs[new_pos] == 0:
                    val = coefs[idx]
                    coefs[idx] = 0
                    coefs[new_pos] = val
                    new_score = _sq_nonzero_count(coefs)
                    delta = new_score - current_score
                    if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_coefs = coefs[:]
                    else:
                        coefs[new_pos] = 0
                        coefs[idx] = val
        else:
            # Swap positions
            nz = [i for i in range(41) if coefs[i] != 0]
            if len(nz) >= 2:
                i, j = rng.sample(nz, 2)
                coefs[i], coefs[j] = coefs[j], coefs[i]
                new_score = _sq_nonzero_count(coefs)
                delta = new_score - current_score
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 0.01)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_coefs = coefs[:]
                else:
                    coefs[i], coefs[j] = coefs[j], coefs[i]
        
        T *= cooling
    
    # Trim
    if best_coefs is not None:
        while len(best_coefs) > 1 and best_coefs[-1] == 0:
            best_coefs.pop()
        if best_coefs[-1] == 0:
             return [10**9, None]
        if len(best_coefs) > 41:
             return [10**9, None]
    
    return [best_score, best_coefs]

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0
    rng = random.Random(42)
    
    best_score = 10**9
    best_coefs = None
    
    # Run factorized search
    s1, c1 = _construct_factorized(rng)
    if c1 is not None and s1 < best_score:
        best_score = s1
        best_coefs = c1
    
    # Run direct search if time permits
    if time.time() < _DEADLINE:
        s2, c2 = _construct_direct_sa(rng)
        if c2 is not None and s2 < best_score:
            best_score = s2
            best_coefs = c2
    
    if best_coefs is None:
        # Fallback
        coefs = [0] * 41
        for i in range(12):
            coefs[i] = 1
        return coefs
    
    return best_coefs
