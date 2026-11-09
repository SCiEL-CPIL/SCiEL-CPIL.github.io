import time
import random
from collections import defaultdict

def _count_sq_nonzero(poly_dict):
    """Count nonzero terms in exact square of polynomial represented as {exp: coeff}."""
    items = list(poly_dict.items())
    n = len(items)
    sq = {}
    for i in range(n):
        e1, c1 = items[i]
        for j in range(i, n):
            e2, c2 = items[j]
            deg = e1 + e2
            sq[deg] = sq.get(deg, 0) + c1 * c2
    return sum(1 for v in sq.values() if v != 0)

def _valid(poly_dict):
    """Check validity: exactly 12 nonzero terms, exps in [0,40], coeffs in [-64,64] nonzero."""
    if len(poly_dict) != 12:
        return False
    for e, c in poly_dict.items():
        if not (0 <= e <= 40):
            return False
        if c == 0:
            return False
        if not (-64 <= c <= 64):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin
    
    best_score = 144
    best_poly = None
    
    random.seed(12345)
    
    # Generate diverse initial candidates
    candidates = []
    
    # AP supports with sign patterns
    for base in range(0, 20):
        for step in range(1, 5):
            exps = [base + i * step for i in range(12)]
            if exps[-1] > 40:
                continue
            for pat in [
                [1]*12,
                [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
                [1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
                [1,2,1,2,1,2,1,2,1,2,1,2],
                [2,1,-1,-2,-1,-2,1,2,1,2,-1,-2],
            ]:
                pd = {e: pat[i] for i, e in enumerate(exps)}
                if _valid(pd):
                    candidates.append(pd)
    
    # Dense clusters
    for off in range(0, 30):
        exps = [off + i for i in range(12)]
        if exps[-1] > 40:
            continue
        for signs in [[1]*12, [1,-1]*6, [1,1,-1,-1]*3]:
            pd = {e: signs[i] for i, e in enumerate(exps)}
            if _valid(pd):
                candidates.append(pd)
    
    # Random
    for _ in range(300):
        exps = sorted(random.sample(range(41), 12))
        pd = {e: random.choice([-2,-1,-1,1,1,2]) for e in exps}
        if _valid(pd):
            candidates.append(pd)
    
    # Evaluate
    for pd in candidates:
        s = _count_sq_nonzero(pd)
        if s < best_score:
            best_score = s
            best_poly = dict(pd)
    
    if best_poly is None:
        best_poly = {i: 1 for i in range(12)}
        best_score = _count_sq_nonzero(best_poly)
    
    current = dict(best_poly)
    cur_score = best_score
    
    # Simulated annealing
    T = 5.0
    cool = 0.998
    
    while time.time() < deadline:
        new = dict(current)
        mtype = random.random()
        
        if mtype < 0.45:
            # Perturb a coefficient
            es = list(new.keys())
            e = random.choice(es)
            nc = random.randint(-64, 64)
            if nc == 0:
                nc = random.choice([1, -1])
            new[e] = nc
        elif mtype < 0.75:
            # Move a term to new exponent
            es = list(new.keys())
            e = random.choice(es)
            c = new[e]
            ne = random.randint(0, 40)
            if ne in new and ne != e:
                continue
            del new[e]
            new[ne] = c
        else:
            # Swap exponents
            es = list(new.keys())
            if len(es) < 2:
                continue
            e1, e2 = random.sample(es, 2)
            c1, c2 = new[e1], new[e2]
            del new[e1]
            del new[e2]
            new[e2] = c1
            new[e1] = c2
        
        if not _valid(new):
            continue
        
        ns = _count_sq_nonzero(new)
        d = ns - cur_score
        
        if d <= 0 or random.random() < 2.718281828 ** (-d / max(T, 0.01)):
            current = new
            cur_score = ns
            if cur_score < best_score:
                best_score = cur_score
                best_poly = dict(current)
        
        T *= cool
        if T < 0.05:
            T = 0.05
    
    # Convert to list
    if best_poly is None:
        return [1] * 12
    
    max_e = max(best_poly.keys())
    res = [0] * (max_e + 1)
    for e, c in best_poly.items():
        res[e] = c
    
    # Trim trailing zeros
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    
    if len(res) > 41:
        res = res[:41]
    
    # Verify
    nz = [x for x in res if x != 0]
    if len(nz) != 12:
        return [1] * 12
    
    for i in range(len(res)):
        res[i] = max(-64, min(64, res[i]))
    
    return res
