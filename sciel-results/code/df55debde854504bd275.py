import random
import numpy as np
import itertools
import time

D = 11
N = 11

# ---------- timing guard ----------
_T0 = None
def _elapsed():
    return time.perf_counter() - _T0
def _timeleft(budget=110.0):
    return budget - _elapsed()


def _verify(verts):
    """Verify min distance >= max norm; return (valid, k)."""
    if len(verts) < 2:
        return True, len(verts)
    nn = (verts ** 2).sum(axis=1).max()
    dists = np.sqrt(((verts[:, None, :] - verts[None, :, :]) ** 2).sum(axis=2))
    off = np.ones(dists.shape, dtype=bool)
    np.fill_diagonal(off, False)
    mind = dists[off].min()
    return mind + 1e-9 >= np.sqrt(nn), len(verts)


def _all_weight2():
    """Return all 110 distinct weight-2 vectors (two coords in {+1,-1})."""
    cands = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                cands.append(v)
    return np.array(cands, dtype=np.int64)


def _try_add_unit_vectors(base, max_trials=5000):
    """Try to add unit vectors (norm=1) compatible with base (weight-2 set).
    A unit vector e_i is compatible with weight-2 vector v iff:
      - if v[i] != 0: need (1 - v[i])^2 + sum_{j!=i} v[j]^2 >= 2
        i.e., if v[i]=1: 0 + 1 = 1 < 2 FAIL
        if v[i]=-1: 4 + 1 = 5 >= 2 OK
      - if v[i] == 0: 1 + 2 = 3 >= 2 OK
    So e_i is compatible with ALL weight-2 vectors iff no weight-2 vector has +1 at position i.
    """
    if base is None or len(base) == 0:
        return np.zeros((0, D), dtype=np.int64)
    
    # For each coordinate i, check if any base vector has value +1 at position i
    # If yes, e_i cannot be added (distance to that vector would be 1 < sqrt(2))
    # If no, e_i can be added
    
    # Also need e_i compatible with any other unit vectors we add
    # Two unit vectors e_i, e_j: d^2 = 2 if i!=j, 0 if i==j
    # So we can add at most one unit vector per coordinate, and they must be distinct coords
    # But also need d(e_i, e_j)^2 = 2 >= 1 (max norm among units is 1, so need d>=1) -> 2>=1 OK
    
    # Actually max norm in combined set is sqrt(2) from weight-2 vectors
    # So need all pairwise distances >= sqrt(2), i.e., d^2 >= 2
    # e_i and e_j (i!=j): d^2 = 2 >= 2 OK
    # e_i and weight-2 v: need d^2 >= 2 as computed above
    
    added = []
    for i in range(D):
        # Check if e_i is compatible with all base vectors
        compatible = True
        for v in base:
            if v[i] == 1:
                # d^2 = (1-1)^2 + sum of other v[j]^2 = 0 + 1 = 1 < 2
                compatible = False
                break
        if compatible:
            e = np.zeros(D, dtype=np.int64)
            e[i] = 1
            added.append(e)
    
    if not added:
        return np.zeros((0, D), dtype=np.int64)
    return np.array(added, dtype=np.int64)


def construct():
    global _T0
    _T0 = time.perf_counter()
    np.random.seed(12345)
    
    # All 110 weight-2 vectors are pairwise compatible (verified: any two distinct
    # weight-2 vectors have d^2 >= 2 since they differ in at least one coordinate
    # where one has ±1 and the other has 0 or opposite sign)
    base = _all_weight2()
    
    # Verify base
    valid, k = _verify(base)
    if not valid:
        # Should not happen, but fallback
        return base
    
    # Try to add unit vectors
    units = _try_add_unit_vectors(base)
    
    if len(units) > 0:
        combined = np.vstack([base, units])
        valid_c, k_c = _verify(combined)
        if valid_c:
            return combined.astype(np.int64)
    
    # Try adding more weight-2-like vectors with different norms
    # Try vectors with one coordinate = 2 (norm=2)
    # v with v[i]=2: norm=2, d^2 to weight-2 w:
    #   if w[i]!=0: (2-w[i])^2 + sum_{j!=i} w[j]^2
    #     w[i]=1: 1+1=2 OK, w[i]=-1: 9+1=10 OK
    #   if w[i]==0: 4+2=6 OK
    # So v[i]=2 is compatible with all weight-2 vectors!
    # Also two such vectors v[i]=2, v[j]=2: d^2 = 4+4=8 >= 2 OK
    # v[i]=2 and e_k: if k=i: (2-1)^2=1 < 2 FAIL (but we only add e_k if no +1 at k in base,
    #   but v[i]=2 has no +1, it has 2. Let me recheck: d(e_k, v) where v[k]=2:
    #   (1-2)^2 = 1 < 2 FAIL)
    # So v[i]=2 conflicts with e_i.
    
    # Let's try adding vectors with a single coordinate = 2
    extra = []
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        # Check compatibility with base (weight-2)
        ok = True
        for w in base:
            d2 = int(np.dot(v - w, v - w))
            if d2 < 2:
                ok = False
                break
        if ok:
            # Check with units
            ok2 = True
            for u in units:
                d2 = int(np.dot(v - u, v - u))
                if d2 < 2:
                    ok2 = False
                    break
            if ok2:
                extra.append(v)
    
    # Check extra among themselves
    final_extra = []
    for v in extra:
        ok = True
        for w in final_extra:
            d2 = int(np.dot(v - w, v - w))
            if d2 < 2:
                ok = False
                break
        if ok:
            final_extra.append(v)
    
    if final_extra:
        combined2 = np.vstack([base, units, np.array(final_extra)])
        valid2, k2 = _verify(combined2)
        if valid2:
            return combined2.astype(np.int64)
    
    # Try vectors with one coordinate = -2
    extra_neg = []
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = -2
        ok = True
        for w in base:
            d2 = int(np.dot(v - w, v - w))
            if d2 < 2:
                ok = False
                break
        if ok:
            ok2 = True
            for u in units:
                d2 = int(np.dot(v - u, v - u))
                if d2 < 2:
                    ok2 = False
                    break
            if ok2:
                for w in final_extra:
                    d2 = int(np.dot(v - w, v - w))
                    if d2 < 2:
                        ok2 = False
                        break
                if ok2:
                    extra_neg.append(v)
    
    if extra_neg:
        all_extra = final_extra + extra_neg
        combined3 = np.vstack([base, units, np.array(all_extra)])
        valid3, k3 = _verify(combined3)
        if valid3:
            return combined3.astype(np.int64)
    
    # Fallback: just return base
    return base.astype(np.int64)


if __name__ == "__main__":
    import time as _t
    _t0 = _t.time()
    a = construct()
    print("construct done in", _t.time() - _t0, "s; shape", a.shape)
    print("verify:", _verify(a))
