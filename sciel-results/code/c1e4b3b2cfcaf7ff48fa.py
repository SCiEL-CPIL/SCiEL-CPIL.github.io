import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
M = len(POINTS)  # 256

# ---------- Precompute isosceles relationships using bitsets for speed ----------
# iso_mask[v] = bitset of pairs (a,b) with a<b, a,b!=v, such that (v,a,b) is isosceles
# We represent each pair (a,b) as a single bit index: pair_idx = a*M + b (with a<b)
# But that's too large. Instead, we precompute for each v the list of (a,b) pairs,
# and use a different approach for fast checking.

# Better: Precompute conflict[v] = set of pairs (a,b) that would make v incompatible
# i.e., if both a and b are in S, then v cannot be added (triangle v,a,b is isosceles)
# This is the same as COVER[v] from the original.

def _build_cover_bitsets():
    """Build cover lists and also precompute pair-to-points mapping for MDS."""
    xs = np.array([p[0] for p in POINTS], dtype=np.int64)
    ys = np.array([p[1] for p in POINTS], dtype=np.int64)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy
    A = np.arange(M)
    iu_a, iu_b = np.triu_indices(M, k=1)
    cover = []
    for p in range(M):
        d_pa = D2[p, A][:, None]
        d_pb = D2[p, A][None, :]
        iso = (d_pa == d_pb) | (d_pa == D2) | (d_pb == D2)
        a = iu_a; b = iu_b
        m = (a != p) & (b != p)
        a = a[m]; b = b[m]
        sel = iso[a, b]
        cover.append(list(zip(a[sel].tolist(), b[sel].tolist())))
    return cover

COVER = _build_cover_bitsets()

# For MDS: for each point p not in S, we need at least one pair (a,b) in S such that
# (p,a,b) is isosceles. So p is "dominated" by S iff ∃(a,b) ∈ COVER[p] with a∈S and b∈S.
# This is a 2-set cover / hypergraph domination problem.

def _make_inS(S):
    inS = [False] * M
    for v in S:
        inS[v] = True
    return inS

def _is_no_isosceles(S, inS):
    """Check if S is no-isosceles: no three points in S form an isosceles triangle."""
    for v in S:
        for a, b in COVER[v]:
            if inS[a] and inS[b]:
                return False
    return True

def _is_maximal(S, inS):
    """Check if S is maximal: every point not in S is covered (has an isosceles pair in S)."""
    for p in range(M):
        if inS[p]:
            continue
        covered = False
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                covered = True
                break
        if not covered:
            return False
    return True

def _is_valid_maximal(S):
    """Full validation: S is no-isosceles and maximal."""
    inS = _make_inS(S)
    return _is_no_isosceles(S, inS) and _is_maximal(S, inS)

def _greedy_mds():
    """
    Greedy Minimum Dominating Set approach:
    1. Start with empty S.
    2. While not all points are dominated, pick the point that dominates the most undominated points
       (among points that can be added without creating an isosceles triangle).
    3. Add that point to S.
    
    A point v can be added if it doesn't create an isosceles triangle with existing points.
    Adding v "dominates" all points p for which ∃b∈S such that (p,v,b) is isosceles,
    plus all points p for which ∃a,b∈S such that (p,a,b) is isosceles (already dominated).
    Actually, adding v helps dominate p if there exists b∈S such that (p,v,b) is isosceles.
    """
    S = set()
    inS = _make_inS(S)
    
    # Track which points are dominated
    dominated = [False] * M
    # Points in S are trivially "dominated" (they don't need covering)
    
    while True:
        # Check if all points are dominated
        all_dominated = True
        for p in range(M):
            if inS[p]:
                continue
            if not dominated[p]:
                all_dominated = False
                break
        if all_dominated:
            break
        
        # Find the best point to add: maximizes number of newly dominated points
        best_v = -1
        best_gain = 0
        
        for v in range(M):
            if inS[v]:
                continue
            # Check if v can be added (no isosceles triangle with existing S)
            can_add = True
            for a, b in COVER[v]:
                if inS[a] and inS[b]:
                    can_add = False
                    break
            if not can_add:
                continue
            
            # Count how many undominated points would become dominated
            gain = 0
            for p in range(M):
                if inS[p] or dominated[p]:
                    continue
                # Check if adding v would dominate p:
                # Need ∃b∈S such that (p,v,b) is isosceles
                for a, b in COVER[p]:
                    if a == v and inS[b]:
                        gain += 1
                        break
                    elif b == v and inS[a]:
                        gain += 1
                        break
            
            if gain > best_gain:
                best_gain = gain
                best_v = v
        
        if best_v == -1:
            # No more points can be added; S is maximal but may not be valid
            # This shouldn't happen if we started from empty and always added compatible points
            break
        
        # Add best_v
        S.add(best_v)
        inS[best_v] = True
        
        # Update dominated status
        for p in range(M):
            if inS[p] or dominated[p]:
                continue
            for a, b in COVER[p]:
                if inS[a] and inS[b]:
                    dominated[p] = True
                    break
    
    return S

def _shrink_minimal(S):
    """Remove points while maintaining maximality (inclusion-minimal maximal set)."""
    S = set(S)
    inS = _make_inS(S)
    improved = True
    while improved:
        improved = False
        items = list(S)
        random.shuffle(items)
        for x in items:
            # Try removing x
            inS[x] = False
            S.discard(x)
            if _is_maximal(S, inS):
                # Still maximal, keep x removed
                improved = True
            else:
                # Not maximal, put x back
                inS[x] = True
                S.add(x)
    return S

def _local_search_swap(S):
    """Try to improve S by swapping one point out and another in, reducing size."""
    S = set(S)
    inS = _make_inS(S)
    best_S = S
    best_size = len(S)
    
    items = list(S)
    random.shuffle(items)
    
    for x in items:
        # Try removing x
        inS[x] = False
        S.discard(x)
        
        if not _is_no_isosceles(S, inS):
            inS[x] = True
            S.add(x)
            continue
        
        # Now try to add some y not in S that maintains no-isosceles
        # and doesn't increase size (we're trying to keep same size or reduce)
        # Actually, we want to find a y such that after removing x and adding y,
        # the set is still no-isosceles and maximal, and ideally smaller.
        # But removing x already reduces size by 1. We just need to check if
        # the resulting set is maximal. If not, we need to add points back.
        
        # Check if S (without x) is maximal
        if _is_maximal(S, inS):
            # Great! We reduced size by 1 and maintained maximality
            return S  # S no longer contains x
        
        # S without x is not maximal. Try to find minimal additions to make it maximal.
        # This is complex, so for now, just restore x
        inS[x] = True
        S.add(x)
    
    return best_S

def construct():
    t0 = time.time()
    deadline = t0 + 285.0  # safety margin well under the 330s kill
    
    best = None
    best_size = M + 1
    
    # Strategy 1: Greedy MDS from scratch, multiple restarts
    for _ in range(10):
        S = _greedy_mds()
        if _is_valid_maximal(S):
            S = _shrink_minimal(S)
            if len(S) < best_size:
                best = S
                best_size = len(S)
    
    # Strategy 2: Start from greedy maximal (original approach), then shrink
    for _ in range(5):
        # Greedy maximal: add points in random order if compatible
        S = set()
        inS = _make_inS(S)
        order = list(range(M))
        random.shuffle(order)
        for v in order:
            can_add = True
            for a, b in COVER[v]:
                if inS[a] and inS[b]:
                    can_add = False
                    break
            if can_add:
                S.add(v)
                inS[v] = True
        S = _shrink_minimal(S)
        if len(S) < best_size:
            best = S
            best_size = len(S)
    
    if best is None:
        # Fallback: just return a valid maximal set
        S = _greedy_mds()
        if not _is_valid_maximal(S):
            # Emergency: build a trivial maximal set
            S = set()
            inS = _make_inS(S)
            order = list(range(M))
            for v in order:
                can_add = True
                for a, b in COVER[v]:
                    if inS[a] and inS[b]:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
                    inS[v] = True
            best = S
            best_size = len(S)
    
    # Iterated local search: try swaps to reduce size
    plateau = 0
    while time.time() < deadline:
        S = set(best)
        inS = _make_inS(S)
        
        # Perturbation: remove 1-3 random points
        k = random.randint(1, 3)
        items = list(S)
        random.shuffle(items)
        for x in items[:k]:
            S.discard(x)
            inS[x] = False
        
        # Repair: add compatible points to restore maximality
        # First ensure no-isosceles
        # Then add points to make it maximal
        order = list(range(M))
        random.shuffle(order)
        for v in order:
            if inS[v]:
                continue
            can_add = True
            for a, b in COVER[v]:
                if inS[a] and inS[b]:
                    can_add = False
                    break
            if can_add:
                S.add(v)
                inS[v] = True
        
        # Shrink to minimal
        S = _shrink_minimal(S)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
            plateau = 0
        else:
            plateau += 1
            if plateau > 50:
                break
    
    # Final shrink
    if time.time() < deadline - 2.0:
        best = _shrink_minimal(best)
    
    return [POINTS[i] for i in sorted(best)]
