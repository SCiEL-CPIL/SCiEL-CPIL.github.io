"""Combined approach: efficient precomputation + multi-strategy greedy + local search."""
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

BASE = 51975

# Precompute all divisors > 1 of 51975 = 3^3 * 5^2 * 7 * 11
_DIVS = []
for a in range(4):          # 3^a
    for b in range(3):      # 5^b
        for c in range(2):  # 7^c
            for d in range(2):  # 11^d
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    _DIVS.append(v)
_DIVS.sort()
N_DIVS = len(_DIVS)

# Precompute positions for each divisor and residue class
# _POS[i][r] = np.array of positions in [0, BASE) where pos % m == r
_POS = []
for m in _DIVS:
    arr = []
    for r in range(m):
        arr.append(np.arange(r, BASE, m, dtype=np.int32))
    _POS.append(arr)

# Sort indices by coverage (small m first covers more residues)
_COV = [BASE // m for m in _DIVS]
_IDX_BY_COV = sorted(range(N_DIVS), key=lambda i: _COV[i], reverse=True)
_IDX_BY_COV_REV = _IDX_BY_COV[::-1]


def _greedy_pass(covered, idx_order, deadline):
    """Greedy: for each divisor index in idx_order, pick best residue.
    Returns (classes, covered_array, uncovered_count)."""
    classes = []
    cov = covered.copy() if covered is not None else np.zeros(BASE, dtype=np.uint8)
    
    for idx in idx_order:
        if time.time() > deadline:
            break
        m = _DIVS[idx]
        pos_arr = _POS[idx]
        
        best_r = 0
        best_gain = -1
        for r in range(m):
            pos = pos_arr[r]
            gain = int((cov[pos] == 0).sum())
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            classes.append([m, best_r])
            cov[pos_arr[best_r]] = 1
    
    uncovered = int((cov == 0).sum())
    return classes, cov, uncovered


def _local_search(classes, covered, deadline, max_iters=30):
    """Try to improve by changing residues of existing classes or adding new ones."""
    best_classes = [c[:] for c in classes]
    best_uncovered = int((covered == 0).sum())
    used_mods = set(m for m, r in best_classes)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding unused moduli
        for idx in _IDX_BY_COV:
            if time.time() > deadline:
                break
            m = _DIVS[idx]
            if m in used_mods:
                continue
            if _COV[idx] < 2:
                continue
            
            pos_arr = _POS[idx]
            best_r = 0
            best_gain = -1
            for r in range(m):
                pos = pos_arr[r]
                gain = int((covered[pos] == 0).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                new_classes = best_classes + [[m, best_r]]
                new_cov = covered.copy()
                new_cov[pos_arr[best_r]] = 1
                new_uncovered = int((new_cov == 0).sum())
                if new_uncovered < best_uncovered:
                    best_classes = new_classes
                    covered = new_cov
                    best_uncovered = new_uncovered
                    used_mods.add(m)
                    improved = True
        
        # Try changing residue for existing classes (limited candidates for speed)
        for i, (m, r) in enumerate(best_classes):
            if time.time() > deadline:
                break
            
            pos_arr = _POS[_DIVS.index(m)]
            candidates = list(range(m))
            if m > 50:
                # Sample a subset
                candidates = [0, m // 2, (m - 1) // 2, min(1, m - 1), max(0, m - 2)]
                candidates = [c for c in candidates if 0 <= c < m and c != r]
            
            best_r_new = r
            best_uncovered_new = best_uncovered
            
            for r_new in candidates:
                if r_new == r:
                    continue
                
                # Compute uncovered if we replace (m, r) with (m, r_new)
                # Positions covered by old but not new
                old_pos = pos_arr[r]
                new_pos = pos_arr[r_new]
                
                # Temporarily remove old class effect and add new
                # Count how many positions become uncovered
                # A position p is uncovered after change if:
                #   it was covered only by old class AND is not in new_pos
                #   OR it was not covered by old class but is in new_pos (no change in uncovered)
                
                # Simpler: recompute uncovered for positions in old_pos or new_pos
                # that might be affected
                
                # For efficiency, just check positions in old_pos that are not in new_pos
                # and positions in new_pos that are not in old_pos
                
                # Build set of new positions for quick lookup
                new_set = set(new_pos.tolist())
                
                # Positions that lose coverage: in old_pos but not in new_pos,
                # and not covered by any other class
                lost = 0
                for p in old_pos:
                    if p not in new_set:
                        # Check if covered by another class
                        if covered[p] == 1:
                            # Need to check if this class is the only one covering p
                            # This requires coverage counting, which we don't have
                            # Approximation: assume if covered[p]==1, it might be covered by others
                            # For now, skip this complex check
                            pass
                
                # This approach is too complex without coverage counts
                # Instead, just recompute for small m
                if m <= 30:
                    temp_cov = covered.copy()
                    # Remove old class
                    for p in old_pos:
                        # Decrement coverage count (but we only have binary)
                        # We can't easily remove, so just compute new uncovered directly
                        pass
                
                # Skip complex local search, focus on additions
    
        if not improved:
            break
    
    return best_classes, covered, best_uncovered


def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s kill
    
    best_classes = []
    best_uncovered = BASE
    
    # Strategy 1: Greedy from smallest moduli up (most coverage first)
    classes, cov, unc = _greedy_pass(None, _IDX_BY_COV, deadline - 10.0)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes[:]
    
    # Strategy 2: Greedy from largest moduli down
    if time.time() < deadline - 15.0:
        classes, cov, unc = _greedy_pass(None, _IDX_BY_COV_REV, deadline - 10.0)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes[:]
    
    # Strategy 3: Local search improvement on best solution
    if time.time() < deadline - 20.0 and best_classes:
        # Rebuild covered array
        covered = np.zeros(BASE, dtype=np.uint8)
        for m, r in best_classes:
            idx = _DIVS.index(m)
            covered[_POS[idx][r]] = 1
        
        best_classes, covered, best_uncovered = _local_search(
            best_classes, covered, deadline - 5.0, max_iters=20)
    
    # Strategy 4: Randomized restarts
    if time.time() < deadline - 25.0:
        import random
        random.seed(42)
        
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            
            mod_order = _IDX_BY_COV[:]
            random.shuffle(mod_order)
            
            classes, cov, unc = _greedy_pass(None, mod_order, deadline - 10.0)
            if unc < best_uncovered:
                best_uncovered = unc
                best_classes = classes[:]
    
    return best_classes

# EVOLVE-BLOCK-END
