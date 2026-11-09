"""Odd covering systems with moduli dividing 51975; local-search over greedy seed."""
import time
import random
import numpy as np

BASE = 51975
# all odd divisors > 1 of 51975
_DIVS = []
for _d in range(3, BASE + 1, 2):
    if BASE % _d == 0:
        _DIVS.append(_d)
_DIVS.sort()

def _uncovered_count(covered):
    return int(covered.count(0))

def _build_greedy(order, rng=None):
    covered = bytearray(BASE)
    classes = []
    for m in order:
        best_r = 0
        best_c = -1
        # choose residue maximizing newly covered
        for r in range(m):
            c = covered[r::m].count(0)
            if c > best_c:
                best_c = c
                best_r = r
            if best_c == BASE // m:
                break
        classes.append([m, best_r])
        covered[best_r::m] = b'\x01' * len(range(best_r, BASE, m))
    return classes, covered

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s

    rng = random.Random(12345)

    # Initial greedy in increasing modulus order
    classes, covered = _build_greedy(_DIVS)
    best_classes = [list(c) for c in classes]
    best_score = _uncovered_count(covered)

    # Precompute coverage masks for each modulus-residue pair using NumPy
    # mask[m_idx][r] = boolean array of length BASE indicating which residues are covered
    # This is too memory-heavy (51975 * sum(m for m in divs)), so we'll compute on-the-fly
    # but cache per-modulus masks.
    
    # Actually, let's precompute for each modulus m, the set of indices for each residue r
    # indices[m][r] = np.array([r, r+m, r+2m, ...])
    # But that's also large. Let's use a different approach.
    
    # For fast evaluation, precompute for each modulus m:
    #   - the number of residues in each class: BASE // m (or ceil)
    #   - use numpy for vectorized coverage computation
    
    def fast_evaluate(cls_list):
        """Compute uncovered count using numpy for speed."""
        cov = np.zeros(BASE, dtype=np.uint8)
        for m, r in cls_list:
            cov[r::m] = 1
        return int(np.count_nonzero(cov == 0))
    
    def fast_evaluate_with_change(cls_list, idx, new_r):
        """Compute uncovered count when class idx has residue new_r."""
        cov = np.zeros(BASE, dtype=np.uint8)
        for j, (m, r) in enumerate(cls_list):
            if j == idx:
                cov[new_r::m] = 1
            else:
                cov[r::m] = 1
        return int(np.count_nonzero(cov == 0))
    
    def fast_evaluate_swap(cls_list, i, j):
        """Compute uncovered count when residues of classes i and j are swapped."""
        cov = np.zeros(BASE, dtype=np.uint8)
        m_i, r_i = cls_list[i]
        m_j, r_j = cls_list[j]
        for k, (m, r) in enumerate(cls_list):
            if k == i:
                cov[r_j::m] = 1
            elif k == j:
                cov[r_i::m] = 1
            else:
                cov[r::m] = 1
        return int(np.count_nonzero(cov == 0))
    
    def fast_evaluate_modswap(cls_list, i, j):
        """Compute uncovered count when moduli of classes i and j are swapped (residues adjusted)."""
        cov = np.zeros(BASE, dtype=np.uint8)
        m_i, r_i = cls_list[i]
        m_j, r_j = cls_list[j]
        # After swap: class i has modulus m_j with residue r_i % m_j, class j has modulus m_i with residue r_j % m_i
        new_r_i = r_i % m_j
        new_r_j = r_j % m_i
        for k, (m, r) in enumerate(cls_list):
            if k == i:
                cov[new_r_i::m_j] = 1
            elif k == j:
                cov[new_r_j::m_i] = 1
            else:
                cov[r::m] = 1
        return int(np.count_nonzero(cov == 0))
    
    cur_classes = [list(c) for c in classes]
    cur_score = best_score
    n = len(cur_classes)
    
    # Precompute: for each class index, store (m, r) as tuple for faster access
    # Also precompute the "base coverage" without each class for incremental updates
    
    # Strategy: iterative improvement with multiple move types
    # Move 1: change residue of one class
    # Move 2: swap residues between two classes  
    # Move 3: swap moduli between two classes
    
    improved = True
    no_improve_rounds = 0
    max_no_improve = 50  # stop if no improvement for many rounds
    
    while improved and time.time() < deadline and no_improve_rounds < max_no_improve:
        improved = False
        no_improve_rounds += 1
        
        # Shuffle class order to vary neighborhood
        idx_order = list(range(n))
        rng.shuffle(idx_order)
        
        # Move 1: change residue of one class
        for i in idx_order:
            if time.time() > deadline:
                break
            m_i, r_i = cur_classes[i]
            best_r = r_i
            best_sc = cur_score
            
            # Try all residues for class i
            for r in range(m_i):
                if r == r_i:
                    continue
                sc = fast_evaluate_with_change(cur_classes, i, r)
                if sc < best_sc:
                    best_sc = sc
                    best_r = r
            
            if best_r != r_i:
                cur_classes[i][1] = best_r
                cur_score = best_sc
                improved = True
                no_improve_rounds = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_classes = [list(c) for c in cur_classes]
            
            if cur_score == 0:
                return best_classes
        
        # Move 2: swap residues between two classes
        if time.time() < deadline and cur_score > 0:
            for i in idx_order:
                if time.time() > deadline:
                    break
                for j in idx_order:
                    if j <= i:
                        continue
                    if time.time() > deadline:
                        break
                    m_i, r_i = cur_classes[i]
                    m_j, r_j = cur_classes[j]
                    
                    # Only valid if residues fit in the other's modulus
                    if r_i < m_j and r_j < m_i:
                        sc = fast_evaluate_swap(cur_classes, i, j)
                        if sc < cur_score:
                            # Apply the swap
                            cur_classes[i][1] = r_j
                            cur_classes[j][1] = r_i
                            cur_score = sc
                            improved = True
                            no_improve_rounds = 0
                            if cur_score < best_score:
                                best_score = cur_score
                                best_classes = [list(c) for c in cur_classes]
        
        # Move 3: swap moduli between two classes
        if time.time() < deadline and cur_score > 0:
            for i in idx_order:
                if time.time() > deadline:
                    break
                for j in idx_order:
                    if j <= i:
                        continue
                    if time.time() > deadline:
                        break
                    m_i, r_i = cur_classes[i]
                    m_j, r_j = cur_classes[j]
                    
                    # Check that after swap, residues are valid
                    new_r_i = r_i % m_j
                    new_r_j = r_j % m_i
                    
                    sc = fast_evaluate_modswap(cur_classes, i, j)
                    if sc < cur_score:
                        # Apply the modulus swap with adjusted residues
                        cur_classes[i][0] = m_j
                        cur_classes[i][1] = new_r_i
                        cur_classes[j][0] = m_i
                        cur_classes[j][1] = new_r_j
                        cur_score = sc
                        improved = True
                        no_improve_rounds = 0
                        if cur_score < best_score:
                            best_score = cur_score
                            best_classes = [list(c) for c in cur_classes]
    
    return best_classes
