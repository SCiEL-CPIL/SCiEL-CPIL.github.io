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

def _build_greedy(order, rng=None):
    covered = bytearray(BASE)
    classes = []
    for m in order:
        best_r = 0
        best_c = -1
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
    deadline = t0 + 110.0

    rng = random.Random(12345)

    # Initial greedy in increasing modulus order
    classes, covered = _build_greedy(_DIVS)
    best_classes = [list(c) for c in classes]
    best_score = int(covered.count(0))

    cur_classes = [list(c) for c in classes]
    cur_score = best_score
    n = len(cur_classes)

    # Precompute: for each class i and each possible residue r of class i,
    # how many of the positions r::m_i are NOT covered by any other class.
    # other_covered[i][r] = count of positions in r::m_i covered by classes j != i
    
    def compute_other_covered(cls_list):
        """For each class i, compute for each residue r: count of positions in r::m_i 
        that are covered by other classes (not class i itself)."""
        n = len(cls_list)
        other_cov = []
        for i in range(n):
            m_i = cls_list[i][0]
            # Build coverage from all classes except i
            cov = np.zeros(BASE, dtype=np.uint8)
            for j in range(n):
                if j == i:
                    continue
                m_j, r_j = cls_list[j]
                cov[r_j::m_j] = 1
            # For each residue r of class i, count covered positions
            arr = np.zeros(m_i, dtype=np.int32)
            for r in range(m_i):
                positions = np.arange(r, BASE, m_i)
                arr[r] = int(np.sum(cov[positions]))
            other_cov.append(arr)
        return other_cov

    def get_score_from_other_cov(other_cov, cls_list):
        """Compute total uncovered given other_cov and current assignments."""
        total_uncovered = 0
        for i, (m, r) in enumerate(cls_list):
            # Positions in r::m covered by others = other_cov[i][r]
            # Total positions in r::m = BASE // m (since m divides BASE)
            total_positions = BASE // m
            # Uncovered by others = total_positions - other_cov[i][r]
            # But class i covers all positions in r::m, so uncovered by i = 0
            # Actually: uncovered = positions NOT covered by anyone
            # = total_positions - (covered by i OR covered by others)
            # covered by i = total_positions (all of r::m)
            # So uncovered = 0 for those positions? No...
            # Wait: uncovered residues are those not covered by ANY class.
            # If class i covers r::m_i, then all positions in r::m_i are covered.
            # So the uncovered count is just the count of positions not in any class's residue.
            # Let me recompute properly.
            pass
        # Just compute directly
        cov = np.zeros(BASE, dtype=np.uint8)
        for m, r in cls_list:
            cov[r::m] = 1
        return int(np.sum(cov == 0))

    # Faster approach: maintain a coverage count array
    # cov_count[x] = number of classes covering residue x
    # Uncovered = count of x where cov_count[x] == 0
    
    cov_count = np.zeros(BASE, dtype=np.int16)
    for m, r in cur_classes:
        cov_count[r::m] += 1
    cur_score = int(np.sum(cov_count == 0))
    best_score = cur_score
    best_classes = [list(c) for c in cur_classes]

    def try_change_residue(cls_list, cov_cnt, i, new_r):
        """Try changing class i's residue to new_r. Returns new score or None if not better."""
        m_i, old_r = cls_list[i]
        if new_r == old_r:
            return None
        
        # Remove old class
        old_positions = np.arange(old_r, BASE, m_i)
        cov_cnt[old_positions] -= 1
        
        # Add new class
        new_positions = np.arange(new_r, BASE, m_i)
        cov_cnt[new_positions] += 1
        
        score = int(np.sum(cov_cnt == 0))
        return score, old_positions, new_positions

    def apply_change(cls_list, cov_cnt, i, new_r, old_positions, new_positions):
        """Apply the change (already computed)."""
        cls_list[i][1] = new_r

    # Local search with incremental updates
    improved = True
    no_improve_rounds = 0
    max_no_improve = 30

    while improved and time.time() < deadline and no_improve_rounds < max_no_improve:
        improved = False
        no_improve_rounds += 1
        
        idx_order = list(range(n))
        rng.shuffle(idx_order)
        
        # Move 1: change residue of one class
        for i in idx_order:
            if time.time() > deadline:
                break
            m_i, r_i = cur_classes[i]
            best_r = r_i
            best_sc = cur_score
            
            # Try all residues
            for r in range(m_i):
                if r == r_i:
                    continue
                # Compute new score incrementally
                old_positions = np.arange(r_i, BASE, m_i)
                new_positions = np.arange(r, BASE, m_i)
                
                # Temporarily remove old, add new
                cov_count[old_positions] -= 1
                cov_count[new_positions] += 1
                sc = int(np.sum(cov_count == 0))
                
                # Revert
                cov_count[new_positions] -= 1
                cov_count[old_positions] += 1
                
                if sc < best_sc:
                    best_sc = sc
                    best_r = r
            
            if best_r != r_i:
                # Apply the best change
                m_i, r_i = cur_classes[i]
                old_positions = np.arange(r_i, BASE, m_i)
                new_positions = np.arange(best_r, BASE, m_i)
                cov_count[old_positions] -= 1
                cov_count[new_positions] += 1
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
                    
                    if r_i >= m_j or r_j >= m_i:
                        continue
                    
                    # Swap residues
                    # Remove both, add swapped
                    pos_i_old = np.arange(r_i, BASE, m_i)
                    pos_j_old = np.arange(r_j, BASE, m_j)
                    pos_i_new = np.arange(r_j, BASE, m_i)
                    pos_j_new = np.arange(r_i, BASE, m_j)
                    
                    cov_count[pos_i_old] -= 1
                    cov_count[pos_j_old] -= 1
                    cov_count[pos_i_new] += 1
                    cov_count[pos_j_new] += 1
                    
                    sc = int(np.sum(cov_count == 0))
                    
                    # Revert
                    cov_count[pos_i_new] -= 1
                    cov_count[pos_j_new] -= 1
                    cov_count[pos_i_old] += 1
                    cov_count[pos_j_old] += 1
                    
                    if sc < cur_score:
                        # Apply
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
                    
                    new_r_i = r_i % m_j
                    new_r_j = r_j % m_i
                    
                    # Remove both old, add both new
                    pos_i_old = np.arange(r_i, BASE, m_i)
                    pos_j_old = np.arange(r_j, BASE, m_j)
                    pos_i_new = np.arange(new_r_i, BASE, m_j)
                    pos_j_new = np.arange(new_r_j, BASE, m_i)
                    
                    cov_count[pos_i_old] -= 1
                    cov_count[pos_j_old] -= 1
                    cov_count[pos_i_new] += 1
                    cov_count[pos_j_new] += 1
                    
                    sc = int(np.sum(cov_count == 0))
                    
                    # Revert
                    cov_count[pos_i_new] -= 1
                    cov_count[pos_j_new] -= 1
                    cov_count[pos_i_old] += 1
                    cov_count[pos_j_old] += 1
                    
                    if sc < cur_score:
                        # Apply
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
