"""Odd covering systems with moduli dividing 51975; improved local search + SA."""
import time
import random
import numpy as np

BASE = 51975

# Precompute all odd divisors > 1 of 51975
_DIVS = []
for _d in range(3, BASE + 1, 2):
    if BASE % _d == 0:
        _DIVS.append(_d)
_DIVS.sort()
N_DIVS = len(_DIVS)

# Precompute position arrays for each divisor and each residue
# pos_cache[m][r] = np.array of positions r, r+m, r+2m, ... < BASE
_pos_cache = {}
for m in _DIVS:
    arr = []
    for r in range(m):
        arr.append(np.arange(r, BASE, m))
    _pos_cache[m] = arr

def _score_from_cov(cov_count):
    return int(np.sum(cov_count == 0))

def _greedy(order, rng=None):
    """Greedy construction: for each modulus in order, pick the residue covering most uncovered."""
    covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in order:
        best_r = 0
        best_c = -1
        for r in range(m):
            positions = _pos_cache[m][r]
            c = int(np.sum(covered[positions] == 0))
            if c > best_c:
                best_c = c
                best_r = r
            if best_c == len(positions):
                break
        classes.append([m, best_r])
        covered[_pos_cache[m][best_r]] = 1
    return classes

def _build_cov(classes):
    cov = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        cov[_pos_cache[m][r]] += 1
    return cov

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s kill
    rng = random.Random(42)

    # Try multiple greedy orderings and pick the best
    best_classes = None
    best_score = BASE  # worst case
    best_cov = None

    # Ordering 1: increasing modulus
    classes1 = _greedy(_DIVS, rng)
    cov1 = _build_cov(classes1)
    s1 = _score_from_cov(cov1)
    if s1 < best_score:
        best_score = s1
        best_classes = [list(c) for c in classes1]
        best_cov = cov1.copy()

    # Ordering 2: decreasing modulus
    classes2 = _greedy(list(reversed(_DIVS)), rng)
    cov2 = _build_cov(classes2)
    s2 = _score_from_cov(cov2)
    if s2 < best_score:
        best_score = s2
        best_classes = [list(c) for c in classes2]
        best_cov = cov2.copy()

    # Ordering 3: random
    for _ in range(3):
        order = list(_DIVS)
        rng.shuffle(order)
        classes3 = _greedy(order, rng)
        cov3 = _build_cov(classes3)
        s3 = _score_from_cov(cov3)
        if s3 < best_score:
            best_score = s3
            best_classes = [list(c) for c in classes3]
            best_cov = cov3.copy()

    cur_classes = [list(c) for c in best_classes]
    cur_score = best_score
    n = len(cur_classes)

    # Track uncovered positions for fast delta computation
    uncovered = np.zeros(BASE, dtype=np.int8)
    for i in range(BASE):
        if best_cov[i] == 0:
            uncovered[i] = 1

    # Local search with simulated annealing
    temp = 15.0
    cool_rate = 0.998
    min_temp = 0.05

    no_improve = 0
    max_no_improve = 80

    # Precompute for each class: its position array
    class_positions = [_pos_cache[m][r] for m, r in cur_classes]

    while time.time() < deadline:
        if cur_score == 0:
            return cur_classes

        improved = False
        idx_order = list(range(n))
        rng.shuffle(idx_order)

        # Phase 1: Single residue changes with fast delta
        for i in idx_order:
            if time.time() > deadline:
                break
            m_i, r_i = cur_classes[i]
            old_pos = class_positions[i]
            
            best_r = r_i
            best_sc = cur_score
            
            # Sample residues if m_i is large, otherwise try all
            if m_i > 300:
                candidates = [r_i] + rng.sample(range(m_i), min(80, m_i - 1))
            else:
                candidates = [r for r in range(m_i) if r != r_i]
            
            for r in candidates:
                if r == r_i:
                    continue
                new_pos = _pos_cache[m_i][r]
                
                # Fast delta: only check positions that change
                # Positions in old_pos but not in new_pos: lose coverage
                # Positions in new_pos but not in old_pos: gain coverage
                # Use set difference via boolean masks
                mask_old = np.zeros(BASE, dtype=np.bool_)
                mask_old[old_pos] = True
                mask_new = np.zeros(BASE, dtype=np.bool_)
                mask_new[new_pos] = True
                
                lost = mask_old & ~mask_new  # positions losing this class
                gained = mask_new & ~mask_old  # positions gaining this class
                
                # For lost positions: if cov was 1, they become uncovered
                # For gained positions: if cov was 0, they become covered
                lost_becomes_uncovered = 0
                if np.any(lost):
                    lost_becomes_uncovered = int(np.sum(best_cov[lost] == 1))
                gained_becomes_covered = 0
                if np.any(gained):
                    gained_becomes_covered = int(np.sum(best_cov[gained] == 0))
                
                sc = cur_score - gained_becomes_covered + lost_becomes_uncovered
                
                if sc < best_sc:
                    best_sc = sc
                    best_r = r
                elif sc == best_sc and rng.random() < 0.15:
                    best_r = r  # random tie-breaking to explore
            
            if best_r != r_i:
                new_pos = _pos_cache[m_i][best_r]
                # Apply the change
                mask_old = np.zeros(BASE, dtype=np.bool_)
                mask_old[old_pos] = True
                mask_new = np.zeros(BASE, dtype=np.bool_)
                mask_new[new_pos] = True
                
                lost = mask_old & ~mask_new
                gained = mask_new & ~mask_old
                
                if np.any(lost):
                    best_cov[lost] -= 1
                    for p in old_pos:
                        if best_cov[p] == 0:
                            uncovered[p] = 1
                if np.any(gained):
                    best_cov[gained] += 1
                    for p in new_pos:
                        if best_cov[p] > 0:
                            uncovered[p] = 0
                
                # Recompute score from uncovered for accuracy
                cur_score = int(np.sum(uncovered))
                
                cur_classes[i][1] = best_r
                class_positions[i] = new_pos
                improved = True
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_classes = [list(c) for c in cur_classes]
                    best_cov = best_cov.copy()

        # Phase 2: Modulus swap - replace one class's modulus with an unused one
        if time.time() < deadline and cur_score > 0:
            used_mods = set(m for m, r in cur_classes)
            available_mods = [m for m in _DIVS if m not in used_mods]
            
            if available_mods:
                # Try replacing a few random classes with available moduli
                for _ in range(min(10, len(available_mods))):
                    if time.time() > deadline:
                        break
                    i = rng.randint(0, n - 1)
                    old_m, old_r = cur_classes[i]
                    new_m = rng.choice(available_mods)
                    
                    # Find best residue for new_m
                    old_pos = class_positions[i]
                    best_r_new = 0
                    best_sc_new = cur_score
                    best_gained = -1
                    
                    for r in range(new_m):
                        new_pos = _pos_cache[new_m][r]
                        mask_old = np.zeros(BASE, dtype=np.bool_)
                        mask_old[old_pos] = True
                        mask_new = np.zeros(BASE, dtype=np.bool_)
                        mask_new[new_pos] = True
                        
                        lost = mask_old & ~mask_new
                        gained = mask_new & ~mask_old
                        
                        lost_becomes_uncovered = int(np.sum(best_cov[lost] == 1)) if np.any(lost) else 0
                        gained_becomes_covered = int(np.sum(best_cov[gained] == 0)) if np.any(gained) else 0
                        
                        sc = cur_score - gained_becomes_covered + lost_becomes_uncovered
                        
                        if sc < best_sc_new:
                            best_sc_new = sc
                            best_r_new = r
                            best_gained = gained_becomes_covered - lost_becomes_uncovered
                    
                    if best_sc_new < cur_score:
                        # Apply
                        new_pos = _pos_cache[new_m][best_r_new]
                        mask_old = np.zeros(BASE, dtype=np.bool_)
                        mask_old[old_pos] = True
                        mask_new = np.zeros(BASE, dtype=np.bool_)
                        mask_new[new_pos] = True
                        
                        lost = mask_old & ~mask_new
                        gained = mask_new & ~mask_old
                        
                        if np.any(lost):
                            best_cov[lost] -= 1
                            for p in old_pos:
                                if best_cov[p] == 0:
                                    uncovered[p] = 1
                        if np.any(gained):
                            best_cov[gained] += 1
                            for p in new_pos:
                                if best_cov[p] > 0:
                                    uncovered[p] = 0
                        
                        cur_score = int(np.sum(uncovered))
                        cur_classes[i] = [new_m, best_r_new]
                        class_positions[i] = new_pos
                        improved = True
                        no_improve = 0
                        if cur_score < best_score:
                            best_score = cur_score
                            best_classes = [list(c) for c in cur_classes]
                            best_cov = best_cov.copy()

        # Phase 3: Pairwise residue changes
        if time.time() < deadline and cur_score > 0:
            pairs_tried = 0
            max_pairs = 150
            for i in idx_order:
                if time.time() > deadline or pairs_tried >= max_pairs:
                    break
                for j in idx_order:
                    if j <= i or pairs_tried >= max_pairs:
                        continue
                    pairs_tried += 1
                    m_i, r_i = cur_classes[i]
                    m_j, r_j = cur_classes[j]
                    
                    new_r_i = rng.randint(0, m_i - 1)
                    new_r_j = rng.randint(0, m_j - 1)
                    
                    if new_r_i == r_i and new_r_j == r_j:
                        continue
                    
                    pos_i_old = class_positions[i]
                    pos_j_old = class_positions[j]
                    pos_i_new = _pos_cache[m_i][new_r_i]
                    pos_j_new = _pos_cache[m_j][new_r_j]
                    
                    # Combined delta
                    mask_i_old = np.zeros(BASE, dtype=np.bool_)
                    mask_i_old[pos_i_old] = True
                    mask_j_old = np.zeros(BASE, dtype=np.bool_)
                    mask_j_old[pos_j_old] = True
                    mask_i_new = np.zeros(BASE, dtype=np.bool_)
                    mask_i_new[pos_i_new] = True
                    mask_j_new = np.zeros(BASE, dtype=np.bool_)
                    mask_j_new[pos_j_new] = True
                    
                    old_mask = mask_i_old | mask_j_old
                    new_mask = mask_i_new | mask_j_new
                    
                    lost = old_mask & ~new_mask
                    gained = new_mask & ~old_mask
                    
                    lost_becomes_uncovered = int(np.sum(best_cov[lost] == 1)) if np.any(lost) else 0
                    gained_becomes_covered = int(np.sum(best_cov[gained] == 0)) if np.any(gained) else 0
                    
                    sc = cur_score - gained_becomes_covered + lost_becomes_uncovered
                    
                    if sc < cur_score:
                        # Apply
                        if np.any(lost):
                            best_cov[lost] -= 1
                            for p in pos_i_old:
                                if best_cov[p] == 0:
                                    uncovered[p] = 1
                            for p in pos_j_old:
                                if best_cov[p] == 0:
                                    uncovered[p] = 1
                        if np.any(gained):
                            best_cov[gained] += 1
                            for p in pos_i_new:
                                if best_cov[p] > 0:
                                    uncovered[p] = 0
                            for p in pos_j_new:
                                if best_cov[p] > 0:
                                    uncovered[p] = 0
                        
                        cur_score = int(np.sum(uncovered))
                        cur_classes[i][1] = new_r_i
                        cur_classes[j][1] = new_r_j
                        class_positions[i] = pos_i_new
                        class_positions[j] = pos_j_new
                        improved = True
                        no_improve = 0
                        if cur_score < best_score:
                            best_score = cur_score
                            best_classes = [list(c) for c in cur_classes]
                            best_cov = best_cov.copy()

        # Phase 4: Simulated annealing - accept worse moves
        if time.time() < deadline and cur_score > 0 and temp > min_temp:
            for i in idx_order:
                if time.time() > deadline:
                    break
                m_i, r_i = cur_classes[i]
                new_r = rng.randint(0, m_i - 1)
                if new_r == r_i:
                    continue
                
                old_pos = class_positions[i]
                new_pos = _pos_cache[m_i][new_r]
                
                mask_old = np.zeros(BASE, dtype=np.bool_)
                mask_old[old_pos] = True
                mask_new = np.zeros(BASE, dtype=np.bool_)
                mask_new[new_pos] = True
                
                lost = mask_old & ~mask_new
                gained = mask_new & ~mask_old
                
                lost_becomes_uncovered = int(np.sum(best_cov[lost] == 1)) if np.any(lost) else 0
                gained_becomes_covered = int(np.sum(best_cov[gained] == 0)) if np.any(gained) else 0
                
                sc = cur_score - gained_becomes_covered + lost_becomes_uncovered
                
                delta = sc - cur_score
                if delta < 0 or (delta > 0 and rng.random() < np.exp(-delta / temp)):
                    # Accept
                    if np.any(lost):
                        best_cov[lost] -= 1
                        for p in old_pos:
                            if best_cov[p] == 0:
                                uncovered[p] = 1
                    if np.any(gained):
                        best_cov[gained] += 1
                        for p in new_pos:
                            if best_cov[p] > 0:
                                uncovered[p] = 0
                    
                    cur_score = int(np.sum(uncovered))
                    cur_classes[i][1] = new_r
                    class_positions[i] = new_pos
                    improved = True
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_classes = [list(c) for c in cur_classes]
                        best_cov = best_cov.copy()
                
                temp *= cool_rate

        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                # Restart from best
                cur_classes = [list(c) for c in best_classes]
                cur_score = best_score
                best_cov = _build_cov(cur_classes)
                uncovered = (best_cov == 0).astype(np.int8)
                class_positions = [_pos_cache[m][r] for m, r in cur_classes]
                temp = max(temp, 8.0)
                no_improve = 0
                if time.time() > deadline - 10:
                    break

    return best_classes
