"""Odd covering systems with moduli dividing 51975 – local-search solver."""
import time
import random
import numpy as np

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# Generate all divisors > 1
def _get_divisors():
    divs = []
    n = BASE
    # factorize
    factors = []
    tmp = n
    p = 3
    while p * p <= tmp:
        if tmp % p == 0:
            cnt = 0
            while tmp % p == 0:
                tmp //= p
                cnt += 1
            factors.append((p, cnt))
        p += 2
    if tmp > 1:
        factors.append((tmp, 1))
    # generate divisors
    divs = [1]
    for p, e in factors:
        new_divs = []
        for d in divs:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    divs = sorted([d for d in divs if d > 1])
    return divs

MODULI = _get_divisors()
N_MOD = len(MODULI)

def _build_masks():
    """Precompute for each modulus m and residue r, a boolean mask of covered positions."""
    masks = []
    for m in MODULI:
        r_masks = []
        for r in range(m):
            mask = np.zeros(BASE, dtype=bool)
            mask[r::m] = True
            r_masks.append(mask)
        masks.append(r_masks)
    return masks

def _score(assignment, masks):
    """Count uncovered residues given assignment (list of residues, one per modulus)."""
    covered = np.zeros(BASE, dtype=bool)
    for i, r in enumerate(assignment):
        covered |= masks[i][r]
    return int(np.sum(~covered))

def _greedy_asc(masks):
    """Greedy ascending moduli."""
    covered = np.zeros(BASE, dtype=bool)
    assignment = []
    for i, m in enumerate(MODULI):
        best_r = 0
        best_cov = -1
        for r in range(m):
            new_cov = int(np.sum(masks[i][r] & ~covered))
            if new_cov > best_cov:
                best_cov = new_cov
                best_r = r
        assignment.append(best_r)
        covered |= masks[i][best_r]
    return assignment

def _greedy_desc(masks):
    """Greedy descending moduli."""
    covered = np.zeros(BASE, dtype=bool)
    assignment = [0] * N_MOD
    for i in range(N_MOD - 1, -1, -1):
        m = MODULI[i]
        best_r = 0
        best_cov = -1
        for r in range(m):
            new_cov = int(np.sum(masks[i][r] & ~covered))
            if new_cov > best_cov:
                best_cov = new_cov
                best_r = r
        assignment[i] = best_r
        covered |= masks[i][best_r]
    return assignment

def _local_search(assignment, masks, deadline):
    """Iterative improvement: try changing one residue at a time if it reduces uncovered."""
    current = list(assignment)
    current_score = _score(current, masks)
    covered = np.zeros(BASE, dtype=bool)
    for i, r in enumerate(current):
        covered |= masks[i][r]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N_MOD):
            m = MODULI[i]
            old_r = current[i]
            # Remove old mask
            old_mask = masks[i][old_r]
            # For each candidate new residue, compute delta
            best_r = old_r
            best_delta = 0
            for r in range(m):
                if r == old_r:
                    continue
                new_mask = masks[i][r]
                # Positions newly covered: new_mask & ~covered
                newly = int(np.sum(new_mask & ~covered))
                # Positions lost: old_mask & ~covered (but only those not covered by others)
                # Actually: positions that were covered ONLY by old_r
                # = old_mask & ~covered_after_removing_old
                # Simpler: delta = newly_covered - lost_covered
                # lost = positions in old_mask that are NOT in new_mask and NOT covered by other classes
                # This is expensive. Use approximate: 
                # covered_without_old = covered & ~old_mask  (not quite, since other classes may cover)
                # Let's just compute score delta properly:
                # new_covered = covered & ~old_mask | new_mask
                # delta = sum(new_covered) - sum(covered)
                # But we need to be careful: covered includes old_mask contributions.
                # Let's compute: positions currently covered = covered
                # If we swap old_r -> r: 
                #   remove old_mask, add new_mask
                #   new_covered = (covered & ~old_mask) | new_mask
                #   delta = popcount(new_covered) - popcount(covered)
                new_covered = (covered & ~old_mask) | new_mask
                delta = int(np.sum(new_covered)) - int(np.sum(covered))
                if delta > best_delta:
                    best_delta = delta
                    best_r = r
            if best_r != old_r and best_delta > 0:
                # Apply swap
                covered = (covered & ~masks[i][old_r]) | masks[i][best_r]
                current[i] = best_r
                current_score -= best_delta
                improved = True
    return current, current_score

def _perturb(assignment, masks, n_changes=2):
    """Randomly change n residues."""
    perturbed = list(assignment)
    for _ in range(n_changes):
        i = random.randint(0, N_MOD - 1)
        m = MODULI[i]
        new_r = random.randint(0, m - 1)
        perturbed[i] = new_r
    return perturbed

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    masks = _build_masks()
    
    # Multiple initializations
    candidates = []
    
    # Greedy ascending
    if time.time() < deadline:
        a = _greedy_asc(masks)
        s = _score(a, masks)
        candidates.append((s, a))
    
    # Greedy descending
    if time.time() < deadline:
        a = _greedy_desc(masks)
        s = _score(a, masks)
        candidates.append((s, a))
    
    # Random initializations
    n_random = 5
    for _ in range(n_random):
        if time.time() >= deadline:
            break
        a = [random.randint(0, MODULI[i] - 1) for i in range(N_MOD)]
        s = _score(a, masks)
        candidates.append((s, a))
    
    # Select best initial
    candidates.sort(key=lambda x: x[0])
    best_score, best_assign = candidates[0]
    
    # Iterated local search
    ls_deadline = deadline - 5  # leave 5s for final polish
    for it in range(1000000):
        if time.time() >= ls_deadline:
            break
        
        # Local search from current best
        improved_assign, improved_score = _local_search(best_assign, masks, ls_deadline)
        if improved_score < best_score:
            best_score = improved_score
            best_assign = improved_assign
        
        if best_score == 0:
            break
        
        # Perturb
        n_changes = random.choice([1, 2, 2, 3])
        perturbed = _perturb(best_assign, masks, n_changes)
        
        # Local search from perturbed
        p_assign, p_score = _local_search(perturbed, masks, ls_deadline)
        if p_score < best_score:
            best_score = p_score
            best_assign = p_assign
    
    # Final polish: one more local search pass
    if time.time() < deadline - 1:
        final_assign, final_score = _local_search(best_assign, masks, deadline - 1)
        if final_score < best_score:
            best_score = final_score
            best_assign = final_assign
    
    # Build result
    result = []
    for i, m in enumerate(MODULI):
        result.append([m, best_assign[i]])
    
    return result
