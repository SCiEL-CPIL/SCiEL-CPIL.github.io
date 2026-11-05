import numpy as np
import time
import random
from math import gcd

BASE = 51975

def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted([d for d in divs if d > 1])

def _uncovered_count(classes, base=BASE):
    """Compute number of uncovered residues."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(covered == 0))

def _compute_refcounts(classes):
    """Compute reference count for each residue."""
    refcount = np.zeros(BASE, dtype=np.int16)
    for m, r in classes:
        refcount[r::m] += 1
    return refcount

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    divs = _divisors_of(BASE)
    n_classes = len(divs)
    
    def _compute_newly_covered(refcount):
        """Precompute count of refcount==0 residues in each residue class for each modulus."""
        # For each modulus divs[i], for each residue r in [0, m), count how many
        # positions x ≡ r (mod m) have refcount[x] == 0
        # Store as a 2D array: newly_covered[i, r] = count
        # But this is too large. Instead, compute on the fly but cache per modulus.
        # Actually, we can just compute it when needed.
        # Better approach: precompute a boolean array zero_mask = (refcount == 0)
        # then gained = np.sum(zero_mask[new_r::m])
        pass
    
    def calc_delta_fast(zero_mask, refcount, idx, old_r, new_r):
        m = divs[idx]
        if old_r == new_r:
            return 0
        # Residues in old slice with refcount 1 become 0 (uncovered)
        lost = np.sum(refcount[old_r::m] == 1)
        # Residues in new slice with refcount 0 become 1 (covered)
        gained = np.sum(zero_mask[new_r::m])
        return lost - gained

    def apply_move(refcount, zero_mask, idx, old_r, new_r):
        m = divs[idx]
        refcount[old_r::m] -= 1
        refcount[new_r::m] += 1
        # Update zero_mask
        zero_mask[old_r::m] = (refcount[old_r::m] == 0)
        zero_mask[new_r::m] = (refcount[new_r::m] == 0)

    def greedy_init(order='desc'):
        covered = np.zeros(BASE, dtype=np.uint8)
        classes = []
        if order == 'desc':
            sorted_mods = sorted(range(n_classes), key=lambda i: divs[i], reverse=True)
        elif order == 'asc':
            sorted_mods = sorted(range(n_classes), key=lambda i: divs[i])
        else:
            sorted_mods = list(range(n_classes))
            random.shuffle(sorted_mods)
        
        for i in sorted_mods:
            m = divs[i]
            best_r = 0
            best_count = -1
            # For small m, iterate all r. For large m, sample.
            if m <= 200:
                candidates = range(m)
            else:
                candidates = random.sample(range(m), min(m, 200))
            
            for r in candidates:
                cnt = np.sum(covered[r::m] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes

    best_classes = None
    best_score = BASE + 1
    
    # 1. Try multiple greedy initializations
    for order in ['desc', 'asc', 'rand', 'rand', 'rand']:
        if time.time() >= deadline:
            break
        classes = greedy_init(order=order)
        score = _uncovered_count(classes)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
            
    if best_classes is None:
        best_classes = greedy_init('desc')
        best_score = _uncovered_count(best_classes)

    # 2. Local Search with Simulated Annealing
    current_classes = best_classes[:]
    current_refcount = _compute_refcounts(current_classes)
    current_zero_mask = (current_refcount == 0)
    current_uncovered = int(np.sum(current_zero_mask))
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.95
    steps_per_temp = 300
    
    T = T_start
    while T > T_end and time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() >= deadline:
                break
            
            # Pick a random modulus index
            idx = random.randint(0, n_classes - 1)
            m = current_classes[idx][0]
            old_r = current_classes[idx][1]
            
            # Pick a new residue with bias toward uncovered regions
            # Sample a few candidates and pick the one with most uncovered
            if m <= 50:
                new_r = random.randint(0, m - 1)
            else:
                # Sample a few candidates
                samples = random.sample(range(m), min(m, 5))
                best_sample_r = old_r
                best_sample_gain = -1
                for sr in samples:
                    gain = np.sum(current_zero_mask[sr::m])
                    if gain > best_sample_gain:
                        best_sample_gain = gain
                        best_sample_r = sr
                new_r = best_sample_r
            
            if old_r == new_r:
                continue
                
            delta = calc_delta_fast(current_zero_mask, current_refcount, idx, old_r, new_r)
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                apply_move(current_refcount, current_zero_mask, idx, old_r, new_r)
                current_classes[idx][1] = new_r
                current_uncovered += delta
                
                if current_uncovered < best_score:
                    best_score = current_uncovered
                    best_classes = current_classes[:]
                    # If we reach 0, we can stop
                    if best_score == 0:
                        return best_classes
        T *= cooling_rate

    # 3. Final Polish: Deterministic 1-opt local search
    # Run a few rounds of best-improvement 1-opt
    improved = True
    rounds = 0
    while improved and time.time() < deadline and rounds < 30:
        improved = False
        rounds += 1
        current_uncovered = int(np.sum(current_zero_mask))
        
        # Shuffle indices for diversification
        indices = list(range(n_classes))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() >= deadline:
                break
            m, r = current_classes[idx]
            
            best_r = r
            best_uncovered = current_uncovered
            
            # Check all residues for small m, sample for large m
            if m <= 100:
                candidates = range(m)
            else:
                candidates = random.sample(range(m), min(m, 200))
                if r not in candidates:
                    candidates = list(candidates) + [r]

            for r_new in candidates:
                if r_new == r:
                    continue
                delta = calc_delta_fast(current_zero_mask, current_refcount, idx, r, r_new)
                if delta < 0:
                    new_uncovered = current_uncovered + delta
                    if new_uncovered < best_uncovered:
                        best_uncovered = new_uncovered
                        best_r = r_new

            if best_r != r:
                apply_move(current_refcount, current_zero_mask, idx, r, best_r)
                current_classes[idx][1] = best_r
                current_uncovered = best_uncovered
                improved = True
                if current_uncovered < best_score:
                    best_score = current_uncovered
                    best_classes = current_classes[:]

    # 4. Try 2-opt moves: swap residues between two moduli
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() >= deadline:
                break
            improved_2 = False
            # Try random pairs of moduli
            for _ in range(500):
                if time.time() >= deadline:
                    break
                i1 = random.randint(0, n_classes - 1)
                i2 = random.randint(0, n_classes - 1)
                if i1 == i2:
                    continue
                m1, r1 = current_classes[i1]
                m2, r2 = current_classes[i2]
                
                # Try swapping: i1 gets r2, i2 gets r1
                # But r2 might not be < m1, so this only works if we're swapping moduli too
                # Actually, we can't swap residues between different moduli directly.
                # Instead, try: set i1 to some r, and i2 to some r' simultaneously
                # This is complex. Let's just do a simple 2-opt: pick two indices and
                # try to improve both simultaneously by trying a few residue pairs.
                pass
            # Skip 2-opt for now, focus on 1-opt polish
            break

    return best_classes
