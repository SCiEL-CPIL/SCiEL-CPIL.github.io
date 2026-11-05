import numpy as np
import time
import random
from math import gcd, log, exp

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
    deadline = start_time + 138  # Safety margin
    
    divs = _divisors_of(BASE)
    n_classes = len(divs)
    
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

    def adaptive_greedy_init(seed=None):
        """
        Adaptive look-ahead greedy: at each step, probabilistically select the next
        modulus based on its marginal gain relative to remaining budget.
        Uses incremental filtering to quickly eliminate bad candidates.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        # Start with empty coverage
        covered = np.zeros(BASE, dtype=np.uint8)
        classes = []
        remaining_indices = list(range(n_classes))
        
        # Precompute divisor values for quick access
        divs_arr = np.array(divs)
        
        while remaining_indices:
            if time.time() >= deadline:
                # Time up - just pick remaining randomly
                for idx in remaining_indices:
                    m = divs[idx]
                    r = rng.randint(0, m - 1)
                    classes.append([m, r])
                    covered[r::m] = 1
                break
            
            # Compute marginal gain for each remaining modulus
            # Use adaptive sampling: fewer samples for large moduli, more for small
            gains = np.zeros(len(remaining_indices))
            
            # Estimate time per modulus based on size
            # Small moduli: cheap to evaluate fully
            # Large moduli: sample fewer residues
            
            for i, idx in enumerate(remaining_indices):
                m = divs[idx]
                
                # Adaptive candidate count based on modulus size and remaining time
                elapsed = time.time() - start_time
                remaining_time = deadline - elapsed
                # Allocate more time to smaller moduli (cheaper per candidate)
                if m <= 50:
                    n_candidates = m  # Evaluate all
                elif m <= 200:
                    n_candidates = min(m, 100)
                elif m <= 1000:
                    n_candidates = min(m, 50)
                else:
                    n_candidates = min(m, 25)
                
                # Sample candidates
                if n_candidates < m:
                    candidates = rng.sample(range(m), n_candidates)
                else:
                    candidates = list(range(m))
                
                # Incremental filtering: compute gain for each candidate
                best_gain = 0
                for r in candidates:
                    gain = np.sum(covered[r::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                
                gains[i] = best_gain
            
            # Adaptive selection: use softmax with temperature that decreases over time
            # Early: more exploration (higher temp), Late: more exploitation (lower temp)
            progress = 1.0 - len(remaining_indices) / n_classes
            # Temperature: starts high (exploration), ends low (exploitation)
            temperature = max(0.1, 5.0 * (1.0 - progress))
            
            # Compute selection probabilities using log-sum-exp for numerical stability
            max_gain = np.max(gains)
            if max_gain == 0:
                # All gains are zero, pick uniformly
                weights = np.ones(len(remaining_indices))
            else:
                # Softmax with temperature
                log_weights = (gains - max_gain) / temperature
                log_max = np.max(log_weights)
                log_weights -= log_max
                weights = np.exp(log_weights)
                weights /= np.sum(weights)
            
            # Select next modulus
            selected_pos = rng.choices(range(len(remaining_indices)), weights=weights, k=1)[0]
            selected_idx = remaining_indices[selected_pos]
            remaining_indices.remove(selected_idx)
            
            m = divs[selected_idx]
            
            # Find best residue for this modulus
            if m <= 100:
                candidates = range(m)
            else:
                candidates = rng.sample(range(m), min(m, 50))
            
            best_r = 0
            best_count = -1
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
    
    # 1. Try multiple adaptive greedy initializations with different seeds
    n_initializations = 8
    for seed in range(n_initializations):
        if time.time() >= deadline:
            break
        classes = adaptive_greedy_init(seed=seed)
        score = _uncovered_count(classes)
        if score < best_score:
            best_score = score
            best_classes = classes[:]
        
        # Also try a deterministic ordering for diversity
        if seed == 0:
            # Descending order
            covered = np.zeros(BASE, dtype=np.uint8)
            classes_desc = []
            sorted_mods = sorted(range(n_classes), key=lambda i: divs[i], reverse=True)
            for i in sorted_mods:
                m = divs[i]
                best_r = 0
                best_count = -1
                if m <= 200:
                    candidates = range(m)
                else:
                    candidates = random.sample(range(m), min(m, 200))
                for r in candidates:
                    cnt = np.sum(covered[r::m] == 0)
                    if cnt > best_count:
                        best_count = cnt
                        best_r = r
                classes_desc.append([m, best_r])
                covered[best_r::m] = 1
            score_desc = _uncovered_count(classes_desc)
            if score_desc < best_score:
                best_score = score_desc
                best_classes = classes_desc[:]
        
        if seed == 1:
            # Ascending order
            covered = np.zeros(BASE, dtype=np.uint8)
            classes_asc = []
            sorted_mods = sorted(range(n_classes), key=lambda i: divs[i])
            for i in sorted_mods:
                m = divs[i]
                best_r = 0
                best_count = -1
                if m <= 200:
                    candidates = range(m)
                else:
                    candidates = random.sample(range(m), min(m, 200))
                for r in candidates:
                    cnt = np.sum(covered[r::m] == 0)
                    if cnt > best_count:
                        best_count = cnt
                        best_r = r
                classes_asc.append([m, best_r])
                covered[best_r::m] = 1
            score_asc = _uncovered_count(classes_asc)
            if score_asc < best_score:
                best_score = score_asc
                best_classes = classes_asc[:]

    if best_classes is None:
        best_classes = adaptive_greedy_init(seed=42)
        best_score = _uncovered_count(best_classes)

    # 2. Local Search with Simulated Annealing
    current_classes = best_classes[:]
    current_refcount = _compute_refcounts(current_classes)
    current_zero_mask = (current_refcount == 0)
    current_uncovered = int(np.sum(current_zero_mask))
    
    # SA Parameters - tuned for more iterations
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.96
    steps_per_temp = 200
    
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
            # Use adaptive sampling based on modulus size
            if m <= 30:
                new_r = random.randint(0, m - 1)
            else:
                # Sample a few candidates and pick the one with most uncovered
                n_samples = min(m, 8)
                samples = random.sample(range(m), n_samples)
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
            if delta < 0 or random.random() < exp(-delta / T):
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

    # 3. Final Polish: Deterministic 1-opt local search with incremental filtering
    improved = True
    rounds = 0
    while improved and time.time() < deadline and rounds < 50:
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
            # Use incremental filtering: skip candidates that can't improve
            if m <= 80:
                candidates = range(m)
            else:
                n_samples = min(m, 150)
                candidates = random.sample(range(m), n_samples)
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

    return best_classes
