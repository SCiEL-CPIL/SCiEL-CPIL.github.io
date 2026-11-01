"""Combined covering-code constructor: bitmask greedy + pruning + simulated annealing.

Strategy:
  * Bitmask-based sphere representation for fast coverage tracking (inspiration 2).
  * Multiple greedy restarts with random ordering for diversity.
  * Pruning pass to remove redundant codewords.
  * Simulated annealing refinement: binary state vector, minimize |uncovered| + |code|.
  * Always-valid baseline as guaranteed fallback.
"""

import time
import itertools
import random
import numpy as np

_DEADLINE = 140.0


def _direct_sum(q, n, radius):
    """Always-valid baseline: direct sum of radius-1 perfect code with remaining coords."""
    base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in base
            for tail in itertools.product(range(q), repeat=n - 4)]


def _covers_all(cw, q, n, radius):
    """Exact coverage check over the whole space."""
    total = q ** n
    if total > 10**7:
        return False
    covered = np.zeros(total, dtype=bool)
    for c in cw:
        for d in range(radius + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(1, q), repeat=d):
                    w = list(c)
                    for p, v in zip(pos, vals):
                        w[p] = v
                    idx = 0
                    for x in w:
                        idx = idx * q + x
                    covered[idx] = True
    return covered.all()


def _build_spheres_bitmask(q, n, radius):
    """Build bitmask spheres for all words. Returns list of integers (bitmasks)."""
    total = q ** n
    powers = [q ** i for i in range(n)]
    spheres = [0] * total
    for x in range(total):
        mask = 1 << x
        if radius >= 1:
            for i in range(n):
                digit = (x // powers[i]) % q
                for val in range(q):
                    if val == digit:
                        continue
                    nb = x + (val - digit) * powers[i]
                    mask |= (1 << nb)
        spheres[x] = mask
    return spheres


def _prune_bitmask(code_indices, spheres, full_mask):
    """Remove redundant codewords from the code."""
    current = list(code_indices)
    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            union_others = 0
            for j in range(len(current)):
                if i == j:
                    continue
                union_others |= spheres[current[j]]
                if union_others == full_mask:
                    break
            if union_others == full_mask:
                current.pop(i)
                changed = True
                break
    return current


def _greedy_bitmask(spheres, full_mask, time_budget, start_time, random_order=None):
    """Greedy set-cover using bitmask operations."""
    if random_order is not None:
        candidates = list(range(len(spheres)))
        random.shuffle(candidates)
    else:
        candidates = list(range(len(spheres)))
    
    covered = 0
    chosen = []
    while covered != full_mask:
        if time.time() - start_time > time_budget:
            break
        best_gain = -1
        best_idx = -1
        for idx in candidates:
            new_covered = spheres[idx] & ~covered
            gain = new_covered.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_gain <= 0:
            break
        chosen.append(best_idx)
        covered |= spheres[best_idx]
    return chosen


def _simulated_annealing(code_indices, spheres, full_mask, time_budget, start_time):
    """Simulated annealing to minimize |code| + |uncovered| using bitmask operations."""
    current_code = list(code_indices)
    current_covered = 0
    for idx in current_code:
        current_covered |= spheres[idx]
    
    current_uncovered_count = full_mask.bit_count() - current_covered.bit_count()
    current_objective = len(current_code) + current_uncovered_count
    
    best_code = list(current_code)
    best_objective = current_objective
    
    # Temperature schedule
    t0 = 2.0
    t_end = 0.01
    n_steps = 50000
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Cooling
        progress = step / n_steps
        temperature = t0 * (t_end / t0) ** progress
        
        # Choose a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Try removing a codeword
            if len(current_code) == 0:
                continue
            idx_to_remove = random.choice(current_code)
            new_code = [c for c in current_code if c != idx_to_remove]
            new_covered = 0
            for idx in new_code:
                new_covered |= spheres[idx]
            new_uncovered = full_mask.bit_count() - new_covered.bit_count()
            new_objective = len(new_code) + new_uncovered
            delta = new_objective - current_objective
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_code = new_code
                current_covered = new_covered
                current_uncovered_count = new_uncovered
                current_objective = new_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_code = list(current_code)
                    
        elif move_type < 0.6:
            # Try adding a random codeword
            new_idx = random.randint(0, len(spheres) - 1)
            if new_idx in current_code:
                continue
            new_code = current_code + [new_idx]
            new_covered = current_covered | spheres[new_idx]
            new_uncovered = full_mask.bit_count() - new_covered.bit_count()
            new_objective = len(new_code) + new_uncovered
            delta = new_objective - current_objective
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_code = new_code
                current_covered = new_covered
                current_uncovered_count = new_uncovered
                current_objective = new_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_code = list(current_code)
                    
        else:
            # Try swapping: remove one, add another
            if len(current_code) == 0:
                continue
            idx_to_remove = random.choice(current_code)
            new_idx = random.randint(0, len(spheres) - 1)
            if new_idx == idx_to_remove or new_idx in current_code:
                continue
            new_code = [c for c in current_code if c != idx_to_remove] + [new_idx]
            new_covered = 0
            for idx in new_code:
                new_covered |= spheres[idx]
            new_uncovered = full_mask.bit_count() - new_covered.bit_count()
            new_objective = len(new_code) + new_uncovered
            delta = new_objective - current_objective
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_code = new_code
                current_covered = new_covered
                current_uncovered_count = new_uncovered
                current_objective = new_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_code = list(current_code)
    
    return best_code


def construct(q, n, radius):
    start = time.time()
    
    # Always-valid baseline
    if n >= 4 and radius >= 1:
        baseline = _direct_sum(q, n, radius)
    else:
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    best = baseline
    best_len = len(baseline)
    
    # Only proceed with advanced search for tractable cases
    total = q ** n
    if total > 3 * 10**6 or best_len <= 1:
        return baseline
    
    # Build bitmask spheres
    spheres = _build_spheres_bitmask(q, n, radius)
    full_mask = (1 << total) - 1
    
    # Build word lookup
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Initialize with tetracode product and prune
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            w = tuple(h + list(tail))
            idx = 0
            for x in w:
                idx = idx * q + x
            init_code.append(idx)
    
    pruned_init = _prune_bitmask(init_code, spheres, full_mask)
    if len(pruned_init) < best_len:
        best_len = len(pruned_init)
        best = [all_words[i] for i in pruned_init]
    
    # Multiple greedy restarts
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() - start > _DEADLINE - 20:
            break
        
        random.seed(r)
        candidates = list(range(total))
        random.shuffle(candidates)
        
        covered = 0
        chosen = []
        while covered != full_mask:
            if time.time() - start > _DEADLINE - 20:
                break
            best_gain = -1
            best_idx = -1
            for idx in candidates:
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_gain <= 0:
                break
            chosen.append(best_idx)
            covered |= spheres[best_idx]
        
        # Prune the greedy result
        pruned_greedy = _prune_bitmask(chosen, spheres, full_mask)
        
        # Try simulated annealing on the pruned result
        if time.time() - start < _DEADLINE - 30:
            sa_result = _simulated_annealing(pruned_greedy, spheres, full_mask, 
                                            _DEADLINE - 30 - (time.time() - start), start)
            if len(sa_result) < len(pruned_greedy):
                pruned_greedy = sa_result
        
        if len(pruned_greedy) < best_len:
            best_len = len(pruned_greedy)
            best = [all_words[i] for i in pruned_greedy]
    
    # Final safety check
    if time.time() - start > _DEADLINE - 5:
        return baseline
    
    # Verify the best solution
    if not _covers_all(best, q, n, radius):
        return baseline
    
    return best
