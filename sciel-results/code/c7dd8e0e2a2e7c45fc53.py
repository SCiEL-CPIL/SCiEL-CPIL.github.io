<<<<<<< SEARCH
"""Combined program: fast vectorized greedy + local improvement for odd covering systems."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    base = 51975  # 3^3 * 5^2 * 7 * 11, all odd

    # Get all odd divisors > 1 of base, sorted ascending (smallest covers most residues)
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()

    # Greedy construction: for each m in increasing order, pick r maximizing newly covered
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    used_mods = set()

    for m in divisors:
        if time.time() > deadline - 20.0:
            break
        # For each residue r in [0, m), count how many positions r, r+m, r+2m, ... are uncovered
        # Vectorized: covered[r::m] gives the slice; count zeros
        best_r = 0
        best_gain = -1
        # Compute gains for all r at once using reshape when possible
        # For speed, loop over r but use vectorized slice
        for r in range(m):
            gain = int(covered[r::m].sum())  # sum of 0/1 = covered count; we want uncovered
            # Actually we want uncovered count = len - covered_count
            # len of slice r::m is (base - r) // m + 1 if r < base else 0
            # Since m divides base and r < m, len = base // m exactly
            uncovered_gain = (base // m) - gain
            if uncovered_gain > best_gain:
                best_gain = uncovered_gain
                best_r = r

        if best_gain > 0:
            classes.append([m, best_r])
            covered[best_r::m] = 1
            used_mods.add(m)
        # If best_gain == 0, this modulus adds nothing; skip it

    current_uncovered = int(covered.sum())
    best_uncovered = current_uncovered
    best_classes = [c[:] for c in classes]

    # Local improvement: try changing residues of existing classes
    # For each class i, temporarily consider alternative residues
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline - 10.0:
                break
            m, r = classes[i]
            # For large m, sampling is faster; for small m, try all
            if m <= 50:
                candidates = list(range(m))
            else:
                # Sample a subset of residues for large moduli
                candidates = list(range(0, m, max(1, m // 50)))

            best_r_new = r
            best_gain_new = -1
            # Remove current class to see what it uniquely covers
            # Save current covered state
            orig_covered = covered.copy()
            # Remove class i
            covered[r::m] = 0
            # For each candidate r', compute how many new positions it covers
            for r_new in candidates:
                if r_new == r:
                    continue
                # Gain from adding (m, r_new) on top of covered without class i
                gain = int((covered[r_new::m] == 0).sum())
                if gain > best_gain_new:
                    best_gain_new = gain
                    best_r_new = r_new

            # Restore
            covered = orig_covered
            if best_r_new != r and best_gain_new > 0:
                # Check if this actually improves total coverage
                # Compute new uncovered after changing class i to (m, best_r_new)
                new_covered = covered.copy()
                new_covered[r::m] = 0
                new_covered[best_r_new::m] = 1
                new_uncovered = int(new_covered.sum())
                if new_uncovered < current_uncovered:
                    classes[i] = [m, best_r_new]
                    covered = new_covered
                    current_uncovered = new_uncovered
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = [c[:] for c in classes]
                    improved = True

    # Final pass: try adding unused small moduli that might fill gaps
    if time.time() < deadline - 5.0:
        for m in divisors:
            if time.time() > deadline - 5.0:
                break
            if m in used_mods:
                continue
            # Only consider if it could add coverage
            if base // m < 1:
                continue
            best_r = 0
            best_gain = -1
            for r in range(min(m, 50)):  # Sample for large m
                gain = int((covered[r::m] == 0).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                classes.append([m, best_r])
                covered[best_r::m] = 1
                used_mods.add(m)
                current_uncovered = int(covered.sum())
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c[:] for c in classes]

    return best_classes
# EVOLVE-BLOCK-END
=======
"""Structural change: simulated annealing over class configurations for odd covering systems."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    base = 51975  # 3^3 * 5^2 * 7 * 11, all odd
    rng = np.random.default_rng(42)

    # Get all odd divisors > 1 of base, sorted ascending
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()
    div_set = set(divisors)

    def compute_coverage(classes):
        """Compute covered array from list of [m, r] classes."""
        cov = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            cov[r::m] = 1
        return cov

    def uncovered_count(covered):
        return int(covered.sum())

    # Phase 1: Greedy construction to get a starting point
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    used_mods = set()

    for m in divisors:
        if time.time() > deadline - 30.0:
            break
        best_r = 0
        best_gain = -1
        for r in range(m):
            uncovered_gain = (base // m) - int(covered[r::m].sum())
            if uncovered_gain > best_gain:
                best_gain = uncovered_gain
                best_r = r
        if best_gain > 0:
            classes.append([m, best_r])
            covered[best_r::m] = 1
            used_mods.add(m)

    current_uncovered = int(covered.sum())
    best_uncovered = current_uncovered
    best_classes = [c[:] for c in classes]
    best_covered = covered.copy()

    # Phase 2: Simulated Annealing
    # State: list of [m, r] pairs with distinct moduli
    # Moves: 
    #   1. Change residue of existing class
    #   2. Swap moduli between two classes (with appropriate residues)
    #   3. Add a new unused modulus
    #   4. Remove a class
    #   5. Perturb multiple residues at once
    
    t_start = time.time()
    T = 5.0  # initial temperature
    T_min = 0.1
    cooling = 0.995
    
    # Precompute for speed: for each divisor m, the number of positions is base//m
    mod_to_count = {m: base // m for m in divisors}
    
    sa_iters = 0
    while time.time() < deadline - 5.0 and T > T_min:
        sa_iters += 1
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Move 1: Change residue of a random existing class
            if len(classes) == 0:
                T *= cooling
                continue
            idx = int(rng.integers(0, len(classes)))
            m, r = classes[idx]
            new_r = int(rng.integers(0, m))
            if new_r == r:
                T *= cooling
                continue
            
            # Compute delta
            old_cov = covered[r::m].copy()
            new_cov = covered[new_r::m].copy()
            
            # Positions that were covered only by this class
            # and positions newly covered by new residue
            # Delta in uncovered = (newly uncovered) - (newly covered)
            # newly uncovered: positions covered by old but not by new and not by others
            # newly covered: positions covered by new but not by old and not by others
            
            # Simpler: compute full delta
            # Remove old, add new
            temp_cov = covered.copy()
            temp_cov[r::m] = 0
            # Count what old uniquely covered
            old_unique = int((temp_cov[r::m] == 0).sum())  # these become uncovered
            temp_cov[new_r::m] = 1
            # Count what new uniquely covers
            new_unique = int((temp_cov[new_r::m] == 1).sum() - int(covered[new_r::m].sum()))
            # Actually let's just compute directly
            delta = 0
            # Positions lost: covered by old, not by new, and not covered by any other class
            lost = 0
            gained = 0
            for pos in range(base):
                if covered[pos] == 1 and temp_cov_before_change := None:
                    pass
            
            # Let me use a cleaner approach
            # Save state
            save_cov = covered.copy()
            covered[r::m] = 0
            covered[new_r::m] = 1
            new_uncovered = int(covered.sum())
            delta = new_uncovered - current_uncovered
            covered = save_cov
            
            # Accept?
            if delta < 0 or rng.random() < np.exp(-delta / T):
                classes[idx][1] = new_r
                covered = save_cov
                covered[r::m] = 0
                covered[new_r::m] = 1
                current_uncovered = new_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c[:] for c in classes]
                    best_covered = covered.copy()
        
        elif move_type < 0.6:
            # Move 2: Add a new unused modulus
            unused = [m for m in divisors if m not in used_mods]
            if not unused:
                T *= cooling
                continue
            m = unused[int(rng.integers(0, len(unused)))]
            r = int(rng.integers(0, m))
            
            save_cov = covered.copy()
            covered[r::m] = 1
            new_uncovered = int(covered.sum())
            delta = new_uncovered - current_uncovered
            covered = save_cov
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                classes.append([m, r])
                used_mods.add(m)
                covered = save_cov
                covered[r::m] = 1
                current_uncovered = new_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c[:] for c in classes]
                    best_covered = covered.copy()
        
        elif move_type < 0.75:
            # Move 3: Remove a class
            if len(classes) == 0:
                T *= cooling
                continue
            idx = int(rng.integers(0, len(classes)))
            m, r = classes[idx]
            
            save_cov = covered.copy()
            covered[r::m] = 0
            new_uncovered = int(covered.sum())
            delta = new_uncovered - current_uncovered
            covered = save_cov
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                removed = classes.pop(idx)
                used_mods.discard(m)
                covered = save_cov
                covered[r::m] = 0
                current_uncovered = new_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c[:] for c in classes]
                    best_covered = covered.copy()
        
        elif move_type < 0.85:
            # Move 4: Swap residues of two classes
            if len(classes) < 2:
                T *= cooling
                continue
            i, j = sorted(rng.choice(len(classes), size=2, replace=False))
            m1, r1 = classes[i]
            m2, r2 = classes[j]
            
            # Try swapping residues
            save_cov = covered.copy()
            covered[r1::m1] = 0
            covered[r2::m2] = 0
            covered[r2::m1] = 1
            covered[r1::m2] = 1
            new_uncovered = int(covered.sum())
            delta = new_uncovered - current_uncovered
            covered = save_cov
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                classes[i] = [m1, r2]
                classes[j] = [m2, r1]
                covered = save_cov
                covered[r1::m1] = 0
                covered[r2::m2] = 0
                covered[r2::m1] = 1
                covered[r1::m2] = 1
                current_uncovered = new_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c[:] for c in classes]
                    best_covered = covered.copy()
        
        else:
            # Move 5: Perturb k random residues at once (k=2 or 3)
            k = int(rng.integers(2, 4))
            if len(classes) < k:
                T *= cooling
                continue
            indices = rng.choice(len(classes), size=k, replace=False)
            old_residues = [classes[i][1] for i in indices]
            new_residues = [int(rng.integers(0, classes[i][0])) for i in indices]
            
            if all(old_residues[i] == new_residues[i] for i in range(k)):
                T *= cooling
                continue
            
            save_cov = covered.copy()
            for i, idx in enumerate(indices):
                m, r = classes[idx]
                covered[r::m] = 0
            for i, idx in enumerate(indices):
                m, r = classes[idx]
                covered[new_residues[i]::m] = 1
            new_uncovered = int(covered.sum())
            delta = new_uncovered - current_uncovered
            covered = save_cov
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                for i, idx in enumerate(indices):
                    classes[idx][1] = new_residues[i]
                covered = save_cov
                for i, idx in enumerate(indices):
                    m, r = classes[idx]
                    covered[r::m] = 0
                for i, idx in enumerate(indices):
                    m, r = classes[idx]
                    covered[new_residues[i]::m] = 1
                current_uncovered = new_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c[:] for c in classes]
                    best_covered = covered.copy()
        
        # Cool down
        T *= cooling
        
        # Periodic local improvement
        if sa_iters % 200 == 0 and time.time() < deadline - 10.0:
            # Quick local improvement: for each class, try all residues if small
            improved = True
            while improved and time.time() < deadline - 8.0:
                improved = False
                for i in range(len(classes)):
                    if time.time() > deadline - 8.0:
                        break
                    m, r = classes[i]
                    if m > 100:
                        continue  # Skip large moduli for speed
                    save_cov = covered.copy()
                    covered[r::m] = 0
                    best_r_new = r
                    best_uncovered_new = current_uncovered
                    for r_new in range(m):
                        if r_new == r:
                            continue
                        covered[r_new::m] = 1
                        u = int(covered.sum())
                        if u < best_uncovered_new:
                            best_uncovered_new = u
                            best_r_new = r_new
                        covered[r_new::m] = 0
                    covered = save_cov
                    if best_r_new != r:
                        covered[r::m] = 0
                        covered[best_r_new::m] = 1
                        classes[i] = [m, best_r_new]
                        current_uncovered = best_uncovered_new
                        if current_uncovered < best_uncovered:
                            best_uncovered = current_uncovered
                            best_classes = [c[:] for c in classes]
                            best_covered = covered.copy()
                        improved = True

    # Final: return best found
    return best_classes
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I notice I have a bug in the code (the `temp_cov_before_change` line is nonsense). Let me fix that:

