import time
import numpy as np
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    
    # Precompute all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # ascending order
    n_div = len(divisors)
    
    # Precompute residue index arrays for each divisor
    # For divisor m, residues[r] = array of positions congruent to r mod m
    residue_positions = {}
    for m in divisors:
        residue_positions[m] = [np.arange(r, base, m) for r in range(m)]
    
    def score_solution(classes):
        """Count uncovered residues for a list of [m, r] classes."""
        if not classes:
            return base
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[residue_positions[m][r]] = 1
        return int(np.sum(covered == 0))
    
    def fast_local_search(classes, base, deadline):
        """High-throughput local search using incremental coverage counts."""
        if not classes:
            return classes, base
        
        n = len(classes)
        if n == 0:
            return classes, base
        
        # Build coverage count array: how many classes cover each position
        cov = np.zeros(base, dtype=np.int16)
        for m, r in classes:
            cov[residue_positions[m][r]] += 1
        current_uncovered = int(np.sum(cov == 0))
        
        best_classes = [c[:] for c in classes]
        best_uncovered = current_uncovered
        rng = random.Random(123)
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # Try changing residue for each class (all residues for small m, random for large)
            for idx in range(n):
                if time.time() > deadline - 3.0:
                    break
                m, r = best_classes[idx]
                pos = residue_positions[m][r]
                
                # Determine candidate residues
                if m <= 30:
                    candidates = [r2 for r2 in range(m) if r2 != r]
                else:
                    candidates = [rng.randint(0, m - 1) for _ in range(15)]
                    candidates = [c for c in candidates if c != r]
                
                for r2 in candidates:
                    if time.time() > deadline - 3.0:
                        break
                    pos2 = residue_positions[m][r2]
                    # Positions that lose coverage: in pos but not in pos2
                    # Positions that gain coverage: in pos2 but not in pos
                    # Net change in uncovered = (gains that were uncovered) - (losses that become uncovered)
                    # A position p in pos loses its only coverage if cov[p] == 1 and p not in pos2
                    # A position p in pos2 gains coverage if cov[p] == 0 and p not in pos
                    
                    # Compute loss: positions in pos where cov==1 and not in pos2
                    # Since pos and pos2 are disjoint (same modulus, different residues), 
                    # positions in pos are NOT in pos2
                    loss = int(np.sum(cov[pos] == 1))
                    # Compute gain: positions in pos2 where cov==0
                    gain = int(np.sum(cov[pos2] == 0))
                    
                    delta_uncovered = loss - gain
                    if delta_uncovered < 0:
                        # Accept: update cov
                        cov[pos] -= 1
                        cov[pos2] += 1
                        best_classes[idx][1] = r2
                        current_uncovered += delta_uncovered
                        best_uncovered = current_uncovered
                        improved = True
                        break
                    # else: no change to cov (we didn't modify it)
            
            # Try adding a new class
            if not improved:
                used_moduli = set(c[0] for c in best_classes)
                # Sort unused divisors by potential gain
                unused = [m for m in divisors if m not in used_moduli]
                for m in unused:
                    if time.time() > deadline - 3.0:
                        break
                    if m > 200:
                        # For large moduli, only try a few residues
                        candidates = [rng.randint(0, m - 1) for _ in range(8)]
                    else:
                        candidates = list(range(m))
                    
                    best_r = None
                    best_gain = 0
                    for r2 in candidates:
                        if time.time() > deadline - 3.0:
                            break
                        pos2 = residue_positions[m][r2]
                        gain = int(np.sum(cov[pos2] == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r2
                    
                    if best_r is not None and best_gain > 0:
                        # Add the class
                        pos2 = residue_positions[m][best_r]
                        cov[pos2] += 1
                        best_classes.append([m, best_r])
                        current_uncovered -= best_gain
                        best_uncovered = current_uncovered
                        improved = True
                        break
            
            # Try removing a class that has zero unique coverage
            if not improved and len(best_classes) > 3:
                for idx in range(len(best_classes)):
                    if time.time() > deadline - 3.0:
                        break
                    m, r = best_classes[idx]
                    pos = residue_positions[m][r]
                    # Check if any position in pos has cov==1 (unique coverage)
                    unique = int(np.sum(cov[pos] == 1))
                    if unique == 0:
                        # Safe to remove
                        cov[pos] -= 1
                        best_classes.pop(idx)
                        n -= 1
                        best_uncovered = current_uncovered  # unchanged
                        improved = True
                        break
        
        return best_classes, best_uncovered
    
    def compute_covered(classes):
        """Compute covered array from classes."""
        if not classes:
            return np.zeros(base, dtype=np.uint8)
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[residue_positions[m][r]] = 1
        return covered
    
    def greedy_pass(div_order, base, deadline, covered_init=None):
        """One greedy pass: pick best (m,r) at each step from div_order."""
        covered = np.zeros(base, dtype=np.uint8) if covered_init is None else covered_init.copy()
        classes = []
        available = set(div_order)
        
        while available:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_m = None
            best_r = None
            
            for m in div_order:
                if m not in available:
                    continue
                
                # Max possible coverage for this modulus
                max_possible = (base + m - 1) // m
                if max_possible <= best_gain:
                    continue
                
                # Find best residue for this modulus using vectorized approach
                best_r_for_m = 0
                best_count_for_m = -1
                for r in range(m):
                    cnt = int(np.sum(covered[residue_positions[m][r]] == 0))
                    if cnt > best_count_for_m:
                        best_count_for_m = cnt
                        best_r_for_m = r
                
                if best_count_for_m > best_gain:
                    best_gain = best_count_for_m
                    best_m = m
                    best_r = best_r_for_m
            
            if best_m is None or best_gain <= 0:
                break
            
            classes.append([best_m, best_r])
            covered[residue_positions[best_m][best_r]] = 1
            available.remove(best_m)
        
        return classes, covered
    
    def local_search_full(classes, base, deadline):
        """Intensive local search using incremental coverage tracking."""
        if not classes:
            return classes, base
        
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        rng = random.Random(7)
        
        # Use fast local search as the primary method
        result_classes, result_score = fast_local_search(best_classes, base, deadline)
        if result_score < best_score:
            best_classes = result_classes
            best_score = result_score
        
        return best_classes, best_score
    
    def milp_residue_optimize(classes, base, deadline):
        """Use MILP to optimize residues for a subset of small moduli."""
        if not classes:
            return classes, base
        
        # Only work with small moduli where MILP is tractable
        small_classes = [(i, m, r) for i, (m, r) in enumerate(classes) if m <= 30]
        if len(small_classes) < 2:
            return classes, score_solution(classes)
        
        # Build MILP: for each small modulus, choose one residue
        # Maximize total coverage
        n_vars = sum(m for _, m, _ in small_classes)
        n_constraints = len(small_classes)
        
        # Objective: maximize coverage of each position
        # c[p] = sum over (i, r) where p in residue_positions[m_i][r] of x[i, r]
        # We want to maximize sum_p min(1, c[p]) which is non-linear
        # Approximate: maximize sum_p c[p] (counts multiple coverage)
        # This is a linear proxy
        
        c_obj = np.zeros(n_vars)
        var_idx = 0
        for ci, (orig_idx, m, r) in enumerate(small_classes):
            for r2 in range(m):
                pos = residue_positions[m][r2]
                c_obj[var_idx:var_idx+len(pos)] += 1.0  # wrong dimension
                var_idx += 1
        
        # Actually, objective is per-variable: each variable contributes to coverage
        # c_obj[j] = number of positions covered by variable j
        c_obj = np.zeros(n_vars)
        var_idx = 0
        for ci, (orig_idx, m, r) in enumerate(small_classes):
            for r2 in range(m):
                c_obj[var_idx] = len(residue_positions[m][r2])
                var_idx += 1
        
        # Constraints: for each modulus, exactly one residue selected
        A_rows = []
        for ci, (orig_idx, m, r) in enumerate(small_classes):
            row = np.zeros(n_vars)
            var_idx = 0
            for cj, (oi2, m2, r2) in enumerate(small_classes):
                if cj == ci:
                    row[var_idx:var_idx+m] = 1.0
                var_idx += m
            A_rows.append(row)
        
        A = np.array(A_rows)
        lb = np.ones(n_constraints)
        ub = np.ones(n_constraints)
        
        try:
            res = milp(
                c=-c_obj,  # minimize negative = maximize
                constraints=LinearConstraint(A, lb, ub),
                integrality=np.ones(n_vars),
                bounds=Bounds(0, 1),
                options={'time_limit': max(1, deadline - time.time() - 10)}
            )
            if res.status == 0 or res.status == 2:  # optimal or time limit with solution
                if res.x is not None and np.any(res.x > 0.5):
                    new_classes = [c[:] for c in classes]
                    var_idx = 0
                    for ci, (orig_idx, m, r) in enumerate(small_classes):
                        for r2 in range(m):
                            if res.x[var_idx] > 0.5:
                                new_classes[orig_idx][1] = r2
                                break
                            var_idx += 1
                    new_score = score_solution(new_classes)
                    old_score = score_solution(classes)
                    if new_score < old_score:
                        return new_classes, new_score
        except Exception:
            pass
        
        return classes, score_solution(classes)
    
    def local_search(classes, base, deadline):
        """Local search: try changing residues and swapping classes."""
        if not classes:
            return classes, base
        
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try changing residue for each class
            for idx in range(len(best_classes)):
                if time.time() > deadline - 5.0:
                    break
                m, r = best_classes[idx]
                # Try a few alternative residues
                candidates = [0, m//2, m-1, (r+1)%m, (r-1)%m]
                if m <= 20:
                    candidates = list(range(m))
                for r2 in candidates:
                    if r2 == r:
                        continue
                    best_classes[idx][1] = r2
                    new_score = score_solution(best_classes)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        break
                    else:
                        best_classes[idx][1] = r
                if improved:
                    break
            
            # Try adding a new class if we have unused divisors
            if not improved:
                used_moduli = set(c[0] for c in best_classes)
                for m in divisors:
                    if m in used_moduli:
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    # Try all residues for this new modulus
                    best_r = None
                    best_new_score = best_score
                    for r in range(m):
                        if m > 100 and r > 20:
                            break
                        best_classes.append([m, r])
                        new_score = score_solution(best_classes)
                        best_classes.pop()
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_r = r
                    if best_r is not None:
                        best_classes.append([m, best_r])
                        best_score = best_new_score
                        improved = True
                        break
            
            # Try removing a class if it doesn't help
            if not improved and len(best_classes) > 5:
                for idx in range(len(best_classes)):
                    if time.time() > deadline - 5.0:
                        break
                    saved = best_classes.pop(idx)
                    new_score = score_solution(best_classes)
                    if new_score <= best_score:
                        best_score = new_score
                        improved = True
                    else:
                        best_classes.insert(idx, saved)
                    if improved:
                        break
        
        return best_classes, best_score
    
    best_score = base
    best_classes = []
    
    # Strategy 1: Greedy with ascending divisors
    if time.time() < deadline - 40.0:
        classes, covered = greedy_pass(divisors, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 2: Greedy with descending divisors
    if time.time() < deadline - 40.0:
        div_desc = list(reversed(divisors))
        classes, covered = greedy_pass(div_desc, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 3: Greedy with medium-first order
    if time.time() < deadline - 40.0:
        med_order = sorted(divisors, key=lambda m: abs(m - 100))
        classes, covered = greedy_pass(med_order, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 4: Randomized greedy attempts
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 30.0 and attempts < 8:
        attempts += 1
        div_rand = list(divisors)
        rng.shuffle(div_rand)
        classes, covered = greedy_pass(div_rand, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # MILP refinement on small moduli
    if best_classes and time.time() < deadline - 25.0:
        best_classes, best_score = milp_residue_optimize(best_classes, base, deadline)
    
    # Full local search
    if best_classes and time.time() < deadline - 15.0:
        best_classes, best_score = local_search_full(best_classes, base, deadline)
    
    # MILP again after local search
    if best_classes and time.time() < deadline - 10.0:
        best_classes, best_score = milp_residue_optimize(best_classes, base, deadline)
    
    # Final local search
    if best_classes and time.time() < deadline - 5.0:
        best_classes, best_score = local_search(best_classes, base, deadline)
    
    # Final polish: exhaustive residue search for small moduli
    if best_classes and time.time() < deadline - 2.0:
        # Try all residues for each small modulus one more time
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for idx in range(len(best_classes)):
                if time.time() > deadline - 2.0:
                    break
                m, r = best_classes[idx]
                if m > 40:
                    continue
                best_r = r
                best_s = best_score
                for r2 in range(m):
                    if r2 == r:
                        continue
                    best_classes[idx][1] = r2
                    s = score_solution(best_classes)
                    if s < best_s:
                        best_s = s
                        best_r = r2
                if best_r != r:
                    best_classes[idx][1] = best_r
                    best_score = best_s
                    changed = True
    
    return best_classes if best_classes else []
