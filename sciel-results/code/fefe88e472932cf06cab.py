import time
import numpy as np
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Precompute all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # ascending order
    n_div = len(divisors)
    
    def score_solution(classes):
        """Count uncovered residues for a list of [m, r] classes."""
        if not classes:
            return base
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(np.sum(covered == 0))
    
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
                
                # Find best residue for this modulus
                best_r_for_m = 0
                best_count_for_m = -1
                for r in range(m):
                    cnt = int(np.sum(covered[r::m] == 0))
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
            covered[best_r::best_m] = 1
            available.remove(best_m)
        
        return classes, covered
    
    def local_search(classes, base, deadline):
        """Local search: try changing residues and swapping classes."""
        if not classes:
            return classes, base
        
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try changing residue for each class
            for idx in range(len(best_classes)):
                if time.time() > deadline - 10.0:
                    break
                m, r = best_classes[idx]
                # Try a few alternative residues
                for r2 in [0, m//2, m-1, (r+1)%m, (r-1)%m]:
                    if r2 == r:
                        continue
                    new_classes = [c[:] for c in best_classes]
                    new_classes[idx] = [m, r2]
                    new_score = score_solution(new_classes)
                    if new_score < best_score:
                        best_score = new_score
                        best_classes = new_classes
                        improved = True
                        break
                if improved:
                    break
            
            # Try swapping two classes
            if not improved and len(best_classes) >= 2:
                for i in range(len(best_classes)):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(i+1, len(best_classes)):
                        new_classes = [c[:] for c in best_classes]
                        new_classes[i], new_classes[j] = new_classes[j], new_classes[i]
                        new_score = score_solution(new_classes)
                        if new_score < best_score:
                            best_score = new_score
                            best_classes = new_classes
                            improved = True
                            break
                    if improved:
                        break
        
        return best_classes, best_score
    
    def ilp_refinement(initial_classes, base, deadline):
        """Use ILP to refine the solution by optimizing residue choices for selected moduli."""
        if not initial_classes:
            return initial_classes, base
        
        # Extract selected moduli and their current residues
        selected_moduli = [c[0] for c in initial_classes]
        n_sel = len(selected_moduli)
        
        if n_sel == 0 or n_sel > 30:  # ILP becomes too large
            return initial_classes, score_solution(initial_classes)
        
        # For each selected modulus, we have m possible residues
        # Variables: x[i][r] = 1 if modulus i uses residue r
        # This is a large problem, so we'll use a simplified approach:
        # Fix the set of moduli, optimize residues via local search with more iterations
        
        # Better approach: Use ILP to decide which moduli to keep
        # But that's also large. Instead, do intensive local search on residues
        
        best_classes = [c[:] for c in initial_classes]
        best_score = score_solution(best_classes)
        
        # Multi-start local search on residues
        rng = random.Random(123)
        for start_iter in range(5):
            if time.time() > deadline - 15.0:
                break
            
            # Perturb residues randomly
            perturbed = [c[:] for c in best_classes]
            for idx in range(len(perturbed)):
                m = perturbed[idx][0]
                if rng.random() < 0.3:  # 30% chance to change
                    perturbed[idx][1] = rng.randint(0, m - 1)
            
            # Local search on perturbed
            current_score = score_solution(perturbed)
            improved = True
            while improved and time.time() < deadline - 15.0:
                improved = False
                for idx in range(len(perturbed)):
                    if time.time() > deadline - 15.0:
                        break
                    m, r = perturbed[idx]
                    # Try all residues for this modulus (if m is small enough)
                    if m <= 50:
                        best_r = r
                        best_cnt = -1
                        for r2 in range(m):
                            # Quick estimate: count how many new residues this would cover
                            # Approximate by checking overlap with other classes
                            cnt = 0
                            for pos in range(r2, base, m):
                                # Check if this position is covered by other classes
                                is_covered = False
                                for other_idx in range(len(perturbed)):
                                    if other_idx == idx:
                                        continue
                                    om, or_ = perturbed[other_idx]
                                    if (pos - or_) % om == 0:
                                        is_covered = True
                                        break
                                if not is_covered:
                                    cnt += 1
                            if cnt > best_cnt:
                                best_cnt = cnt
                                best_r = r2
                        if best_r != r:
                            perturbed[idx][1] = best_r
                            new_score = score_solution(perturbed)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                perturbed[idx][1] = r  # revert
                    else:
                        # For large moduli, try a few random residues
                        for _ in range(5):
                            r2 = rng.randint(0, m - 1)
                            if r2 == r:
                                continue
                            old_r = perturbed[idx][1]
                            perturbed[idx][1] = r2
                            new_score = score_solution(perturbed)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                perturbed[idx][1] = old_r
                    if improved:
                        break
            
            if current_score < best_score:
                best_score = current_score
                best_classes = [c[:] for c in perturbed]
        
        return best_classes, best_score
    
    best_score = base
    best_classes = []
    
    # Strategy 1: Greedy with ascending divisors
    if time.time() < deadline - 30.0:
        classes, covered = greedy_pass(divisors, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 2: Greedy with descending divisors
    if time.time() < deadline - 30.0:
        div_desc = list(reversed(divisors))
        classes, covered = greedy_pass(div_desc, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 3: Randomized greedy attempts
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 40.0 and attempts < 5:
        attempts += 1
        div_rand = list(divisors)
        rng.shuffle(div_rand)
        classes, covered = greedy_pass(div_rand, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # ILP-inspired refinement (intensive local search on residues)
    if best_classes and time.time() < deadline - 20.0:
        best_classes, best_score = ilp_refinement(best_classes, base, deadline)
    
    # Final local search
    if best_classes and time.time() < deadline - 10.0:
        best_classes, best_score = local_search(best_classes, base, deadline)
    
    return best_classes if best_classes else []
