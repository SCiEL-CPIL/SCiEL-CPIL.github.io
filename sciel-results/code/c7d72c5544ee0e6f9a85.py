"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy construction (ascending, prime-group, randomized).
    2. Local search: add, swap, 2-swap moves.
    3. Simulated annealing with proper temperature schedule.
    4. Residue re-optimization to free capacity.
    5. Final polish.
    """
    deadline = time.time() + 143.0  # Safety margin before 150s limit
    
    all_moduli = list(range(20, 201))
    num_moduli = len(all_moduli)
    
    # Precompute gcd table for all pairs
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for j, m2 in enumerate(all_moduli):
            if j > i:
                g = gcd(m1, m2)
                gcd_table[(m1, m2)] = g
                gcd_table[(m2, m1)] = g
    
    def find_valid_residue_fast(m, current_classes):
        """Find a residue r in [0, m) such that (m, r) is disjoint from all current classes."""
        if not current_classes:
            return 0
        
        for cm, cr in current_classes:
            g = gcd_table.get((m, cm), gcd(m, cm))
            if g == 1:
                return None
        
        forbidden = [False] * m
        for cm, cr in current_classes:
            g = gcd_table.get((m, cm), gcd(m, cm))
            if g == 1:
                return None
            forbidden_residue = cr % g
            for r in range(forbidden_residue, m, g):
                forbidden[r] = True
        
        for r in range(m):
            if not forbidden[r]:
                return r
        return None
    
    def find_all_valid_residues(m, current_classes):
        """Find ALL valid residues for m given current_classes."""
        if not current_classes:
            return list(range(m))
        
        for cm, cr in current_classes:
            g = gcd_table.get((m, cm), gcd(m, cm))
            if g == 1:
                return []
        
        forbidden = [False] * m
        for cm, cr in current_classes:
            g = gcd_table.get((m, cm), gcd(m, cm))
            forbidden_residue = cr % g
            for r in range(forbidden_residue, m, g):
                forbidden[r] = True
        
        return [r for r in range(m) if not forbidden[r]]
    
    def calc_score(cls):
        """Calculate sum of 1/m for the given classes."""
        total = Fraction(0, 1)
        for m, r in cls:
            total += Fraction(1, m)
        return total
    
    def calc_score_float(cls):
        """Calculate sum of 1/m as float for quick comparisons."""
        return sum(1.0 / m for m, r in cls)
    
    def is_pairwise_valid(cls):
        """Check if all pairs are disjoint."""
        n = len(cls)
        for i in range(n):
            for j in range(i+1, n):
                m1, r1 = cls[i]
                m2, r2 = cls[j]
                g = gcd_table.get((m1, m2), gcd(m1, m2))
                if g == 1:
                    return False
                if r1 % g == r2 % g:
                    return False
        return True
    
    def local_search(initial_classes, time_limit):
        """Improve a set of classes using local search moves."""
        current_classes = [c[:] for c in initial_classes]
        current_score = calc_score(current_classes)
        best_score = current_score
        best_classes = [c[:] for c in current_classes]
        
        if not is_pairwise_valid(current_classes):
            return best_classes, best_score
        
        in_set = set(c[0] for c in current_classes)
        weights = {m: Fraction(1, m) for m in all_moduli}
        moduli_by_weight_desc = sorted(all_moduli, key=lambda x: -1.0/x)
        
        # Upper bound: sum of all 1/m not yet in set
        def upper_bound():
            ub = calc_score_float(current_classes)
            for m in all_moduli:
                if m not in in_set:
                    ub += 1.0 / m
            return ub
        
        while time.time() < time_limit:
            improved = False
            
            # Move 1: Try adding a missing modulus
            for m in moduli_by_weight_desc:
                if m in in_set:
                    continue
                r = find_valid_residue_fast(m, current_classes)
                if r is not None:
                    new_classes = current_classes + [[m, r]]
                    current_classes = new_classes
                    current_score = current_score + weights[m]
                    in_set.add(m)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                    break
            
            # Move 2: Try 1-swap
            if not improved and time.time() < time_limit:
                sorted_indices = sorted(range(len(current_classes)),
                                       key=lambda i: 1.0/current_classes[i][0])
                
                for i in sorted_indices:
                    m_old = current_classes[i][0]
                    
                    for m_new in moduli_by_weight_desc:
                        if m_new in in_set or m_new == m_old:
                            continue
                        if 1.0/m_new <= 1.0/m_old:
                            break
                        
                        candidate = [c[:] for c in current_classes]
                        candidate.pop(i)
                        r_new = find_valid_residue_fast(m_new, candidate)
                        if r_new is not None:
                            candidate.append([m_new, r_new])
                            new_score = current_score - weights[m_old] + weights[m_new]
                            if new_score > current_score:
                                current_classes = candidate
                                current_score = new_score
                                in_set.remove(m_old)
                                in_set.add(m_new)
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_classes = [c[:] for c in current_classes]
                                break
                    if improved:
                        break
            
            # Move 3: Try 2-swap
            if not improved and time.time() < time_limit:
                n = len(current_classes)
                for i in range(n):
                    if time.time() >= time_limit:
                        break
                    for j in range(i+1, n):
                        if time.time() >= time_limit:
                            break
                        m_remove1 = current_classes[i][0]
                        m_remove2 = current_classes[j][0]
                        
                        candidate = [c[:] for c in current_classes]
                        candidate.pop(j)
                        candidate.pop(i)
                        
                        add_options = [m for m in moduli_by_weight_desc
                                      if m not in set(c[0] for c in candidate)]
                        
                        found = False
                        for m_add1 in add_options[:20]:
                            if found or time.time() >= time_limit:
                                break
                            r_add1 = find_valid_residue_fast(m_add1, candidate)
                            if r_add1 is None:
                                continue
                            temp_candidate = candidate + [[m_add1, r_add1]]
                            
                            for m_add2 in add_options[:20]:
                                if m_add2 == m_add1:
                                    continue
                                r_add2 = find_valid_residue_fast(m_add2, temp_candidate)
                                if r_add2 is None:
                                    continue
                                
                                new_candidate = temp_candidate + [[m_add2, r_add2]]
                                new_score = (current_score
                                            - weights[m_remove1]
                                            - weights[m_remove2]
                                            + weights[m_add1]
                                            + weights[m_add2])
                                if new_score > current_score:
                                    current_classes = new_candidate
                                    current_score = new_score
                                    in_set = set(c[0] for c in current_classes)
                                    improved = True
                                    found = True
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_classes = [c[:] for c in current_classes]
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_classes, best_score
    
    def simulated_annealing(initial_classes, time_limit):
        """Proper SA with temperature schedule to escape local optima."""
        current_classes = [c[:] for c in initial_classes]
        current_score = calc_score_float(current_classes)
        best_score = current_score
        best_classes = [c[:] for c in current_classes]
        
        in_set = set(c[0] for c in current_classes)
        moduli_by_weight_desc = sorted(all_moduli, key=lambda x: -1.0/x)
        
        # Temperature schedule
        T_start = 0.15
        T_end = 0.001
        # Estimate number of iterations we can do
        start_time = time.time()
        # Assume ~500us per iteration
        est_iters = int((time_limit - start_time) * 1000000 / 500)
        est_iters = max(est_iters, 1000)
        cooling_rate = (T_end / T_start) ** (1.0 / est_iters)
        
        T = T_start
        iter_count = 0
        
        while time.time() < time_limit - 1.0:
            iter_count += 1
            T *= cooling_rate
            
            # Choose a random move type
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random missing modulus
                missing = [m for m in all_moduli if m not in in_set]
                if not missing:
                    continue
                m = random.choice(missing)
                # Bias toward smaller moduli
                if random.random() < 0.5:
                    m = min(missing, key=lambda x: x)
                
                r = find_valid_residue_fast(m, current_classes)
                if r is None:
                    continue
                
                old_score = current_score
                new_classes = current_classes + [[m, r]]
                new_score = old_score + 1.0/m
                
                delta = new_score - old_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_classes = new_classes
                    current_score = new_score
                    in_set.add(m)
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                        
            elif move_type < 0.6:
                # Remove a random modulus
                if not current_classes:
                    continue
                idx = random.randint(0, len(current_classes) - 1)
                m_old = current_classes[idx][0]
                
                old_score = current_score
                new_classes = [c[:] for c in current_classes]
                new_classes.pop(idx)
                new_score = old_score - 1.0/m_old
                
                delta = new_score - old_score
                # Always accept removals that free up space (or use SA)
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_classes = new_classes
                    current_score = new_score
                    in_set.remove(m_old)
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                        
            elif move_type < 0.85:
                # Swap: remove one, add another
                if not current_classes:
                    continue
                idx = random.randint(0, len(current_classes) - 1)
                m_old = current_classes[idx][0]
                
                missing = [m for m in all_moduli if m not in in_set]
                if not missing:
                    continue
                # Bias toward smaller moduli
                candidates = sorted(missing, key=lambda x: x)[:30]
                m_new = random.choice(candidates)
                
                candidate = [c[:] for c in current_classes]
                candidate.pop(idx)
                r_new = find_valid_residue_fast(m_new, candidate)
                if r_new is None:
                    continue
                candidate.append([m_new, r_new])
                
                old_score = current_score
                new_score = old_score - 1.0/m_old + 1.0/m_new
                
                delta = new_score - old_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_classes = candidate
                    current_score = new_score
                    in_set.remove(m_old)
                    in_set.add(m_new)
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                        
            else:
                # Residue reassignment: try different residue for a random modulus
                if not current_classes:
                    continue
                idx = random.randint(0, len(current_classes) - 1)
                m = current_classes[idx][0]
                others = [c for i, c in enumerate(current_classes) if i != idx]
                
                valid_residues = find_all_valid_residues(m, others)
                if len(valid_residues) <= 1:
                    continue
                
                old_r = current_classes[idx][1]
                new_r = random.choice(valid_residues)
                if new_r == old_r:
                    continue
                
                # Score doesn't change with residue, but it may enable future additions
                # Accept with some probability to explore
                if random.random() < 0.3 + 0.7 * T / T_start:
                    current_classes[idx] = [m, new_r]
        
        return best_classes, best_score
    
    def residue_reoptimization(classes, time_limit):
        """Try all valid residues for each modulus to maximize future addition capacity."""
        current_classes = [c[:] for c in classes]
        best_classes = [c[:] for c in current_classes]
        best_score = calc_score_float(current_classes)
        
        # For each modulus, try all valid residues and see which allows most additions
        for idx in range(len(current_classes)):
            if time.time() >= time_limit:
                break
            m = current_classes[idx][0]
            others = [c for i, c in enumerate(current_classes) if i != idx]
            
            valid_residues = find_all_valid_residues(m, others)
            if len(valid_residues) <= 1:
                continue
            
            best_r = current_classes[idx][1]
            best_add_count = -1
            
            for r in valid_residues[:50]:  # Limit for speed
                test_classes = [c[:] for c in current_classes]
                test_classes[idx] = [m, r]
                
                # Count how many additional moduli can be added
                in_set = set(c[0] for c in test_classes)
                add_count = 0
                for m_add in all_moduli:
                    if m_add in in_set:
                        continue
                    r_add = find_valid_residue_fast(m_add, test_classes)
                    if r_add is not None:
                        add_count += 1
                        if add_count > best_add_count + 5:
                            break
                
                if add_count > best_add_count:
                    best_add_count = add_count
                    best_r = r
            
            if best_r != current_classes[idx][1]:
                current_classes[idx] = [m, best_r]
        
        # After reoptimization, try to add more moduli
        in_set = set(c[0] for c in current_classes)
        for m in sorted(all_moduli, key=lambda x: x):
            if m in in_set:
                continue
            r = find_valid_residue_fast(m, current_classes)
            if r is not None:
                current_classes.append([m, r])
                in_set.add(m)
        
        score = calc_score_float(current_classes)
        if score > best_score:
            best_classes = [c[:] for c in current_classes]
            best_score = score
        
        return best_classes, best_score
    
    def greedy_ascending():
        """Greedy: add moduli in ascending order (smallest first = highest weight)."""
        classes = []
        in_set = set()
        for m in all_moduli:
            r = find_valid_residue_fast(m, classes)
            if r is not None:
                classes.append([m, r])
                in_set.add(m)
        return classes
    
    def greedy_by_prime_groups():
        """Group moduli by their smallest prime factor, pick best from each group."""
        from math import isqrt
        
        def smallest_prime_factor(n):
            if n % 2 == 0:
                return 2
            for i in range(3, isqrt(n) + 1, 2):
                if n % i == 0:
                    return i
            return n
        
        groups = {}
        for m in all_moduli:
            spf = smallest_prime_factor(m)
            if spf not in groups:
                groups[spf] = []
            groups[spf].append(m)
        
        best_classes = []
        best_score = Fraction(0, 1)
        
        for strategy in range(4):
            classes = []
            in_set = set()
            
            if strategy == 0:
                for spf in sorted(groups.keys()):
                    for m in sorted(groups[spf]):
                        if m in in_set:
                            continue
                        r = find_valid_residue_fast(m, classes)
                        if r is not None:
                            classes.append([m, r])
                            in_set.add(m)
                            break
            elif strategy == 1:
                for spf in sorted(groups.keys()):
                    count = 0
                    for m in sorted(groups[spf]):
                        if m in in_set:
                            continue
                        r = find_valid_residue_fast(m, classes)
                        if r is not None:
                            classes.append([m, r])
                            in_set.add(m)
                            count += 1
                            if count >= 2:
                                break
            elif strategy == 2:
                def factor_count(m):
                    cnt = 0
                    for other in all_moduli:
                        if other != m and gcd(m, other) > 1:
                            cnt += 1
                    return cnt
                
                for m in sorted(all_moduli, key=lambda x: (-factor_count(x), x)):
                    if m in in_set:
                        continue
                    r = find_valid_residue_fast(m, classes)
                    if r is not None:
                        classes.append([m, r])
                        in_set.add(m)
            else:
                for m in sorted(all_moduli):
                    if m in in_set:
                        continue
                    r = find_valid_residue_fast(m, classes)
                    if r is not None:
                        classes.append([m, r])
                        in_set.add(m)
            
            if classes:
                score = calc_score(classes)
                if score > best_score:
                    best_score = score
                    best_classes = [c[:] for c in classes]
        
        return best_classes
    
    random.seed(42)
    
    def greedy_randomized():
        """Greedy with randomized order, multiple restarts."""
        best = []
        best_score = Fraction(0, 1)
        
        for restart in range(15):
            classes = []
            in_set = set()
            order = all_moduli[:]
            random.shuffle(order)
            order.sort(key=lambda m: (random.random() * 0.15 + 1.0/m, m))
            
            for m in order:
                if m in in_set:
                    continue
                r = find_valid_residue_fast(m, classes)
                if r is not None:
                    classes.append([m, r])
                    in_set.add(m)
            
            if classes:
                score = calc_score(classes)
                if score > best_score:
                    best_score = score
                    best = [c[:] for c in classes]
        
        return best
    
    # Run multiple strategies
    best_overall = []
    best_overall_score = Fraction(0, 1)
    
    strategies = [
        ("greedy_ascending", greedy_ascending),
        ("greedy_by_prime_groups", greedy_by_prime_groups),
        ("greedy_randomized", greedy_randomized),
    ]
    
    for name, func in strategies:
        if time.time() > deadline - 40:
            break
        try:
            classes = func()
            if classes and is_pairwise_valid(classes):
                score = calc_score(classes)
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall = [c[:] for c in classes]
        except Exception:
            pass
    
    # Local search on the best initial solution
    if best_overall and time.time() < deadline - 15:
        ls_time = min(time.time() + 10, deadline - 12)
        improved_classes, improved_score = local_search(best_overall, ls_time)
        if improved_score > best_overall_score:
            best_overall = improved_classes
            best_overall_score = improved_score
    
    # Simulated annealing to escape local optima
    if best_overall and time.time() < deadline - 8:
        sa_time = min(time.time() + 8, deadline - 5)
        sa_classes, sa_score = simulated_annealing(best_overall, sa_time)
        if sa_score > calc_score_float(best_overall):
            best_overall = sa_classes
            best_overall_score = calc_score(best_overall)
    
    # Residue reoptimization
    if best_overall and time.time() < deadline - 3:
        ro_classes, ro_score = residue_reoptimization(best_overall, deadline - 2)
        if ro_score > calc_score_float(best_overall):
            best_overall = ro_classes
            best_overall_score = calc_score(best_overall)
    
    # Final polish pass: try to add any remaining moduli
    if best_overall and time.time() < deadline - 1:
        in_set = set(c[0] for c in best_overall)
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for m in sorted(all_moduli, key=lambda x: x):
                if m in in_set:
                    continue
                r = find_valid_residue_fast(m, best_overall)
                if r is not None:
                    best_overall.append([m, r])
                    in_set.add(m)
                    changed = True
    
    # Final validation
    if best_overall:
        if not is_pairwise_valid(best_overall):
            best_overall = greedy_ascending()
            if not is_pairwise_valid(best_overall):
                best_overall = []
    
    return [c[:] for c in best_overall]

# EVOLVE-BLOCK-END
