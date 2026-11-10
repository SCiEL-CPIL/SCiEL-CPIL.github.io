import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def check_conflict(res, m, selected):
        """Check if residue res with modulus m conflicts with any selected class."""
        for m2, r2 in selected:
            g = gcd(m, m2)
            if res % g == r2 % g:
                return True
        return False

    def find_valid_residue(m, selected):
        """Find any valid residue for m given selected set. Returns None if none."""
        for r in range(m):
            if not check_conflict(r, m, selected):
                return r
        return None

    def calculate_score(selected):
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return total

    def is_valid_set(selected):
        """Check if the entire set is valid."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                m1, r1 = selected[i]
                m2, r2 = selected[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    def greedy_construct(order):
        """Greedy construction given an order of moduli."""
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])
        return selected

    # 1. Initial Greedy Construction
    # Ascending order is the natural greedy for maximizing sum 1/m
    best_selected = greedy_construct(range(20, 201))
    best_score = calculate_score(best_selected)
    
    # Try a few randomized greedy restarts to find a better starting point
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        selected = []
        # Randomized greedy: at each step, pick from the smallest few available moduli
        available = list(range(20, 201))
        random.shuffle(available)
        # But prefer smaller moduli: sort with some randomness
        # Actually, let's just do a simple randomized greedy where we pick the smallest
        # available modulus that has a valid residue, but with some random tie-breaking
        # for which residue to pick.
        
        # Better: try different orders
        # Order: sorted by m, but with random perturbations in residue choice
        selected = []
        for m in range(20, 201):
            if time.time() > deadline:
                break
            valid_res = []
            for r in range(m):
                if not check_conflict(r, m, selected):
                    valid_res.append(r)
            if valid_res:
                # Pick a random valid residue to explore different paths
                r = random.choice(valid_res)
                selected.append([m, r])
        
        score = calculate_score(selected)
        if score > best_score:
            best_score = score
            best_selected = [x[:] for x in selected]

    selected = [x[:] for x in best_selected]
    current_score = best_score

    # Identify missing moduli
    selected_moduli = set(m for m, r in selected)
    missing_moduli = [m for m in range(20, 201) if m not in selected_moduli]
    missing_moduli.sort()

    # 2. Local Search
    # We use a more robust local search:
    # - Try to add a missing modulus by perturbing existing residues.
    # - Try to swap a selected modulus with a missing one if it improves score.
    
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() < deadline:
        if not missing_moduli:
            break
            
        found_improvement = False
        
        # Strategy 1: Add a missing modulus by perturbing one existing residue
        # Pick a target missing modulus (prefer smaller m for higher 1/m)
        if random.random() < 0.8:
            target_m = missing_moduli[0]
        else:
            target_m = random.choice(missing_moduli)
            
        if not selected:
            break
            
        # Pick a random existing modulus to perturb
        victim_idx = random.randint(0, len(selected) - 1)
        victim_m, victim_r = selected[victim_idx]
        
        # Try a few random new residues for victim
        attempts = 50
        for _ in range(attempts):
            if time.time() > deadline:
                break
            new_victim_r = random.randint(0, victim_m - 1)
            if new_victim_r == victim_r:
                continue
                
            old_r = selected[victim_idx][1]
            selected[victim_idx][1] = new_victim_r
            
            # Check if target_m can be added now
            target_r = find_valid_residue(target_m, selected)
            if target_r is not None:
                # We can add target_m!
                new_score = current_score + Fraction(1, target_m)
                selected.append([target_m, target_r])
                current_score = new_score
                if target_m in missing_moduli:
                    missing_moduli.remove(target_m)
                found_improvement = True
                no_improvement_count = 0
                break
            
            # Revert victim if not accepted
            selected[victim_idx][1] = old_r
            
        if found_improvement:
            continue
            
        # Strategy 2: Swap a selected modulus for a missing one
        # Pick victim with larger m (lower 1/m) and target with smaller m (higher 1/m)
        victim_candidates = [i for i, (m, r) in enumerate(selected) 
                           if any(tm < m for tm in missing_moduli)]
        if victim_candidates:
            victim_idx = random.choice(victim_candidates)
            victim_m, victim_r = selected[victim_idx]
            
            # Pick target: prefer smaller m
            better_targets = [m for m in missing_moduli if m < victim_m]
            if better_targets:
                target_m = random.choice(better_targets)
                
                # Remove victim
                selected.pop(victim_idx)
                
                # Try to add target
                target_r = find_valid_residue(target_m, selected)
                if target_r is not None:
                    # Swap successful and valid
                    selected.append([target_m, target_r])
                    current_score += Fraction(1, target_m) - Fraction(1, victim_m)
                    if target_m in missing_moduli:
                        missing_moduli.remove(target_m)
                    missing_moduli.append(victim_m)
                    missing_moduli.sort()
                    found_improvement = True
                    no_improvement_count = 0
                else:
                    # Revert
                    selected.append([victim_m, victim_r])
        
        # Strategy 3: 2-swap: remove two selected, add two missing
        if not found_improvement and len(selected) >= 2 and len(missing_moduli) >= 2:
            # Try to find a pair of missing moduli that can replace a pair of selected
            # This is expensive, so only do it occasionally
            if random.random() < 0.1:
                # Pick two random selected to remove
                idxs = random.sample(range(len(selected)), 2)
                victim1, victim2 = selected[idxs[0]], selected[idxs[1]]
                
                # Pick two random missing to add
                targets = random.sample(missing_moduli, 2)
                
                # Remove both victims
                for idx in sorted([idxs[0], idxs[1]], reverse=True):
                    selected.pop(idx)
                
                # Try to add both targets
                r1 = find_valid_residue(targets[0], selected)
                if r1 is not None:
                    selected.append([targets[0], r1])
                    r2 = find_valid_residue(targets[1], selected)
                    if r2 is not None:
                        selected.append([targets[1], r2])
                        delta = Fraction(1, targets[0]) + Fraction(1, targets[1]) - Fraction(1, victim1[0]) - Fraction(1, victim2[0])
                        if delta > 0:
                            current_score += delta
                            for tm in targets:
                                if tm in missing_moduli:
                                    missing_moduli.remove(tm)
                            missing_moduli.append(victim1[0])
                            missing_moduli.append(victim2[0])
                            missing_moduli.sort()
                            found_improvement = True
                            no_improvement_count = 0
                        else:
                            # Revert
                            selected.pop()
                            selected.pop()
                            selected.append([victim1[0], victim1[1]])
                            selected.append([victim2[0], victim2[1]])
                    else:
                        # Revert
                        selected.pop()
                        selected.append([victim1[0], victim1[1]])
                        selected.append([victim2[0], victim2[1]])
                else:
                    # Revert
                    selected.append([victim1[0], victim1[1]])
                    selected.append([victim2[0], victim2[1]])
        
        if not found_improvement:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Try a random restart of local search
                no_improvement_count = 0
                # Perturb the solution randomly
                for _ in range(3):
                    idx = random.randint(0, len(selected) - 1)
                    m, r = selected[idx]
                    new_r = random.randint(0, m - 1)
                    # Check if still valid
                    old_r = selected[idx][1]
                    selected[idx][1] = new_r
                    if not is_valid_set(selected):
                        selected[idx][1] = old_r

    # Final validation
    if not is_valid_set(selected):
        # Fallback to greedy if something went wrong
        selected = []
        for m in range(20, 201):
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])

    return selected
