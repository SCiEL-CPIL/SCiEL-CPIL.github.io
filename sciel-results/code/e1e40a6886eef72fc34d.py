import time
import math
import random

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    best_score = 0.0
    best_classes = []
    
    # Precompute GCDs for moduli in [20, 200]
    moduli = list(range(20, 201))
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
            
    def check_disjoint(m, r, classes, exclude_idx=None):
        """
        Check if (m, r) is disjoint from all classes in 'classes' except at exclude_idx.
        classes is a list of [m, r] pairs.
        """
        for i, (m2, r2) in enumerate(classes):
            if i == exclude_idx:
                continue
            g = gcd_table.get((m, m2), math.gcd(m, m2))
            if r % g == r2 % g:
                return False
        return True

    def get_valid_residues(m, classes, exclude_idx=None):
        """
        Return list of valid residues for modulus m given current classes.
        """
        valid = []
        for r in range(m):
            if check_disjoint(m, r, classes, exclude_idx):
                valid.append(r)
        return valid

    def initial_greedy():
        classes = []
        # Sort by 1/m descending (i.e., m ascending) to prioritize high value
        for m in sorted(moduli, key=lambda x: 1.0/x):
            residues = list(range(m))
            random.shuffle(residues)
            found = False
            for r in residues:
                if check_disjoint(m, r, classes):
                    classes.append([m, r])
                    found = True
                    break
        return classes

    def local_search_add(classes):
        """
        Greedily add any possible modulus.
        """
        changed = True
        while changed:
            changed = False
            current_mods = {m for m, r in classes}
            for m in moduli:
                if m in current_mods:
                    continue
                valid_residues = get_valid_residues(m, classes)
                if valid_residues:
                    # Pick a random valid residue to add diversity
                    r = random.choice(valid_residues)
                    classes.append([m, r])
                    current_mods.add(m)
                    changed = True
        return classes

    def simulated_annealing():
        # Initialize
        current_classes = initial_greedy()
        current_classes = local_search_add(current_classes)
        current_score = sum(1.0 / m for m, r in current_classes)
        
        best_classes = [c[:] for c in current_classes]
        best_score = current_score
        
        T = 1.0
        cooling_rate = 0.99
        min_T = 0.01
        
        while T > min_T and time.time() < deadline:
            # Generate a neighbor
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random modulus
                current_mods = {m for m, r in current_classes}
                inactive_mods = [m for m in moduli if m not in current_mods]
                if not inactive_mods:
                    T *= cooling_rate
                    continue
                
                m = random.choice(inactive_mods)
                valid_residues = get_valid_residues(m, current_classes)
                if valid_residues:
                    r = random.choice(valid_residues)
                    new_classes = current_classes + [[m, r]]
                    new_score = current_score + 1.0 / m
                    # Always accept improvement
                    current_classes = new_classes
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                else:
                    # No valid residue, do nothing (or try to change existing residues?)
                    pass
                    
            elif move_type < 0.8:
                # Try to change residue of an existing modulus
                if not current_classes:
                    T *= cooling_rate
                    continue
                
                idx = random.randint(0, len(current_classes) - 1)
                m, r_old = current_classes[idx]
                valid_residues = get_valid_residues(m, current_classes, exclude_idx=idx)
                if len(valid_residues) > 1:
                    # Pick a different valid residue
                    r_new = random.choice(valid_residues)
                    if r_new != r_old:
                        # Score doesn't change, but feasibility landscape does
                        # Accept with high probability to explore
                        if random.random() < 0.8:
                            current_classes[idx] = [m, r_new]
                            
            else:
                # Try to remove a modulus
                if not current_classes:
                    T *= cooling_rate
                    continue
                
                idx = random.randint(0, len(current_classes) - 1)
                m, r = current_classes[idx]
                new_classes = current_classes[:idx] + current_classes[idx+1:]
                new_score = current_score - 1.0 / m
                
                # Accept with probability exp(delta / T)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_classes = new_classes
                    current_score = new_score
            
            T *= cooling_rate
            
        return best_classes, best_score

    # Main loop
    iterations = 0
    while time.time() < deadline:
        candidate_classes, score = simulated_annealing()
        
        if score > best_score:
            best_score = score
            best_classes = candidate_classes
            
        iterations += 1
        # Optional: break if we hit theoretical max?
        # Theoretical max is sum(1/m for m in 20..200)
        
    return best_classes
