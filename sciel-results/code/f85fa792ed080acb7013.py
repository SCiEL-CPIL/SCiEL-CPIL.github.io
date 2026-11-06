"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction (descending order for flexibility)
    2. Hill climbing with full residue search
    3. Simulated annealing with efficient neighborhood
    4. Final high-precision polish
    """
    deadline = time.time() + 140.0
    random.seed(42)
    
    def find_valid_residue(m, classes):
        """Find a residue r in [0, m) such that [m, r] is disjoint from all classes."""
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                g = gcd(m, m2)
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def find_all_valid_residues(m, classes):
        """Find all residues r in [0, m) such that [m, r] is disjoint from all classes."""
        valid_residues = []
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                g = gcd(m, m2)
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                valid_residues.append(r)
        return valid_residues
    
    def calc_score(classes):
        """Calculate the sum of 1/m as a float."""
        return sum(1.0 / m for m, r in classes)
    
    def is_valid_classes(classes):
        """Check if all pairs are disjoint."""
        n = len(classes)
        for i in range(n):
            for j in range(i+1, n):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if (r1 % g) == (r2 % g):
                    return False
        return True
    
    def try_add(classes, m):
        """Try to add modulus m with a valid residue. Returns new classes or None."""
        r = find_valid_residue(m, classes)
        if r is not None:
            return classes + [[m, r]]
        return None
    
    # Initial greedy construction - try descending order first
    moduli = list(range(20, 201))
    classes = []
    for m in sorted(moduli, reverse=True):
        if time.time() > deadline - 10.0:
            break
        new_classes = try_add(classes, m)
        if new_classes is not None:
            classes = new_classes
    
    best_classes = classes[:]
    best_score = calc_score(classes)
    
    # Also try ascending order and keep the better one
    classes_asc = []
    for m in sorted(moduli):
        if time.time() > deadline - 10.0:
            break
        new_classes = try_add(classes_asc, m)
        if new_classes is not None:
            classes_asc = new_classes
    if calc_score(classes_asc) > best_score:
        best_classes = classes_asc[:]
        best_score = calc_score(classes_asc)
    
    def hill_climb(classes, time_limit=5.0):
        """Hill climbing polish with full residue search."""
        improved = True
        start = time.time()
        current_score = calc_score(classes)
        
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in moduli if m not in current_moduli]
            
            # Try adding missing moduli
            for m in missing:
                if time.time() - start > time_limit:
                    break
                new_classes = try_add(classes, m)
                if new_classes is not None:
                    new_score = calc_score(new_classes)
                    if new_score > current_score + 1e-12:
                        classes = new_classes
                        current_score = new_score
                        improved = True
            
            # Try swapping: remove one, add another
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in moduli if m not in current_moduli]
            for m_add in sorted(missing, key=lambda x: -1.0/x):
                if time.time() - start > time_limit:
                    break
                for m_remove in sorted(current_moduli, key=lambda x: 1.0/x):
                    if 1.0/m_add <= 1.0/m_remove:
                        continue
                    new_classes = [c for c in classes if c[0] != m_remove]
                    r = find_valid_residue(m_add, new_classes)
                    if r is not None:
                        new_classes.append([m_add, r])
                        new_score = calc_score(new_classes)
                        if new_score > current_score + 1e-12:
                            classes = new_classes
                            current_score = new_score
                            improved = True
                            break
            
            # Try reassigning residues to enable more additions
            for i, (m, r) in enumerate(classes):
                if time.time() - start > time_limit:
                    break
                # Try different residues for this modulus
                for r_new in range(m):
                    if r_new == r:
                        continue
                    new_classes = classes[:]
                    new_classes[i] = [m, r_new]
                    if is_valid_classes(new_classes):
                        # Check if this enables adding a missing modulus
                        current_moduli = set(c[0] for c in new_classes)
                        missing = [mm for mm in moduli if mm not in current_moduli]
                        for mm in missing:
                            if 1.0/mm > 0.005:  # Only try high-value moduli
                                rr = find_valid_residue(mm, new_classes)
                                if rr is not None:
                                    candidate = new_classes + [[mm, rr]]
                                    candidate_score = calc_score(candidate)
                                    if candidate_score > current_score + 1e-12:
                                        classes = candidate
                                        current_score = candidate_score
                                        improved = True
                                        break
                        if improved:
                            break
        return classes, calc_score(classes)
    
    # Simulated annealing with efficient moves
    def sa_polish(classes, temperature=0.5, steps=5000):
        """Simulated annealing with 2-swap moves."""
        current = classes[:]
        current_score = calc_score(current)
        best = classes[:]
        best_score = current_score
        
        for step in range(steps):
            if time.time() > deadline - 10.0:
                break
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Single residue reassignment
                if len(current) == 0:
                    continue
                i = random.randint(0, len(current) - 1)
                m, r = current[i]
                r_new = random.randint(0, m - 1)
                if r_new == r:
                    continue
                new_classes = current[:]
                new_classes[i] = [m, r_new]
                valid = True
                for j in range(len(new_classes)):
                    if j == i:
                        continue
                    m2, r2 = new_classes[j]
                    g = gcd(m, m2)
                    if (r_new % g) == (r2 % g):
                        valid = False
                        break
                if not valid:
                    continue
                delta = 0.0
            elif move_type < 0.6:
                # Add a missing modulus
                current_moduli = set(c[0] for c in current)
                missing = [m for m in moduli if m not in current_moduli]
                if not missing:
                    continue
                m_add = random.choice(missing)
                r = find_valid_residue(m_add, current)
                if r is None:
                    continue
                new_classes = current + [[m_add, r]]
                delta = 1.0 / m_add
            elif move_type < 0.8:
                # Remove a modulus
                if len(current) <= 2:
                    continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                new_classes = [c for c in current if c[0] != m_remove]
                delta = -1.0 / m_remove
            else:
                # 2-swap
                current_moduli = set(c[0] for c in current)
                missing = [m for m in moduli if m not in current_moduli]
                if not missing or len(current) == 0:
                    continue
                m_add = random.choice(missing)
                m_remove = random.choice([c[0] for c in current])
                new_classes = [c for c in current if c[0] != m_remove]
                r = find_valid_residue(m_add, new_classes)
                if r is None:
                    continue
                new_classes.append([m_add, r])
                delta = 1.0/m_add - 1.0/m_remove
            
            new_score = current_score + delta
            
            if new_score > current_score or random.random() < math.exp(delta / temperature):
                current = new_classes
                current_score = new_score
                if current_score > best_score:
                    best = current[:]
                    best_score = current_score
            
            temperature *= 0.9995
        
        return best, best_score
    
    # Run multiple rounds
    for round_num in range(6):
        if time.time() > deadline - 15.0:
            break
        
        # Hill climb polish
        polished, pol_score = hill_climb(best_classes, time_limit=4.0)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
        
        # SA from current best
        sa_result, sa_score = sa_polish(best_classes, temperature=0.5, steps=4000)
        if sa_score > best_score:
            best_classes = sa_result
            best_score = sa_score
        
        # Final polish
        polished, pol_score = hill_climb(best_classes, time_limit=4.0)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    # Final high-precision polish: try all residues for each modulus
    if time.time() < deadline - 5.0:
        final_classes = best_classes[:]
        final_score = best_score
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(final_classes)):
                m, r = final_classes[i]
                for r_new in range(m):
                    if r_new == r:
                        continue
                    new_classes = final_classes[:]
                    new_classes[i] = [m, r_new]
                    if is_valid_classes(new_classes):
                        # Try to add any missing high-value modulus
                        current_moduli = set(c[0] for c in new_classes)
                        missing = [mm for mm in moduli if mm not in current_moduli]
                        for mm in sorted(missing, key=lambda x: -1.0/x):
                            if 1.0/mm < 0.005:
                                break
                            rr = find_valid_residue(mm, new_classes)
                            if rr is not None:
                                candidate = new_classes + [[mm, rr]]
                                candidate_score = calc_score(candidate)
                                if candidate_score > final_score + 1e-12:
                                    final_classes = candidate
                                    final_score = candidate_score
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
        if final_score > best_score:
            best_classes = final_classes
            best_score = final_score
    
    return best_classes
# EVOLVE-BLOCK-END
