"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from fractions import Fraction

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction
    2. Simulated annealing with 2-swap neighborhood to escape local optima
    3. Polish with hill climbing
    """
    deadline = time.time() + 145.0
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
    
    def calc_score(classes):
        """Calculate the sum of 1/m as a float."""
        return sum(1.0 / m for m, r in classes)
    
    def try_add(classes, m):
        """Try to add modulus m with a valid residue. Returns new classes or None."""
        r = find_valid_residue(m, classes)
        if r is not None:
            return classes + [[m, r]]
        return None
    
    # Initial greedy construction
    moduli = list(range(20, 201))
    classes = []
    for m in moduli:
        if time.time() > deadline - 10.0:
            break
        new_classes = try_add(classes, m)
        if new_classes is not None:
            classes = new_classes
    
    best_classes = classes[:]
    best_score = calc_score(classes)
    
    def hill_climb(classes, time_limit=5.0):
        """Hill climbing polish."""
        improved = True
        start = time.time()
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in moduli if m not in current_moduli]
            
            # Try adding
            for m in missing:
                if time.time() - start > time_limit:
                    break
                new_classes = try_add(classes, m)
                if new_classes is not None:
                    new_score = calc_score(new_classes)
                    if new_score > calc_score(classes) + 1e-12:
                        classes = new_classes
                        improved = True
            
            # Try swapping
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in moduli if m not in current_moduli]
            for m_add in missing:
                if time.time() - start > time_limit:
                    break
                for m_remove in current_moduli:
                    if 1.0/m_add <= 1.0/m_remove:
                        continue
                    new_classes = [c for c in classes if c[0] != m_remove]
                    r = find_valid_residue(m_add, new_classes)
                    if r is not None:
                        new_classes.append([m_add, r])
                        new_score = calc_score(new_classes)
                        if new_score > calc_score(classes) + 1e-12:
                            classes = new_classes
                            improved = True
                            break
        return classes
    
    # Simulated annealing
    def sa_polish(classes, temperature=1.0, steps=5000):
        """Simulated annealing with 2-swap moves."""
        current = classes[:]
        current_score = calc_score(current)
        best = classes[:]
        best_score = current_score
        
        for step in range(steps):
            if time.time() > deadline - 10.0:
                break
            
            # Generate a random move
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
                # Check validity
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
                delta = 0.0  # Same moduli, same score
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
                # 2-swap: swap one modulus for another
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
            
            # Acceptance criterion
            if new_score > current_score or random.random() < math.exp(delta / temperature):
                current = new_classes
                current_score = new_score
                if current_score > best_score:
                    best = current[:]
                    best_score = current_score
            
            # Cool down
            temperature *= 0.9995
        
        return best, best_score
    
    # Run multiple SA rounds
    for round_num in range(5):
        if time.time() > deadline - 15.0:
            break
        
        # Polish current best
        polished = hill_climb(best_classes, time_limit=3.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
        
        # SA from current best
        sa_result, sa_score = sa_polish(best_classes, temperature=0.5, steps=3000)
        if sa_score > best_score:
            best_classes = sa_result
            best_score = sa_score
        
        # Final polish
        polished = hill_climb(best_classes, time_limit=3.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    # Final hill climb
    if time.time() < deadline - 5.0:
        polished = hill_climb(best_classes, time_limit=5.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    return best_classes
# EVOLVE-BLOCK-END
