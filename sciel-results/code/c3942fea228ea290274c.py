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
    1. Greedy initial construction (multiple orders)
    2. Efficient local search with bitmask constraint tracking
    3. Simulated annealing with 2-residue swap neighborhood
    4. Final high-precision polish
    """
    deadline = time.time() + 145.0
    random.seed(42)
    
    MODULI = list(range(20, 201))
    N_MODULI = len(MODULI)
    
    # Precompute gcd matrix
    gcd_matrix = [[0] * N_MODULI for _ in range(N_MODULI)]
    for i, m1 in enumerate(MODULI):
        for j, m2 in enumerate(MODULI):
            gcd_matrix[i][j] = gcd(m1, m2)
    
    # Precompute weights
    weights = [1.0 / m for m in MODULI]
    
    def calc_score(classes):
        """Calculate the sum of 1/m as a float."""
        return sum(weights[MODULI.index(m)] for m, r in classes)
    
    def is_valid_classes(classes):
        """Check if all pairs are disjoint."""
        n = len(classes)
        for i in range(n):
            for j in range(i + 1, n):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                idx1 = MODULI.index(m1)
                idx2 = MODULI.index(m2)
                g = gcd_matrix[idx1][idx2]
                if (r1 % g) == (r2 % g):
                    return False
        return True
    
    def find_valid_residue(m, classes):
        """Find a residue r in [0, m) such that [m, r] is disjoint from all classes."""
        m_idx = MODULI.index(m)
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                idx2 = MODULI.index(m2)
                g = gcd_matrix[m_idx][idx2]
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def find_all_valid_residues(m, classes):
        """Find all residues r in [0, m) such that [m, r] is disjoint from all classes."""
        m_idx = MODULI.index(m)
        valid_residues = []
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                idx2 = MODULI.index(m2)
                g = gcd_matrix[m_idx][idx2]
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                valid_residues.append(r)
        return valid_residues
    
    def try_add(classes, m):
        """Try to add modulus m with a valid residue. Returns new classes or None."""
        r = find_valid_residue(m, classes)
        if r is not None:
            return classes + [[m, r]]
        return None
    
    # === Greedy initial construction ===
    def greedy_construct(order):
        """Greedy construction with given modulus order."""
        classes = []
        for m in order:
            if time.time() > deadline - 10.0:
                break
            new_classes = try_add(classes, m)
            if new_classes is not None:
                classes = new_classes
        return classes
    
    # Try multiple greedy orders
    orders = [
        sorted(MODULI, reverse=True),  # Descending
        sorted(MODULI),                # Ascending
        sorted(MODULI, key=lambda m: -1.0/m),  # By weight descending
        list(reversed(sorted(MODULI, key=lambda m: -1.0/m))),  # By weight ascending
    ]
    
    best_classes = []
    best_score = 0.0
    for order in orders:
        classes = greedy_construct(order)
        score = calc_score(classes)
        if score > best_score:
            best_score = score
            best_classes = classes[:]
    
    # === Efficient hill climbing ===
    def hill_climb(classes, time_limit=5.0):
        """Hill climbing with add, swap, and residue reassignment moves."""
        improved = True
        start = time.time()
        current_score = calc_score(classes)
        
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli_set = set(c[0] for c in classes)
            missing = [m for m in MODULI if m not in current_moduli_set]
            
            # Try adding missing moduli (sorted by weight)
            for m in sorted(missing, key=lambda x: -1.0/x):
                if time.time() - start > time_limit:
                    break
                new_classes = try_add(classes, m)
                if new_classes is not None:
                    new_score = current_score + 1.0/m
                    if new_score > current_score + 1e-12:
                        classes = new_classes
                        current_score = new_score
                        improved = True
                        current_moduli_set = set(c[0] for c in classes)
                        missing = [mm for mm in MODULI if mm not in current_moduli_set]
            
            # Try swaps: remove low-weight, add high-weight
            if missing:
                current_moduli_set = set(c[0] for c in classes)
                for m_add in sorted(missing, key=lambda x: -1.0/x):
                    if time.time() - start > time_limit:
                        break
                    for m_remove in sorted(current_moduli_set, key=lambda x: 1.0/x):
                        if 1.0/m_add <= 1.0/m_remove:
                            continue
                        new_classes = [c for c in classes if c[0] != m_remove]
                        r = find_valid_residue(m_add, new_classes)
                        if r is not None:
                            new_classes.append([m_add, r])
                            new_score = current_score - 1.0/m_remove + 1.0/m_add
                            if new_score > current_score + 1e-12:
                                classes = new_classes
                                current_score = new_score
                                improved = True
                                current_moduli_set = set(c[0] for c in classes)
                                missing = [mm for mm in MODULI if mm not in current_moduli_set]
                                break
            
            # Try reassigning residues to enable additions
            if missing and time.time() - start < time_limit:
                for i, (m, r) in enumerate(classes):
                    if time.time() - start > time_limit:
                        break
                    for r_new in range(m):
                        if r_new == r:
                            continue
                        new_classes = classes[:]
                        new_classes[i] = [m, r_new]
                        if is_valid_classes(new_classes):
                            current_moduli_set = set(c[0] for c in new_classes)
                            for mm in sorted(missing, key=lambda x: -1.0/x):
                                if 1.0/mm < 0.004:
                                    break
                                rr = find_valid_residue(mm, new_classes)
                                if rr is not None:
                                    candidate = new_classes + [[mm, rr]]
                                    candidate_score = current_score + 1.0/mm
                                    if candidate_score > current_score + 1e-12:
                                        classes = candidate
                                        current_score = candidate_score
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break
        
        return classes, calc_score(classes)
    
    # === Simulated Annealing with 2-residue swaps ===
    def sa_polish(classes, time_limit=10.0):
        """Simulated annealing with diverse moves including 2-residue swaps."""
        current = classes[:]
        current_score = calc_score(current)
        best = classes[:]
        best_score = current_score
        
        start_temp = 0.3
        end_temp = 0.01
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Adaptive temperature based on progress
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / time_limit)
            temperature = start_temp * (end_temp / start_temp) ** progress
            
            move_type = random.random()
            
            if move_type < 0.25:
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
                    idx_m = MODULI.index(m)
                    idx_m2 = MODULI.index(m2)
                    g = gcd_matrix[idx_m][idx_m2]
                    if (r_new % g) == (r2 % g):
                        valid = False
                        break
                if not valid:
                    continue
                delta = 0.0
            elif move_type < 0.45:
                # Add a missing modulus
                current_moduli_set = set(c[0] for c in current)
                missing = [m for m in MODULI if m not in current_moduli_set]
                if not missing:
                    continue
                m_add = random.choice(missing)
                r = find_valid_residue(m_add, current)
                if r is None:
                    continue
                new_classes = current + [[m_add, r]]
                delta = 1.0 / m_add
            elif move_type < 0.65:
                # Remove a modulus
                if len(current) <= 2:
                    continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                new_classes = [c for c in current if c[0] != m_remove]
                delta = -1.0 / m_remove
            elif move_type < 0.85:
                # 2-swap: remove one, add another
                current_moduli_set = set(c[0] for c in current)
                missing = [m for m in MODULI if m not in current_moduli_set]
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
            else:
                # 2-residue swap: change residues of two classes simultaneously
                if len(current) < 2:
                    continue
                i, j = random.sample(range(len(current)), 2)
                m1, r1 = current[i]
                m2, r2 = current[j]
                r1_new = random.randint(0, m1 - 1)
                r2_new = random.randint(0, m2 - 1)
                if r1_new == r1 and r2_new == r2:
                    continue
                new_classes = current[:]
                new_classes[i] = [m1, r1_new]
                new_classes[j] = [m2, r2_new]
                # Check validity
                valid = True
                idx_m1 = MODULI.index(m1)
                idx_m2 = MODULI.index(m2)
                g12 = gcd_matrix[idx_m1][idx_m2]
                if (r1_new % g12) == (r2_new % g12):
                    valid = False
                if valid:
                    for k in range(len(new_classes)):
                        if k == i or k == j:
                            continue
                        mk, rk = new_classes[k]
                        idx_mk = MODULI.index(mk)
                        g1k = gcd_matrix[idx_m1][idx_mk]
                        if (r1_new % g1k) == (rk % g1k):
                            valid = False
                            break
                        g2k = gcd_matrix[idx_m2][idx_mk]
                        if (r2_new % g2k) == (rk % g2k):
                            valid = False
                            break
                if not valid:
                    continue
                delta = 0.0
            
            new_score = current_score + delta
            
            if new_score > current_score or (delta < 0 and random.random() < math.exp(delta / max(temperature, 1e-10))):
                current = new_classes
                current_score = new_score
                if current_score > best_score:
                    best = current[:]
                    best_score = current_score
        
        return best, best_score
    
    # === Main search loop ===
    for round_num in range(8):
        if time.time() > deadline - 15.0:
            break
        
        # Hill climb
        polished, pol_score = hill_climb(best_classes, time_limit=4.0)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
        
        # SA polish
        sa_result, sa_score = sa_polish(best_classes, time_limit=8.0)
        if sa_score > best_score:
            best_classes = sa_result
            best_score = sa_score
        
        # Another hill climb
        polished, pol_score = hill_climb(best_classes, time_limit=3.0)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    # === Final high-precision polish ===
    if time.time() < deadline - 5.0:
        final_classes = best_classes[:]
        final_score = best_score
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            current_moduli_set = set(c[0] for c in final_classes)
            missing = [m for m in MODULI if m not in current_moduli_set]
            
            # Try all residue combinations for pairs of classes
            for i in range(len(final_classes)):
                if time.time() > deadline - 3.0:
                    break
                m1, r1 = final_classes[i]
                for j in range(i + 1, len(final_classes)):
                    if time.time() > deadline - 3.0:
                        break
                    m2, r2 = final_classes[j]
                    for r1_new in range(m1):
                        if r1_new == r1:
                            continue
                        for r2_new in range(m2):
                            if r2_new == r2:
                                continue
                            new_classes = final_classes[:]
                            new_classes[i] = [m1, r1_new]
                            new_classes[j] = [m2, r2_new]
                            if is_valid_classes(new_classes):
                                # Try to add missing high-value moduli
                                current_moduli_set = set(c[0] for c in new_classes)
                                for mm in sorted(missing, key=lambda x: -1.0/x):
                                    if 1.0/mm < 0.003:
                                        break
                                    rr = find_valid_residue(mm, new_classes)
                                    if rr is not None:
                                        candidate = new_classes + [[mm, rr]]
                                        candidate_score = final_score + 1.0/mm
                                        if candidate_score > final_score + 1e-12:
                                            final_classes = candidate
                                            final_score = candidate_score
                                            improved = True
                                            break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # Single residue reassignment
            if not improved:
                for i in range(len(final_classes)):
                    if time.time() > deadline - 3.0:
                        break
                    m, r = final_classes[i]
                    for r_new in range(m):
                        if r_new == r:
                            continue
                        new_classes = final_classes[:]
                        new_classes[i] = [m, r_new]
                        if is_valid_classes(new_classes):
                            current_moduli_set = set(c[0] for c in new_classes)
                            for mm in sorted(missing, key=lambda x: -1.0/x):
                                if 1.0/mm < 0.003:
                                    break
                                rr = find_valid_residue(mm, new_classes)
                                if rr is not None:
                                    candidate = new_classes + [[mm, rr]]
                                    candidate_score = final_score + 1.0/mm
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
