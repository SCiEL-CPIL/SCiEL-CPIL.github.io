from math import gcd
import time
import random
from fractions import Fraction

def _is_valid(classes):
    """Check pairwise disjointness."""
    n = len(classes)
    for i in range(n):
        m1, r1 = classes[i]
        for j in range(i + 1, n):
            m2, r2 = classes[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                return False
    return True

def _score(classes):
    """Sum of 1/m as float."""
    return sum(1.0 / m for m, r in classes)

def _score_frac(classes):
    """Sum of 1/m as Fraction."""
    return sum(Fraction(1, m) for m, r in classes)

def _greedy_construct(order, deadline, gcd_cache):
    """
    Greedily add moduli in given order. For each modulus, find a residue
    that is disjoint with all previously selected.
    """
    selected = []
    for m in order:
        if time.time() > deadline:
            break
        # Find a residue r in [0, m) such that for all (m_i, r_i) in selected:
        # r % gcd(m, m_i) != r_i % gcd(m, m_i)
        found = False
        for r in range(m):
            valid = True
            for (m_i, r_i) in selected:
                g = gcd_cache[m][m_i]
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                selected.append([m, r])
                found = True
                break
        # If no residue found, skip this modulus
    return selected

def _local_search(classes, deadline, gcd_cache):
    """
    Local search: try to add unselected moduli by adjusting residues of
    existing ones, and try swaps to free up capacity.
    """
    classes = [list(c) for c in classes]
    selected_mods = set(m for m, r in classes)
    all_mods = list(range(20, 201))
    unselected = [m for m in all_mods if m not in selected_mods]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each unselected modulus
        for m in unselected[:]:
            if time.time() > deadline:
                break
            # Try to find a residue for m that works with current selection
            for r in range(m):
                valid = True
                for (m_i, r_i) in classes:
                    g = gcd_cache[m][m_i]
                    if r % g == r_i % g:
                        valid = False
                        break
                if valid:
                    classes.append([m, r])
                    unselected.remove(m)
                    selected_mods.add(m)
                    improved = True
                    break
        
        # Try swapping residues of existing classes to free capacity
        if not improved:
            # Try removing one class and re-adding two
            for idx in range(len(classes)):
                if time.time() > deadline:
                    break
                m_remove = classes[idx][0]
                saved = classes.pop(idx)
                # Try to add the removed one back with different residue, or add others
                # Try adding unselected moduli
                added_any = False
                for m in unselected[:]:
                    for r in range(m):
                        valid = True
                        for (m_i, r_i) in classes:
                            g = gcd_cache[m][m_i]
                            if r % g == r_i % g:
                                valid = False
                                break
                        if valid:
                            classes.append([m, r])
                            unselected.remove(m)
                            selected_mods.add(m)
                            added_any = True
                            break
                    if added_any:
                        break
                # Also try re-adding the removed one
                if not added_any:
                    for r in range(m_remove):
                        valid = True
                        for (m_i, r_i) in classes:
                            g = gcd_cache[m_remove][m_i]
                            if r % g == r_i % g:
                                valid = False
                                break
                        if valid:
                            classes.append([m_remove, r])
                            added_any = True
                            break
                if not added_any:
                    # Restore
                    classes.insert(idx, saved)
    
    return classes

def _simulated_annealing(classes, deadline, gcd_cache):
    """
    Simulated annealing to escape local optima.
    Moves: 
    1. Add an unselected modulus with a random valid residue.
    2. Remove a random selected modulus.
    3. Change the residue of a random selected modulus to a random valid one.
    4. Swap residues of two selected moduli.
    """
    classes = [list(c) for c in classes]
    selected_mods = set(m for m, r in classes)
    all_mods = list(range(20, 201))
    unselected = [m for m in all_mods if m not in selected_mods]
    
    current_score = _score_frac(classes)
    best_classes = [list(c) for c in classes]
    best_score = current_score
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    
    while time.time() < deadline and T > T_min:
        T *= cooling_rate
        unselected = [m for m in all_mods if m not in selected_mods]
        
        # Choose a random move
        move_type = random.randint(0, 3)
        new_classes = [list(c) for c in classes]
        
        if move_type == 0 and unselected:
            # Add a random unselected modulus
            m = random.choice(unselected)
            # Find all valid residues for m
            valid_residues = []
            for r in range(m):
                valid = True
                for (m_i, r_i) in classes:
                    g = gcd_cache[m][m_i]
                    if r % g == r_i % g:
                        valid = False
                        break
                if valid:
                    valid_residues.append(r)
            if valid_residues:
                r = random.choice(valid_residues)
                new_classes.append([m, r])
                selected_mods.add(m)
        
        elif move_type == 1 and classes:
            # Remove a random selected modulus
            idx = random.randint(0, len(classes) - 1)
            m = classes[idx][0]
            new_classes.pop(idx)
            selected_mods.discard(m)
        
        elif move_type == 2 and classes:
            # Change residue of a random selected modulus
            idx = random.randint(0, len(classes) - 1)
            m, r = classes[idx]
            # Find valid residues for m excluding current r
            valid_residues = []
            for r_new in range(m):
                if r_new == r:
                    continue
                valid = True
                for j, (m_i, r_i) in enumerate(classes):
                    if j == idx:
                        continue
                    g = gcd_cache[m][m_i]
                    if r_new % g == r_i % g:
                        valid = False
                        break
                if valid:
                    valid_residues.append(r_new)
            if valid_residues:
                new_classes[idx][1] = random.choice(valid_residues)
        
        elif move_type == 3 and len(classes) >= 2:
            # Swap residues of two selected moduli
            i, j = random.sample(range(len(classes)), 2)
            m1, r1 = classes[i]
            m2, r2 = classes[j]
            # Check if swapping is valid
            # We need r2 to be valid for m1 (against all others except m2)
            # and r1 to be valid for m2 (against all others except m1)
            valid_swap = True
            for k, (m_k, r_k) in enumerate(classes):
                if k == i or k == j:
                    continue
                g1 = gcd_cache[m1][m_k]
                if r2 % g1 == r_k % g1:
                    valid_swap = False
                    break
            if valid_swap:
                for k, (m_k, r_k) in enumerate(classes):
                    if k == i or k == j:
                        continue
                    g2 = gcd_cache[m2][m_k]
                    if r1 % g2 == r_k % g2:
                        valid_swap = False
                        break
            # Also check m1 and m2 against each other
            g12 = gcd_cache[m1][m2]
            if r2 % g12 == r1 % g12:
                valid_swap = False
            
            if valid_swap:
                new_classes[i][1] = r2
                new_classes[j][1] = r1
        
        # Evaluate new state
        if _is_valid(new_classes):
            new_score = _score_frac(new_classes)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                classes = new_classes
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_classes = [list(c) for c in classes]
        
        # Occasionally try to add missing moduli greedily to the current state
        if random.random() < 0.1:
            classes = _local_search(classes, min(time.time() + 0.1, deadline), gcd_cache)
            current_score = _score_frac(classes)
            if current_score > best_score:
                best_score = current_score
                best_classes = [list(c) for c in classes]
    
    return best_classes

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    all_moduli = list(range(20, 201))
    
    # Pre-compute gcds for speed
    gcd_cache = {}
    for m1 in all_moduli:
        gcd_cache[m1] = {}
        for m2 in all_moduli:
            gcd_cache[m1][m2] = gcd(m1, m2)
    
    best_classes = None
    best_score = 0.0
    
    # Strategy 1: Greedy by ascending m
    order1 = list(range(20, 201))
    classes1 = _greedy_construct(order1, deadline, gcd_cache)
    classes1 = _local_search(classes1, deadline, gcd_cache)
    if _is_valid(classes1):
        s1 = _score(classes1)
        if s1 > best_score:
            best_score = s1
            best_classes = classes1
    
    # Strategy 2: Use pairwise coprime moduli
    coprime_mods = []
    for m in range(20, 201):
        is_coprime = True
        for cm in coprime_mods:
            if gcd(m, cm) != 1:
                is_coprime = False
                break
        if is_coprime:
            coprime_mods.append(m)
    
    classes2 = [[m, 0] for m in coprime_mods]
    if _is_valid(classes2):
        s2 = _score(classes2)
        if s2 > best_score:
            best_score = s2
            best_classes = classes2
    
    # Strategy 3: Start with primes, then greedily add composites
    primes_in_range = [m for m in range(20, 201) if all(m % p != 0 for p in range(2, int(m**0.5)+1))]
    classes4 = [[m, 0] for m in primes_in_range]
    for m in range(20, 201):
        if m in primes_in_range:
            continue
        if time.time() > deadline:
            break
        for r in range(m):
            valid = True
            for (m_i, r_i) in classes4:
                g = gcd_cache[m][m_i]
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                classes4.append([m, r])
                break
    classes4 = _local_search(classes4, deadline, gcd_cache)
    if _is_valid(classes4):
        s4 = _score(classes4)
        if s4 > best_score:
            best_score = s4
            best_classes = classes4
    
    # Strategy 4: Simulated Annealing
    if time.time() < deadline - 10:
        random.seed(42)
        # Start SA from the best found so far
        if best_classes:
            sa_classes = _simulated_annealing(best_classes, deadline, gcd_cache)
            if _is_valid(sa_classes):
                s_sa = _score(sa_classes)
                if s_sa > best_score:
                    best_score = s_sa
                    best_classes = sa_classes
        else:
            # Start from a random greedy construction
            order_rand = list(range(20, 201))
            random.shuffle(order_rand)
            rand_classes = _greedy_construct(order_rand, deadline, gcd_cache)
            sa_classes = _simulated_annealing(rand_classes, deadline, gcd_cache)
            if _is_valid(sa_classes):
                s_sa = _score(sa_classes)
                if s_sa > best_score:
                    best_score = s_sa
                    best_classes = sa_classes
    
    # Final local search polish
    if best_classes and time.time() < deadline - 5:
        final_classes = _local_search(best_classes, deadline, gcd_cache)
        if _is_valid(final_classes):
            s_final = _score(final_classes)
            if s_final > best_score:
                best_score = s_final
                best_classes = final_classes
    
    if best_classes is None:
        # Fallback
        best_classes = [[20, 0]]
    
    return best_classes
