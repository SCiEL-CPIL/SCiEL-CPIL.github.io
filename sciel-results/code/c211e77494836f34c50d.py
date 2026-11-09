import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe
    
    # State: list of (exponent, coefficient)
    # We need exactly 12 terms.
    # Exponents in [0, 40]
    # Coefficients in [-64, 64] \ {0}
    
    def count_nonzeros(sq_dict):
        return sum(1 for v in sq_dict.values() if v != 0)

    def calculate_square(terms):
        # terms is list of (exp, coeff)
        sq = {}
        for i in range(len(terms)):
            e_i, c_i = terms[i]
            # Diagonal
            key = 2 * e_i
            sq[key] = sq.get(key, 0) + c_i * c_i
            # Off-diagonal
            for j in range(i + 1, len(terms)):
                e_j, c_j = terms[j]
                key = e_i + e_j
                sq[key] = sq.get(key, 0) + 2 * c_i * c_j
        return sq

    def update_square(sq, old_term, new_term, other_terms):
        # Remove old term contributions
        e_old, c_old = old_term
        for e_o, c_o in other_terms:
            key = e_old + e_o
            sq[key] = sq.get(key, 0) - 2 * c_old * c_o
        key_diag = 2 * e_old
        sq[key_diag] = sq.get(key_diag, 0) - c_old * c_old
        
        # Add new term contributions
        e_new, c_new = new_term
        for e_o, c_o in other_terms:
            key = e_new + e_o
            sq[key] = sq.get(key, 0) + 2 * c_new * c_o
        key_diag_new = 2 * e_new
        sq[key_diag_new] = sq.get(key_diag_new, 0) + c_new * c_new
        
        # Clean up zeros to keep dict small (optional, but count_nonzeros iterates)
        # Actually, keeping zeros in dict is fine, count_nonzeros checks v != 0
        return sq

    # Initial population
    def random_terms():
        exponents = random.sample(range(41), 12)
        coefficients = [random.choice([x for x in range(-64, 65) if x != 0]) for _ in range(12)]
        # Sort by exponent for consistency? Not strictly necessary if we track indices, 
        # but helps if we want to merge same exponents (which we don't allow, distinct exponents)
        return list(zip(exponents, coefficients))

    best_terms = random_terms()
    best_sq = calculate_square(best_terms)
    best_score = count_nonzeros(best_sq)
    
    current_terms = best_terms[:]
    current_sq = best_sq
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min # or restart
            
        # Generate neighbor
        neighbor_terms = current_terms[:]
        move_type = random.randint(0, 3)
        
        if move_type == 0:
            # Change exponent
            idx = random.randint(0, 11)
            old_e, c = neighbor_terms[idx]
            new_e = random.randint(0, 40)
            # Ensure distinct exponents? 
            # If new_e collides with another exponent, we have < 12 terms in P(x) effectively, 
            # but the problem says "exactly 12 nonzero terms". 
            # Usually this implies 12 distinct monomials. If two have same exponent, they merge into 1 term.
            # So we must ensure exponents are distinct.
            other_exps = [e for i, (e, _) in enumerate(neighbor_terms) if i != idx]
            if new_e in other_exps:
                continue
            neighbor_terms[idx] = (new_e, c)
            
            # Update square
            old_term = current_terms[idx]
            new_term = neighbor_terms[idx]
            other_terms = [t for i, t in enumerate(current_terms) if i != idx]
            
            # We need to be careful: update_square assumes sq is current_sq.
            # We should work on a copy or revert.
            # Since sq is a dict, we can copy it.
            new_sq = dict(current_sq)
            new_sq = update_square(new_sq, old_term, new_term, other_terms)
            
        elif move_type == 1:
            # Change coefficient
            idx = random.randint(0, 11)
            e, old_c = neighbor_terms[idx]
            new_c = random.choice([x for x in range(-64, 65) if x != 0 and x != old_c])
            neighbor_terms[idx] = (e, new_c)
            
            old_term = current_terms[idx]
            new_term = neighbor_terms[idx]
            other_terms = [t for i, t in enumerate(current_terms) if i != idx]
            
            new_sq = dict(current_sq)
            new_sq = update_square(new_sq, old_term, new_term, other_terms)
            
        elif move_type == 2:
            # Swap exponents of two terms
            i, j = random.sample(range(12), 2)
            e_i, c_i = current_terms[i]
            e_j, c_j = current_terms[j]
            
            # Swap
            neighbor_terms[i] = (e_j, c_i)
            neighbor_terms[j] = (e_i, c_j)
            
            # Update square
            # This is more complex. 
            # Remove i and j, add new i and j.
            other_terms = [t for k, t in enumerate(current_terms) if k != i and k != j]
            
            new_sq = dict(current_sq)
            # Remove i
            new_sq = update_square(new_sq, current_terms[i], (0,0), other_terms + [current_terms[j]]) # Wait, update_square logic is for 1 term change.
            # Let's just recalculate for swaps to avoid bugs, or implement multi-term update.
            # Given 12 terms, recalc is 144 ops. Very fast.
            # Let's stick to recalc for swaps to be safe.
            new_sq = calculate_square(neighbor_terms)
            
        else:
            # Swap coefficients
            i, j = random.sample(range(12), 2)
            e_i, c_i = current_terms[i]
            e_j, c_j = current_terms[j]
            
            neighbor_terms[i] = (e_i, c_j)
            neighbor_terms[j] = (e_j, c_i)
            
            new_sq = calculate_square(neighbor_terms)

        new_score = count_nonzeros(new_sq)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_terms = neighbor_terms
            current_sq = new_sq
            current_score = new_score
            
            if current_score < best_score:
                best_terms = current_terms[:]
                best_sq = current_sq
                best_score = current_score
                
        T *= cooling_rate
        
    # Return best found
    # Convert to list format
    # Length <= 41
    # Ascending degree
    # Nonzero final coefficient
    
    # Sort by exponent
    best_terms.sort(key=lambda x: x[0])
    
    # Determine length
    max_exp = best_terms[-1][0]
    length = max_exp + 1
    
    # Ensure length <= 41
    if length > 41:
        # Fallback to seed
        return [1] * 12
    
    coeff_list = [0] * length
    for e, c in best_terms:
        coeff_list[e] = c
        
    # Check constraints
    # Exactly 12 nonzero terms
    if sum(1 for c in coeff_list if c != 0) != 12:
        return [1] * 12
        
    # Nonzero final coefficient
    if coeff_list[-1] == 0:
        return [1] * 12
        
    return coeff_list

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    def count_nonzeros(sq_dict):
        return sum(1 for v in sq_dict.values() if v != 0)

    def calculate_square(terms):
        sq = {}
        for i in range(len(terms)):
            e_i, c_i = terms[i]
            key = 2 * e_i
            sq[key] = sq.get(key, 0) + c_i * c_i
            for j in range(i + 1, len(terms)):
                e_j, c_j = terms[j]
                key = e_i + e_j
                sq[key] = sq.get(key, 0) + 2 * c_i * c_j
        return sq

    def update_square(sq, old_term, new_term, other_terms):
        e_old, c_old = old_term
        for e_o, c_o in other_terms:
            key = e_old + e_o
            sq[key] = sq.get(key, 0) - 2 * c_old * c_o
        key_diag = 2 * e_old
        sq[key_diag] = sq.get(key_diag, 0) - c_old * c_old
        
        e_new, c_new = new_term
        for e_o, c_o in other_terms:
            key = e_new + e_o
            sq[key] = sq.get(key, 0) + 2 * c_new * c_o
        key_diag_new = 2 * e_new
        sq[key_diag_new] = sq.get(key_diag_new, 0) + c_new * c_new
        
        return sq

    def random_terms():
        exponents = random.sample(range(41), 12)
        coefficients = [random.choice([x for x in range(-64, 65) if x != 0]) for _ in range(12)]
        return list(zip(exponents, coefficients))

    best_terms = random_terms()
    best_sq = calculate_square(best_terms)
    best_score = count_nonzeros(best_sq)
    
    current_terms = best_terms[:]
    current_sq = best_sq
    current_score = best_score
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        neighbor_terms = current_terms[:]
        move_type = random.randint(0, 3)
        
        if move_type == 0:
            idx = random.randint(0, 11)
            old_e, c = neighbor_terms[idx]
            new_e = random.randint(0, 40)
            other_exps = [e for i, (e, _) in enumerate(neighbor_terms) if i != idx]
            if new_e in other_exps:
                continue
            neighbor_terms[idx] = (new_e, c)
            
            old_term = current_terms[idx]
            new_term = neighbor_terms[idx]
            other_terms = [t for i, t in enumerate(current_terms) if i != idx]
            
            new_sq = dict(current_sq)
            new_sq = update_square(new_sq, old_term, new_term, other_terms)
            
        elif move_type == 1:
            idx = random.randint(0, 11)
            e, old_c = neighbor_terms[idx]
            new_c = random.choice([x for x in range(-64, 65) if x != 0 and x != old_c])
            neighbor_terms[idx] = (e, new_c)
            
            old_term = current_terms[idx]
            new_term = neighbor_terms[idx]
            other_terms = [t for i, t in enumerate(current_terms) if i != idx]
            
            new_sq = dict(current_sq)
            new_sq = update_square(new_sq, old_term, new_term, other_terms)
            
        elif move_type == 2:
            i, j = random.sample(range(12), 2)
            e_i, c_i = current_terms[i]
            e_j, c_j = current_terms[j]
            
            neighbor_terms[i] = (e_j, c_i)
            neighbor_terms[j] = (e_i, c_j)
            
            new_sq = calculate_square(neighbor_terms)
            
        else:
            i, j = random.sample(range(12), 2)
            e_i, c_i = current_terms[i]
            e_j, c_j = current_terms[j]
            
            neighbor_terms[i] = (e_i, c_j)
            neighbor_terms[j] = (e_j, c_i)
            
            new_sq = calculate_square(neighbor_terms)

        new_score = count_nonzeros(new_sq)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_terms = neighbor_terms
            current_sq = new_sq
            current_score = new_score
            
            if current_score < best_score:
                best_terms = current_terms[:]
                best_sq = current_sq
                best_score = current_score
                
        T *= cooling_rate
        
    best_terms.sort(key=lambda x: x[0])
    max_exp = best_terms[-1][0]
    length = max_exp + 1
    
    if length > 41:
        return [1] * 12
        
    coeff_list = [0] * length
    for e, c in best_terms:
        coeff_list[e] = c
        
    if sum(1 for c in coeff_list if c != 0) != 12:
        return [1] * 12
        
    if coeff_list[-1] == 0:
        return [1] * 12
        
    return coeff_list
# EVOLVE-BLOCK-END
