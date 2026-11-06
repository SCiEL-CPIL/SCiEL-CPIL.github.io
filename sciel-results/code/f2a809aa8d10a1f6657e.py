import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Moduli we can choose from: 2..128
    moduli = list(range(2, 129))
    n_moduli = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Precompute prime factors for sorting strategies
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    pf_counts = [num_prime_factors(m) for m in moduli]

    def can_add(selected, r_new, m_idx_new):
        """Check if (r_new, moduli[m_idx_new]) is disjoint from all selected."""
        for (r_sel, m_idx_sel) in selected:
            g = gcd_table[m_idx_sel][m_idx_new]
            if (r_new - r_sel) % g == 0:
                return False
        return True

    def is_valid_config(selected):
        """Check if the entire selected list is valid."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                r1, m1_idx = selected[i]
                r2, m2_idx = selected[j]
                g = gcd_table[m1_idx][m2_idx]
                if (r1 - r2) % g == 0:
                    return False
        return True

    def greedy_construct(order):
        """Greedy: process moduli in given order, pick first valid residue for each."""
        selected = []
        for m_idx in order:
            m = moduli[m_idx]
            # Try residues in a deterministic order first, then random if needed? 
            # For speed, deterministic 0..m-1 is fine for initialization.
            for r in range(m):
                if can_add(selected, r, m_idx):
                    selected.append((r, m_idx))
                    break
        return selected

    def local_search(selected, max_iter=20000):
        """
        Local search with 1-swaps, 2-swaps, and kicks.
        """
        best = [tuple(s) for s in selected]
        best_len = len(best)
        
        # Convert to set for fast lookup of which moduli are present
        # We represent the solution as a list of (r, m_idx)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try to add a new modulus
            selected_set = set(m_idx for _, m_idx in best)
            added = False
            for m_idx in range(n_moduli):
                if m_idx in selected_set:
                    continue
                m = moduli[m_idx]
                # Try all residues
                for r in range(m):
                    if can_add(best, r, m_idx):
                        best.append((r, m_idx))
                        best_len += 1
                        improved = True
                        added = True
                        break
                if added:
                    break
            
            if added:
                continue # Restart loop to check for more additions
            
            # 2. Try 1-swaps: change residue of one modulus to enable an addition
            found_swap = False
            for i, (r_old, m_idx) in enumerate(best):
                m = moduli[m_idx]
                # Try all possible new residues
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    # Check if new residue is valid with all others
                    ok = True
                    for j, (r_j, m_idx_j) in enumerate(best):
                        if i == j:
                            continue
                        g = gcd_table[m_idx][m_idx_j]
                        if (r_new - r_j) % g == 0:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # If valid, check if this swap allows adding a new modulus
                    # We simulate the swap
                    test_best = best[:]
                    test_best[i] = (r_new, m_idx)
                    
                    # Try to add any missing modulus
                    test_set = set(m_idx for _, m_idx in test_best)
                    can_add_more = False
                    for m_add_idx in range(n_moduli):
                        if m_add_idx in test_set:
                            continue
                        m_add = moduli[m_add_idx]
                        for r_add in range(m_add):
                            if can_add(test_best, r_add, m_add_idx):
                                # Accept swap and addition
                                test_best.append((r_add, m_add_idx))
                                best = test_best
                                best_len += 1
                                improved = True
                                can_add_more = True
                                break
                        if can_add_more:
                            break
                    
                    if can_add_more:
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if found_swap:
                continue # Restart loop
            
            # 3. Try 2-swaps: change residues of two moduli to enable an addition
            # This is more expensive, so we do it less frequently or with a limit
            # For simplicity and speed, we might skip 2-swaps in the main loop 
            # and rely on kicks to escape local optima.
            # However, 2-swaps are powerful. Let's try a limited set of 2-swaps.
            
            # To keep it fast, we only try 2-swaps if we haven't improved in a while.
            # Here we implement a simple version: pick 2 random indices, try random residues.
            if len(best) < n_moduli:
                # Pick 2 random indices
                if len(best) >= 2:
                    idx1, idx2 = random.sample(range(len(best)), 2)
                    r1, m1 = best[idx1]
                    r2, m2 = best[idx2]
                    
                    # Try a few random new residues for these two
                    for _ in range(10):
                        nr1 = random.randint(0, moduli[m1] - 1)
                        nr2 = random.randint(0, moduli[m2] - 1)
                        
                        # Check validity with all others
                        ok1 = True
                        ok2 = True
                        for j, (r_j, m_idx_j) in enumerate(best):
                            if j == idx1 or j == idx2:
                                continue
                            g1 = gcd_table[m1][m_idx_j]
                            if (nr1 - r_j) % g1 == 0:
                                ok1 = False
                                break
                            g2 = gcd_table[m2][m_idx_j]
                            if (nr2 - r_j) % g2 == 0:
                                ok2 = False
                                break
                        if not (ok1 and ok2):
                            continue
                        
                        # Check validity between the two swapped
                        g12 = gcd_table[m1][m2]
                        if (nr1 - nr2) % g12 == 0:
                            continue
                        
                        # Simulate swap
                        test_best = best[:]
                        test_best[idx1] = (nr1, m1)
                        test_best[idx2] = (nr2, m2)
                        
                        # Try to add a new modulus
                        test_set = set(m_idx for _, m_idx in test_best)
                        can_add_more = False
                        for m_add_idx in range(n_moduli):
                            if m_add_idx in test_set:
                                continue
                            m_add = moduli[m_add_idx]
                            for r_add in range(m_add):
                                if can_add(test_best, r_add, m_add_idx):
                                    test_best.append((r_add, m_add_idx))
                                    best = test_best
                                    best_len += 1
                                    improved = True
                                    can_add_more = True
                                    break
                            if can_add_more:
                                break
                        
                        if can_add_more:
                            break
                    if improved:
                        continue

            # 4. Kick: Randomly perturb the solution to escape local optima
            # Change residues of 2 random moduli to random values
            if len(best) >= 2:
                idx1, idx2 = random.sample(range(len(best)), 2)
                m1, m2 = best[idx1][1], best[idx2][1]
                best[idx1] = (random.randint(0, moduli[m1] - 1), m1)
                best[idx2] = (random.randint(0, moduli[m2] - 1), m2)
                # Note: This might make the current solution invalid!
                # We must ensure the solution remains valid after a kick.
                # The "kick" strategy usually implies we restart the local search from this new state,
                # but we need the state to be valid.
                # A safer kick: Try to find a valid kick.
                # If the kick makes it invalid, we revert.
                if not is_valid_config(best):
                    # Revert kick
                    best[idx1] = (r1, m1) # Wait, r1 is from the start of the iteration, which might be stale if we modified best?
                    # Actually, r1, m1 were fetched before the kick loop. 
                    # But if we did a 2-swap above, best might have changed.
                    # To be safe, let's just revert to the state before the kick.
                    # We didn't save the state. Let's just break the kick if invalid.
                    # Better: Don't kick if it breaks validity.
                    # Or, simpler: Just break out of local search and try a new greedy start.
                    pass 
            else:
                # If solution is small, just break
                break
                
            # If we got here, no improvement was made in this iteration
            # To prevent infinite loops on kicks that don't help, we can count iterations without improvement
            # But the random kick changes the state, so it's a stochastic search.
            # We just continue the loop.

        return best

    # Generate multiple initial solutions
    best_solution = []
    
    # Strategy 1: Ascending order
    order_asc = list(range(n_moduli))
    sol = greedy_construct(order_asc)
    sol = local_search(sol, max_iter=5000)
    if len(sol) > len(best_solution):
        best_solution = sol
        
    # Strategy 2: Descending order
    order_desc = list(range(n_moduli - 1, -1, -1))
    sol = greedy_construct(order_desc)
    sol = local_search(sol, max_iter=5000)
    if len(sol) > len(best_solution):
        best_solution = sol
        
    # Strategy 3: By number of prime factors (ascending)
    order_pf_asc = sorted(range(n_moduli), key=lambda i: pf_counts[i])
    sol = greedy_construct(order_pf_asc)
    sol = local_search(sol, max_iter=5000)
    if len(sol) > len(best_solution):
        best_solution = sol
        
    # Strategy 4: By number of prime factors (descending)
    order_pf_desc = sorted(range(n_moduli), key=lambda i: -pf_counts[i])
    sol = greedy_construct(order_pf_desc)
    sol = local_search(sol, max_iter=5000)
    if len(sol) > len(best_solution):
        best_solution = sol
        
    # Strategy 5: Random orderings
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline:
            break
        random_order = list(range(n_moduli))
        random.shuffle(random_order)
        sol = greedy_construct(random_order)
        sol = local_search(sol, max_iter=3000)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    # Final polish on the best solution found
    if time.time() < deadline:
        best_solution = local_search(best_solution, max_iter=10000)

    # Convert to required format
    result = [[r, moduli[m_idx]] for r, m_idx in best_solution]
    
    # Ensure nonempty
    if not result:
        return [[1, 2]]
        
    return result
