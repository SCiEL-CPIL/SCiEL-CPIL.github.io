import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n_mod = len(moduli)  # 127

    # Precompute gcd for all pairs of moduli
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    # Precompute gcd matrix
    gcd_mat = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict masks
    # conflict[i][j][r_i] = bitmask of residues r_j that conflict with r_i
    conflict = [[None] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        mi = moduli[i]
        for j in range(n_mod):
            if i == j:
                continue
            mj = moduli[j]
            g = gcd_mat[i][j]
            masks = [0] * mi
            for r_i in range(mi):
                target = r_i % g
                mask = 0
                for r_j in range(mj):
                    if r_j % g == target:
                        mask |= (1 << r_j)
                masks[r_i] = mask
            conflict[i][j] = masks

    def popcount(x):
        return bin(x).count('1')

    def greedy_solve(order):
        """Greedy: process moduli in given order, pick residue that blocks fewest future options."""
        n = len(order)
        avail = [(1 << moduli[i]) - 1 for i in range(n_mod)]
        chosen = {}  # idx -> residue
        
        # Position map for quick "future" lookup
        pos_in_order = {idx: pos for pos, idx in enumerate(order)}
        
        for pos, idx in enumerate(order):
            if time.time() > deadline - 2.0:
                break
            mi = moduli[idx]
            if avail[idx] == 0:
                continue
            
            # Find best residue: minimize number of future residues blocked
            best_r = -1
            best_cost = float('inf')
            
            # Future indices are those with pos_in_order > pos
            # We can just iterate through remaining order
            future = order[pos + 1:]
            
            for r in range(mi):
                if not (avail[idx] & (1 << r)):
                    continue
                cost = 0
                for j in future:
                    if avail[j] == 0:
                        continue
                    mask = conflict[idx][j][r]
                    cost += popcount(avail[j] & mask)
                if cost < best_cost:
                    best_cost = cost
                    best_r = r
            
            if best_r >= 0:
                chosen[idx] = best_r
                # Update available masks for future moduli
                for j in future:
                    mask = conflict[idx][j][best_r]
                    avail[j] &= ~mask
        
        return chosen

    def local_search(chosen, max_iter=50):
        """Try to add back moduli that were skipped by reassigning residues."""
        if time.time() > deadline - 5.0:
            return chosen
        
        chosen = dict(chosen)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            
            # Find moduli not in chosen
            missing = [i for i in range(n_mod) if i not in chosen]
            if not missing:
                break
            
            # Try adding each missing modulus
            for idx in missing:
                if time.time() > deadline - 5.0:
                    break
                mi = moduli[idx]
                
                # Check if any residue is available
                # A residue r is valid if for all chosen j, r % gcd(mi, mj) != chosen[j] % gcd(mi, mj)
                valid_residues = []
                for r in range(mi):
                    ok = True
                    for j, rj in chosen.items():
                        g = gcd_mat[idx][j]
                        if r % g == rj % g:
                            ok = False
                            break
                    if ok:
                        valid_residues.append(r)
                
                if not valid_residues:
                    continue
                
                # Try each valid residue, pick the one that blocks fewest other missing moduli
                best_r = None
                best_cost = float('inf')
                
                for r in valid_residues:
                    cost = 0
                    for j in missing:
                        if j == idx:
                            continue
                        mj = moduli[j]
                        g = gcd_mat[idx][j]
                        # How many residues of j would be blocked?
                        target = r % g
                        blocked_count = 0
                        for rj in range(mj):
                            if rj % g == target:
                                blocked_count += 1
                        cost += blocked_count
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_r = r
                
                if best_r is not None:
                    chosen[idx] = best_r
                    improved = True
                    break  # Restart the loop after adding one
        
        return chosen

    def is_valid_solution(chosen):
        """Check if the solution is strictly valid."""
        items = list(chosen.items())
        for i in range(len(items)):
            idx_i, r_i = items[i]
            mi = moduli[idx_i]
            for j in range(i + 1, len(items)):
                idx_j, r_j = items[j]
                mj = moduli[idx_j]
                g = gcd_mat[idx_i][idx_j]
                if r_i % g == r_j % g:
                    return False
        return True

    best_sol = {}
    best_len = 0

    # Deterministic orderings
    orderings = [
        list(range(n_mod)),
        list(range(n_mod - 1, -1, -1)),
        sorted(range(n_mod), key=lambda i: moduli[i]),
        sorted(range(n_mod), key=lambda i: -moduli[i]),
    ]
    
    # Sort by number of prime factors (more factors = more constraints)
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

    orderings.append(sorted(range(n_mod), key=lambda i: num_prime_factors(moduli[i])))
    orderings.append(sorted(range(n_mod), key=lambda i: -num_prime_factors(moduli[i])))
    
    # Sort by totient (phi) - smaller phi means more constraints
    def totient(n):
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    orderings.append(sorted(range(n_mod), key=lambda i: totient(moduli[i])))
    orderings.append(sorted(range(n_mod), key=lambda i: -totient(moduli[i])))

    for order in orderings:
        if time.time() > deadline - 10.0:
            break
        chosen = greedy_solve(order)
        chosen = local_search(chosen)
        if is_valid_solution(chosen):
            if len(chosen) > best_len:
                best_len = len(chosen)
                best_sol = dict(chosen)

    # Randomized restarts
    random.seed(42)
    trials = 0
    while time.time() < deadline - 10.0 and trials < 300:
        trials += 1
        order = list(range(n_mod))
        random.shuffle(order)
        chosen = greedy_solve(order)
        chosen = local_search(chosen)
        if is_valid_solution(chosen):
            if len(chosen) > best_len:
                best_len = len(chosen)
                best_sol = dict(chosen)

    # Final local search on best solution
    if time.time() < deadline - 5.0:
        best_sol = local_search(best_sol, max_iter=100)
        if is_valid_solution(best_sol):
            best_len = len(best_sol)
        else:
            # Fallback: rebuild valid subset
            valid_subset = {}
            for idx in sorted(best_sol.keys()):
                r = best_sol[idx]
                ok = True
                for j, rj in valid_subset.items():
                    g = gcd_mat[idx][j]
                    if r % g == rj % g:
                        ok = False
                        break
                if ok:
                    valid_subset[idx] = r
            best_sol = valid_subset
            best_len = len(best_sol)

    if not best_sol:
        # Trivial fallback
        best_sol = {0: 0}
        best_len = 1

    return [[r, moduli[idx]] for idx, r in best_sol.items()]
