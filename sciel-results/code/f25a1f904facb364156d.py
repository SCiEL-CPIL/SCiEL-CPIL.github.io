"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Exact check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for c in range(rank):
                    s += u_row[c] * v_row[c] * w_row[c]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_index_maps(n, m, p):
    """Precompute which (i,j,k) triples each row/col in U,V,W participates in."""
    # For U[i*m+j], it participates in triples where that i,j and any k
    u_map = {}  # u_idx -> list of (i,j,k)
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            u_map[u_idx] = [(i, j, k) for k in range(p)]
    
    # For V[j*p+k], it participates in triples where that j,k and any i
    v_map = {}  # v_idx -> list of (i,j,k)
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            v_map[v_idx] = [(i, j, k) for i in range(n)]
    
    # For W[k*n+i], it participates in triples where that k,i and any j
    w_map = {}  # w_idx -> list of (i,j,k)
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            w_map[w_idx] = [(i, j, k) for j in range(m)]
    
    return u_map, v_map, w_map


def _search_rank(r, n, m, p, time_budget, seed):
    """Local search for an exact rank-r decomposition. Returns (U,V,W) or None."""
    start = time.time()
    rng = random.Random(seed)

    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    # Precompute index maps
    u_map, v_map, w_map = _build_index_maps(n, m, p)
    
    # Precompute triple-to-row mappings
    # triple (i,j,k) -> (u_idx, v_idx, w_idx)
    triple_to_rows = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triple_to_rows[(i,j,k)] = (i*m+j, j*p+k, k*n+i)
    
    # Sparse random init with better density
    density = 0.4
    for row in range(n * m):
        for col in range(r):
            if rng.random() < density:
                U[row][col] = rng.choice([1, -1])
    for row in range(m * p):
        for col in range(r):
            if rng.random() < density:
                V[row][col] = rng.choice([1, -1])
    for row in range(p * n):
        for col in range(r):
            if rng.random() < density:
                W[row][col] = rng.choice([1, -1])

    # Precompute current sums for each triple
    # sums[(i,j,k)] = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    sums = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx, v_idx, w_idx = triple_to_rows[(i,j,k)]
                s = 0
                u_row = U[u_idx]
                v_row = V[v_idx]
                w_row = W[w_idx]
                for c in range(r):
                    s += u_row[c] * v_row[c] * w_row[c]
                sums[(i,j,k)] = s

    def compute_error():
        e = 0
        for (i,j,k), s in sums.items():
            e += (s - 1) * (s - 1)
        return e

    cur = compute_error()
    best = cur
    best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
    no_improve = 0
    steps = 0
    max_steps = 8000

    # Fast delta computation: when we change U[u_idx][c] by delta,
    # the change in sum for triple (i,j,k) is delta * V[v_idx][c] * W[w_idx][c]
    def get_affected_triples(factor, idx, col):
        """Get list of (triple_key, other_factor_idx, other_col_value) for affected triples."""
        if factor == 'U':
            triples = u_map[idx]
            return [(t, triple_to_rows[t][1], V[triple_to_rows[t][1]][col]) for t in triples]
        elif factor == 'V':
            triples = v_map[idx]
            return [(t, triple_to_rows[t][2], W[triple_to_rows[t][2]][col]) for t in triples]
        else:  # 'W'
            triples = w_map[idx]
            return [(t, triple_to_rows[t][0], U[triple_to_rows[t][0]][col]) for t in triples]

    factors = [('U', U, n * m), ('V', V, m * p), ('W', W, p * n)]
    
    while no_improve < 300 and steps < max_steps:
        if time.time() - start > time_budget:
            break
        steps += 1
        
        # Choose factor, row, col
        factor_name, M, nrows = factors[rng.randrange(3)]
        row = rng.randrange(nrows)
        col = rng.randrange(r)
        old = M[row][col]
        
        # Try a new value
        candidates = [v for v in [0, 1, -1] if v != old]
        new = rng.choice(candidates)
        
        # Compute delta error using precomputed sums
        delta_val = new - old
        affected = get_affected_triples(factor_name, row, col)
        
        # Compute new error
        new_err = cur
        for (t, other_idx, other_val) in affected:
            old_sum = sums[t]
            new_sum = old_sum + delta_val * other_val
            new_err -= (old_sum - 1) * (old_sum - 1)
            new_err += (new_sum - 1) * (new_sum - 1)
        
        if new_err < cur:
            # Accept: update sums
            for (t, other_idx, other_val) in affected:
                sums[t] += delta_val * other_val
            M[row][col] = new
            cur = new_err
            no_improve = 0
            if cur < best:
                best = cur
                best_state = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        else:
            # Reject with some probability (simulated annealing-like)
            if rng.random() > 0.15:
                pass  # keep old value
            else:
                # Accept anyway to escape local minima
                for (t, other_idx, other_val) in affected:
                    sums[t] += delta_val * other_val
                M[row][col] = new
                cur = new_err
            no_improve += 1

    Ub, Vb, Wb = best_state
    if _verify(Ub, Vb, Wb, n, m, p):
        return Ub, Vb, Wb
    
    # Try a final polish: greedy improvement on the best state
    U2, V2, W2 = [row[:] for row in Ub], [row[:] for row in Vb], [row[:] for row in Wb]
    # Recompute sums for polish
    sums2 = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx, v_idx, w_idx = triple_to_rows[(i,j,k)]
                s = 0
                for c in range(r):
                    s += U2[u_idx][c] * V2[v_idx][c] * W2[w_idx][c]
                sums2[(i,j,k)] = s
    
    cur2 = sum((s-1)**2 for s in sums2.values())
    improved = True
    polish_steps = 0
    while improved and polish_steps < 500:
        improved = False
        polish_steps += 1
        for factor_name, M2, nrows in [('U', U2, n*m), ('V', V2, m*p), ('W', W2, p*n)]:
            for row in range(nrows):
                for col in range(r):
                    old = M2[row][col]
                    for new in [v for v in [0,1,-1] if v != old]:
                        delta_val = new - old
                        affected = get_affected_triples(factor_name, row, col)
                        new_err2 = cur2
                        for (t, other_idx, other_val) in affected:
                            # Use original V/W/U values from best_state, not M2
                            if factor_name == 'U':
                                ov = Vb[other_idx][col]
                            elif factor_name == 'V':
                                ov = Wb[other_idx][col]
                            else:
                                ov = Ub[other_idx][col]
                            old_sum = sums2[t]
                            new_sum = old_sum + delta_val * ov
                            new_err2 -= (old_sum - 1) * (old_sum - 1)
                            new_err2 += (new_sum - 1) * (new_sum - 1)
                        if new_err2 < cur2:
                            for (t, other_idx, other_val) in affected:
                                if factor_name == 'U':
                                    sums2[t] += delta_val * Vb[other_idx][col]
                                elif factor_name == 'V':
                                    sums2[t] += delta_val * Wb[other_idx][col]
                                else:
                                    sums2[t] += delta_val * Ub[other_idx][col]
                            M2[row][col] = new
                            cur2 = new_err2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    if _verify(U2, V2, W2, n, m, p):
        return U2, V2, W2
    return None


def _groebner_f3(n, m, p, rank, time_budget, seed):
    """
    Solve the tensor decomposition equations over F_3 using a Groebner basis.
    Variables: u[i*m+j][c], v[j*p+k][c], w[k*n+i][c] for c in range(rank)
    Equations: for each (i,j,k), sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] = 1
    
    We use a simple Groebner basis approach with lex order.
    To make it tractable, we fix some variables to 1 (normalization) and solve.
    """
    start = time.time()
    rng = random.Random(seed)
    
    MOD = 3
    n_vars = n*m*rank + m*p*rank + p*n*rank
    
    # Variable indexing
    # U vars: 0 to n*m*rank - 1
    # V vars: n*m*rank to n*m*rank + m*p*rank - 1
    # W vars: n*m*rank + m*p*rank to n_vars - 1
    
    def u_var(i, j, c):
        return (i*m + j) * rank + c
    
    def v_var(j, k, c):
        return n*m*rank + (j*p + k) * rank + c
    
    def w_var(k, i, c):
        return n*m*rank + m*p*rank + (k*n + i) * rank + c
    
    # Build polynomial system
    # Each equation: sum_c u[i*m+j][c] * v[j*p+k][c] * w[k*n+i][c] - 1 = 0
    # In F_3, -1 = 2
    
    # We'll represent polynomials as dict {tuple_of_vars: coeff}
    # A monomial is a tuple of variable indices (sorted, with multiplicity)
    
    equations = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                poly = {}
                for c in range(rank):
                    uv = u_var(i, j, c)
                    vv = v_var(j, k, c)
                    wv = w_var(k, i, c)
                    mon = tuple(sorted([uv, vv, wv]))
                    poly[mon] = poly.get(mon, 0) + 1
                # Subtract 1 (i.e., add 2 in F_3)
                poly[()] = poly.get((), 0) + 2
                # Reduce coefficients mod 3
                poly = {k: v % MOD for k, v in poly.items() if v % MOD != 0}
                if poly:
                    equations.append(poly)
    
    # Normalization: fix some variables to 1 to break symmetry
    # This reduces the search space significantly
    # Fix U[0][0] = 1 (variable u_var(0,0,0))
    # Fix V[0][0] = 1 (variable v_var(0,0,0))
    # Fix W[0][0] = 1 (variable w_var(0,0,0))
    
    fixed_vars = {
        u_var(0, 0, 0): 1,
        v_var(0, 0, 0): 1,
        w_var(0, 0, 0): 1,
    }
    
    # Substitute fixed variables into equations
    def substitute(poly, fixed):
        new_poly = {}
        for mon, coeff in poly.items():
            # Check if any variable in mon is fixed
            new_coeff = coeff
            remaining_vars = []
            for v in mon:
                if v in fixed:
                    new_coeff = (new_coeff * fixed[v]) % MOD
                else:
                    remaining_vars.append(v)
            new_mon = tuple(sorted(remaining_vars))
            if new_coeff % MOD != 0:
                new_poly[new_mon] = (new_poly.get(new_mon, 0) + new_coeff) % MOD
        return {k: v for k, v in new_poly.items() if v % MOD != 0}
    
    subs_eqs = [substitute(eq, fixed_vars) for eq in equations]
    subs_eqs = [eq for eq in subs_eqs if eq]
    
    # Use a simple Groebner basis / linear algebra approach
    # For small systems, we can use exhaustive search with pruning
    
    # Get all free variables
    all_vars = set()
    for eq in subs_eqs:
        for mon in eq:
            all_vars.update(mon)
    
    # Sort free variables for consistent ordering
    free_vars = sorted(all_vars)
    n_free = len(free_vars)
    
    # If too many free variables, this approach won't work
    if n_free > 25:
        return None
    
    # Try to solve using backtracking with constraint propagation
    # This is essentially a SAT-like solver for polynomial equations over F_3
    
    var_to_idx = {v: i for i, v in enumerate(free_vars)}
    
    def eval_poly(poly, assignment):
        """Evaluate polynomial given assignment of some variables."""
        result = 0
        for mon, coeff in poly.items():
            val = coeff
            for v in mon:
                if v in assignment:
                    val = (val * assignment[v]) % MOD
                else:
                    return None  # Can't evaluate yet
            result = (result + val) % MOD
        return result
    
    def try_solve(eqs, assignment, depth):
        """Backtracking solver."""
        if time.time() - start > time_budget:
            return None
        
        if depth >= n_free:
            # Check all equations
            for eq in eqs:
                val = eval_poly(eq, assignment)
                if val != 0:
                    return None
            return assignment
        
        var = free_vars[depth]
        for val in [0, 1, 2]:
            new_assignment = assignment.copy()
            new_assignment[var] = val
            result = try_solve(eqs, new_assignment, depth + 1)
            if result is not None:
                return result
        return None
    
    # Better approach: use Gaussian elimination on the linear parts first
    # Then handle nonlinear parts
    
    # Actually, let's use a more systematic approach:
    # 1. Collect all monomials that appear
    # 2. Introduce new variables for each monomial
    # 3. Solve the resulting linear system
    # 4. Check consistency
    
    # For this problem, let's try a different strategy:
    # Use the fact that we can normalize and then do a structured search
    
    # Simpler approach: random initialization + local search over F_3
    # This is similar to the existing search but over F_3
    
    def f3_search(time_budget_f3, seed_f3):
        """Local search over F_3."""
        start_f3 = time.time()
        rng_f3 = random.Random(seed_f3)
        
        # Initialize randomly over F_3
        U = [[rng_f3.choice([0, 1, 2]) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng_f3.choice([0, 1, 2]) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng_f3.choice([0, 1, 2]) for _ in range(rank)] for _ in range(p*n)]
        
        # Apply normalization
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        def compute_sums():
            sums = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s = (s + U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]) % MOD
                        sums[(i,j,k)] = s
            return sums
        
        sums = compute_sums()
        cur_err = sum((s - 1) * (s - 1) % MOD for s in sums.values())
        
        no_improve = 0
        max_steps = 20000
        
        while no_improve < 1000:
            if time.time() - start_f3 > time_budget_f3:
                break
            
            # Choose a random element to change
            factor = rng_f3.choice(['U', 'V', 'W'])
            if factor == 'U':
                row = rng_f3.randrange(n*m)
                col = rng_f3.randrange(rank)
                old = U[row][col]
                M = U
            elif factor == 'V':
                row = rng_f3.randrange(m*p)
                col = rng_f3.randrange(rank)
                old = V[row][col]
                M = V
            else:
                row = rng_f3.randrange(p*n)
                col = rng_f3.randrange(rank)
                old = W[row][col]
                M = W
            
            new_val = rng_f3.choice([v for v in [0, 1, 2] if v != old])
            delta = (new_val - old) % MOD
            
            # Compute affected triples
            if factor == 'U':
                i = row // m
                j = row % m
                affected = [(i, j, k) for k in range(p)]
            elif factor == 'V':
                j = row // p
                k = row % p
                affected = [(i, j, k) for i in range(n)]
            else:
                k = row // n
                i = row % n
                affected = [(i, j, k) for j in range(m)]
            
            new_err = cur_err
            for (i, j, k) in affected:
                old_sum = sums[(i,j,k)]
                new_sum = (old_sum + delta * (V[j*p+k][col] if factor == 'U' else W[k*n+i][col] if factor == 'V' else U[i*m+j][col])) % MOD
                new_err = (new_err - (old_sum - 1) * (old_sum - 1) + (new_sum - 1) * (new_sum - 1)) % (MOD * MOD)
            
            if new_err < cur_err:
                for (i, j, k) in affected:
                    sums[(i,j,k)] = (sums[(i,j,k)] + delta * (V[j*p+k][col] if factor == 'U' else W[k*n+i][col] if factor == 'V' else U[i*m+j][col])) % MOD
                M[row][col] = new_val
                cur_err = new_err
                no_improve = 0
            else:
                no_improve += 1
                if rng_f3.random() < 0.1:
                    for (i, j, k) in affected:
                        sums[(i,j,k)] = (sums[(i,j,k)] + delta * (V[j*p+k][col] if factor == 'U' else W[k*n+i][col] if factor == 'V' else U[i*m+j][col])) % MOD
                    M[row][col] = new_val
                    cur_err = new_err
        
        # Check if all sums are 1
        if all(s == 1 for s in sums.values()):
            return U, V, W
        return None
    
    # Try multiple seeds
    for seed_try in range(10):
        if time.time() - start > time_budget - 1:
            break
        res = f3_search(time_budget - 2, seed)
        if res is not None:
            U, V, W = res
            # Convert to integers (0, 1, 2 are already valid)
            return U, V, W
    
    return None


def construct():
    start_time = time.time()
    timeout = 140.0  # safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS

    # Guaranteed valid baseline
    U0, V0, W0 = _schoolbook(n, m, p)
    best = {"U": U0, "V": V0, "W": W0}
    best_rank = n * m * p

    # First try F_3 Groebner-based search for rank 19
    if time.time() - start_time < timeout - 20:
        res = _groebner_f3(n, m, p, 19, time_budget=30.0, seed=42)
        if res is not None:
            U, V, W = res
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = 19
                return best

    # Try progressively lower ranks with existing search
    for target_rank in range(23, 18, -1):
        if time.time() - start_time > timeout - 8:
            break
        for seed in range(6):
            if time.time() - start_time > timeout - 8:
                break
            res = _search_rank(target_rank, n, m, p, time_budget=12.0, seed=seed)
            if res is not None:
                U, V, W = res
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # found valid at this rank, try next (lower) rank

    # Final safety: ensure we return something valid
    if time.time() - start_time > timeout - 1:
        return best

    return best
# EVOLVE-BLOCK-END
