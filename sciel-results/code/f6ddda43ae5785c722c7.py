"""Rewritten program with spectral initialization for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from itertools import combinations

def _build_clause_data(n, clauses):
    """Precompute clause structure as numpy arrays for vectorized operations."""
    m = len(clauses)
    if m == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.int8), {i: [] for i in range(n)}
    
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_sgn = np.zeros((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_sgn[i, j] = 1 if lit > 0 else -1
    
    # For each variable, precompute which (clause, position) pairs contain it
    var_to_clauses = {v: [] for v in range(n)}
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            var_to_clauses[v].append((i, j))
            
    return clause_vars, clause_sgn, var_to_clauses

def _spectral_seed(n, clauses, rng):
    """
    Initialize assignment using spectral bisection heuristic.
    We construct a graph where an edge between u and v has weight w_uv.
    We want to maximize the number of satisfied clauses.
    A common heuristic for planted SAT is to assume that literals appearing together
    in clauses tend to have the same truth value if they have the same sign,
    or opposite if different signs, relative to the planted solution.
    
    However, a more direct spectral approach for Max-Cut-like problems on SAT:
    We can view the problem as minimizing conflicts.
    Let's build a Laplacian based on variable co-occurrence.
    
    Actually, a simpler and effective spectral initialization for planted SAT:
    1. Count positive and negative occurrences (Phase).
    2. Construct a matrix A where A[i,j] represents the "affinity" or "conflict".
       If i and j appear in the same clause:
       - If they have the same sign (both + or both -), assigning them differently 
         might be bad if the clause is "tight". But usually, in 3-SAT, we want 
         at least one True.
       
    Let's stick to a robust spectral method:
    Construct a weighted graph G=(V,E) where V={0..n-1}.
    For each clause (l1, l2, l3):
      The clause is unsatisfied iff all literals are False.
      This is equivalent to: NOT (l1 OR l2 OR l3) == l1' AND l2' AND l3'
      
    Alternative: Use the Laplacian of the variable interaction graph.
    We want to find a cut that minimizes "unsatisfied" cost.
    
    Let's use a heuristic spectral seed:
    1. Build a symmetric matrix M of size n x n.
    2. For each pair of variables (u, v) appearing in the same clause:
       If the clause is (u, v, w) with signs s_u, s_v, s_w.
       The condition for the clause to be satisfied is s_u*x_u + s_v*x_v + s_w*x_w >= 1 (if we map True=1, False=0).
       
    A simpler, proven effective spectral heuristic for planted instances:
    Map the SAT instance to a Max-Cut problem on a graph where nodes are variables.
    Edge weight between u and v is determined by their co-occurrence.
    If u and v appear in a clause with the SAME sign, we prefer them to be DIFFERENT? No.
    If u and v appear in a clause with OPPOSITE signs, we prefer them to be SAME?
    
    Let's use the following logic for the seed:
    1. Compute phase (majority vote of signs).
    2. Construct a matrix B where B[i,j] = 1 if i and j appear together in a clause.
    3. Use the eigenvector corresponding to the second smallest eigenvalue of the Laplacian of B?
    
    Actually, let's use a direct spectral method on the "clause matrix".
    Let X be the n x m matrix of clause literals (1 if var in clause, -1 if negated, 0 otherwise).
    We want to find x in {+1, -1}^n such that X^T x has few zeros (or rather, the clause is satisfied).
    Clause c is satisfied if sum(lit_val) != 0? No, if any lit is True.
    Let's map True=1, False=-1.
    Clause (l1, l2, l3) is satisfied if l1*x1 + l2*x2 + l3*x3 != -3 (i.e., not all -1).
    Equivalently, we want to maximize the sum of min(1, |l1*x1 + l2*x2 + l3*x3|)? No.
    
    Let's stick to the prompt's suggestion: "Formulate ... as a Max-Cut problem ... edges represent conflicting literal pairs".
    Conflicting literal pairs: If a clause contains (u, v), and we assign u and v such that both are False, it's bad.
    This is hard to map directly to standard Max-Cut.
    
    Instead, let's use a "Spectral Phase" approach:
    1. Build a matrix A where A[i,j] = count of clauses containing both i and j.
    2. Compute the leading eigenvector of A?
    
    Let's try a different spectral heuristic:
    Construct a graph where an edge exists between i and j if they appear in a clause.
    Weight the edge based on sign agreement.
    If i and j have the same sign in a clause, we want them to be "similar" in some sense?
    If they have opposite signs, "different"?
    
    Let's use the Fiedler vector of the Laplacian of the co-occurrence graph.
    """
    # 1. Phase seed as baseline
    pos_count = np.zeros(n)
    neg_count = np.zeros(n)
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    
    # 2. Build co-occurrence matrix with sign-aware weights
    # We want to maximize agreement with the "planted" structure.
    # Hypothesis: In planted SAT, variables appearing with the same sign in many clauses
    # might be correlated with the solution.
    
    # Let's build a matrix M where M[i,j] = 1 if i,j appear in a clause with same sign,
    # M[i,j] = -1 if opposite sign?
    # Then find the leading eigenvector.
    
    M = np.zeros((n, n))
    for clause in clauses:
        lits = clause
        for i in range(3):
            for j in range(i+1, 3):
                v_i = abs(lits[i]) - 1
                v_j = abs(lits[j]) - 1
                s_i = 1 if lits[i] > 0 else -1
                s_j = 1 if lits[j] > 0 else -1
                
                if v_i == v_j:
                    continue # Self loop, ignore or handle separately
                    
                # If signs are same, we might want x_i == x_j? 
                # If signs are different, we might want x_i != x_j?
                # This is a heuristic. Let's assume "same sign -> same value" and "diff sign -> diff value" 
                # is a good proxy for planted instances where clauses are generated from a solution.
                # Wait, if clause is (x, y, z) and solution is T,T,T, then x,y,z are all True.
                # Signs are all +. So same sign -> same value (True).
                # If clause is (x, -y, z) and solution T,T,T, then x=T, -y=F, z=T.
                # Here x and z are same sign (+), and both True.
                # x and -y are diff sign. x=T, y=T. So x and y are same value.
                # So "diff sign" does NOT imply "diff value".
                
                # Actually, the clause is satisfied if ANY literal is True.
                # This doesn't force specific values for unsatisfied literals.
                
                # Let's use a simpler spectral method: 
                # Solve the linear system derived from the clause matrix?
                
                # For now, let's just use the phase seed but perturbed by a random walk on the clause graph.
                # Or, use the eigenvector of the clause incidence matrix.
                
                # Let's try: X is n x m. Find x such that X^T x is large positive?
                # No, we want each column sum (clause) to be >= 1 (if 1=True, -1=False)?
                # Clause satisfied if sum(lit*x) != -3.
                
                # Let's just use a robust phase seed + random perturbation for now, 
                # and rely on the local search to fix it. The spectral part might be overkill 
                # or incorrect without more specific planted structure knowledge.
                
                # However, the prompt specifically asks for spectral bisection.
                # Let's implement a standard spectral bisection on the variable co-occurrence graph.
                # Graph: Nodes = vars. Edge (u,v) weight = 1 if they appear in same clause.
                # Laplacian L = D - A.
                # Fiedler vector v2.
                # Assignment: x_i = 1 if v2[i] > 0 else -1.
                
                # This tends to separate clusters. If the planted solution has clusters, this helps.
                pass # Logic moved to main spectral function below

    # Build Co-occurrence Matrix A
    A = np.zeros((n, n))
    for clause in clauses:
        vars_in_clause = [abs(lit) - 1 for lit in clause]
        # Add 1 to all pairs
        for i in range(3):
            for j in range(i+1, 3):
                u = vars_in_clause[i]
                v = vars_in_clause[j]
                if u != v:
                    A[u, v] += 1
                    A[v, u] += 1
    
    # Compute Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Compute eigenvalues and eigenvectors
    # We want the second smallest eigenvalue (Fiedler vector)
    try:
        # Use scipy if available, else numpy
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        L_sparse = csr_matrix(L)
        # Find the two smallest eigenvalues
        # k=2, sigma=0 to find smallest
        vals, vecs = eigsh(L_sparse, k=2, sigma=0, which='LM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
    except Exception:
        # Fallback to numpy if scipy fails or not available
        vals, vecs = np.linalg.eigh(L)
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
        
    # Normalize Fiedler vector to mean 0
    fiedler -= np.mean(fiedler)
    
    # Create assignment based on sign of Fiedler vector
    # But we should also consider the phase.
    # Let's blend: if phase is strong, use phase. If not, use spectral.
    # Or, use spectral to break ties in phase?
    
    # Simple strategy: 
    # 1. Compute phase seed.
    # 2. Compute spectral seed.
    # 3. If they agree, keep. If not, pick randomly or use a weighted vote.
    
    # Let's just return the spectral seed, but flipped if it disagrees with phase on majority?
    # Actually, let's just return the spectral seed. The local search will handle the rest.
    
    # Map Fiedler vector to booleans
    # True if > 0, False if <= 0
    assignment = fiedler > 0
    
    # Ensure it's a list of bools
    return assignment.tolist()

def _phase_seed(n, clauses, rng):
    """Initialize assignment using phase information from clause polarities."""
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    assignment = []
    for i in range(n):
        if pos_count[i] > neg_count[i]:
            assignment.append(True)
        elif neg_count[i] > pos_count[i]:
            assignment.append(False)
        else:
            assignment.append(rng.random() < 0.5)
    return assignment

def _count_unsat(assignment, clauses):
    """Count number of unsatisfied clauses."""
    count = 0
    for clause in clauses:
        sat = False
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                sat = True
                break
        if not sat:
            count += 1
    return count

def _compute_sat_state(clause_vars, clause_sgn, assignment):
    """Compute per-clause satisfaction state: sat_count (how many literals satisfied)."""
    var_idx = clause_vars
    signs = clause_sgn
    truth = assignment[var_idx]
    sat_lit = (signs == 1) & truth | (signs == -1) & (~truth)
    sat_count = sat_lit.sum(axis=1)
    return sat_count

def _vectorized_impact_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, max_steps, deadline):
    """
    Vectorized WalkSAT with impact-based literal selection.
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)
    
    for step in range(max_steps):
        if time.time() >= deadline:
            break

        unsat_mask = sat_count == 0
        n_unsat = int(unsat_mask.sum())
        if n_unsat == 0:
            break

        # Pick a random unsatisfied clause
        unsat_indices = np.where(unsat_mask)[0]
        ci = int(rng.choice(unsat_indices.tolist()))

        lits_vars = clause_vars[ci]  # (3,)
        lits_sgn = clause_sgn[ci]    # (3,)

        # Adaptive noise
        progress = step / max(1, max_steps)
        noise = 0.30 - 0.15 * progress

        if rng.random() < noise:
            li = rng.randint(0, 2)
            v = int(lits_vars[li])
        else:
            # Greedy: compute gain for each of the 3 literals
            gains = np.zeros(3, dtype=np.int32)
            for li in range(3):
                v = int(lits_vars[li])
                old_val = assignment[v]
                new_val = not old_val

                vcl = var_to_clauses[v]
                if not vcl:
                    gains[li] = 0
                    continue

                cidx = np.array([p[0] for p in vcl], dtype=np.int32)
                lidx = np.array([p[1] for p in vcl], dtype=np.int32)
                s2 = clause_sgn[cidx, lidx]

                old_lit_sat = (s2 == 1) & old_val | (s2 == -1) & (~old_val)
                new_lit_sat = (s2 == 1) & new_val | (s2 == -1) & (~new_val)

                sc = sat_count[cidx]

                newly_sat = (sc == 0) & new_lit_sat
                newly_unsat = (sc == 1) & old_lit_sat & (~new_lit_sat)

                gains[li] = int(newly_sat.sum()) - int(newly_unsat.sum())

            max_gain = gains.max()
            best_candidates = np.where(gains == max_gain)[0]
            li = int(rng.choice(best_candidates.tolist()))
            v = int(lits_vars[li])

        # Apply the flip
        old_val = assignment[v]
        assignment[v] = not assignment[v]
        new_val = not old_val

        # Update sat_count for all clauses containing v
        for (ci2, li2) in var_to_clauses[v]:
            ci2 = int(ci2)
            li2 = int(li2)
            s2 = int(clause_sgn[ci2, li2])
            if s2 == 1:
                old_lit_sat = old_val
                new_lit_sat = new_val
            else:
                old_lit_sat = not old_val
                new_lit_sat = not new_val
            if new_lit_sat and not old_lit_sat:
                sat_count[ci2] += 1
            elif old_lit_sat and not new_lit_sat:
                sat_count[ci2] -= 1

    return assignment.tolist()

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # Leave 50s margin

    m = len(clauses)
    if m == 0:
        return [False] * n

    clause_vars, clause_sgn, var_to_clauses = _build_clause_data(n, clauses)

    best_assignment = None
    best_unsat_count = float('inf')

    def _try_update(assignment):
        nonlocal best_assignment, best_unsat_count
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        return unsat_count == 0

    # Strategy 1: Spectral Seed + WalkSAT
    rng_spectral = random.Random(12345)
    try:
        spectral_assign = _spectral_seed(n, clauses, rng_spectral)
    except Exception:
        spectral_assign = _phase_seed(n, clauses, rng_spectral)
    
    if _try_update(spectral_assign):
        return spectral_assign

    # Run WalkSAT from spectral seed
    assignment = _vectorized_impact_walksat(
        spectral_assign, clause_vars, clause_sgn, var_to_clauses, n, m, rng_spectral, 2000, deadline
    )
    if _try_update(assignment):
        return assignment

    # Strategy 2: Phase Seed + WalkSAT (Multiple restarts)
    for attempt in range(100):
        if time.time() >= deadline:
            break
        seed = attempt * 7919 + 12345
        rng = random.Random(seed)

        if attempt == 0:
            assignment = _phase_seed(n, clauses, rng)
        elif attempt == 1:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            if best_assignment is not None:
                assignment = best_assignment[:]
                for _ in range(n // 3):
                    idx = rng.randrange(n)
                    assignment[idx] = not assignment[idx]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        steps = 1500 + attempt * 100
        assignment = _vectorized_impact_walksat(
            assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, steps, deadline
        )

        if _try_update(assignment):
            return assignment

    # Strategy 3: More random restarts
    for attempt in range(50):
        if time.time() >= deadline:
            break
        seed = 100000 + attempt * 31337
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        assignment = _vectorized_impact_walksat(
            assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, 2500, deadline
        )

        if _try_update(assignment):
            return assignment

    # Final polish
    if best_assignment is not None and time.time() < deadline - 2:
        unsat_count = _count_unsat(best_assignment, clauses)
        if 0 < unsat_count <= 4:
            unsat_vars = set()
            for clause in clauses:
                sat = False
                for lit in clause:
                    if best_assignment[abs(lit) - 1] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    for lit in clause:
                        unsat_vars.add(abs(lit) - 1)
            unsat_vars = list(unsat_vars)
            max_flip = min(len(unsat_vars), 5)
            for r in range(1, max_flip + 1):
                if time.time() >= deadline - 1:
                    break
                for combo in combinations(unsat_vars, r):
                    candidate = best_assignment[:]
                    for v in combo:
                        candidate[v] = not candidate[v]
                    if _count_unsat(candidate, clauses) == 0:
                        return candidate
            for v in range(n):
                if time.time() >= deadline - 1:
                    break
                candidate = best_assignment[:]
                candidate[v] = not candidate[v]
                if _count_unsat(candidate, clauses) == 0:
                    return candidate

    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
