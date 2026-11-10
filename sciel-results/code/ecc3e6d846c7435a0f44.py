import numpy as np
import scipy.sparse as sp

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Precompute clause literal arrays
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8) # 1 for pos, -1 for neg
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = 1 if lit > 0 else -1

    # Build sparse M_pos and M_neg
    # M_pos[v, c] = 1 if clause c contains +v
    # M_neg[v, c] = 1 if clause c contains -v
    rows_pos = []
    cols_pos = []
    rows_neg = []
    cols_neg = []
    
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if clause_signs[i, j] == 1:
                rows_pos.append(v)
                cols_pos.append(i)
            else:
                rows_neg.append(v)
                cols_neg.append(i)

    M_pos = sp.csr_matrix((np.ones(len(rows_pos)), (rows_pos, cols_pos)), shape=(n, m))
    M_neg = sp.csr_matrix((np.ones(len(rows_neg)), (rows_neg, cols_neg)), shape=(n, m))

    rng = np.random.RandomState(42)
    assignment = rng.rand(n) < 0.5

    # Compute initial clause_sat
    a = assignment[clause_vars] # (m, 3)
    s = (clause_signs == 1) # (m, 3)
    lit_sat = (a == s)
    clause_sat = lit_sat.any(axis=1)

    if clause_sat.all():
        return assignment.tolist()

    max_iters = 200000
    p_walk = 0.5
    
    for _ in range(max_iters):
        if clause_sat.all():
            break
            
        unsat_mask = ~clause_sat
        num_unsat = np.sum(unsat_mask)
        
        if num_unsat == 0:
            break

        # Compute gains using sparse matrix multiplication
        unsat_counts_pos = np.asarray(M_pos @ unsat_mask).flatten()
        unsat_counts_neg = np.asarray(M_neg @ unsat_mask).flatten()
        
        sat_counts_pos = np.asarray(M_pos @ clause_sat).flatten()
        sat_counts_neg = np.asarray(M_neg @ clause_sat).flatten()
        
        gain_true = unsat_counts_pos - sat_counts_neg
        gain_false = unsat_counts_neg - sat_counts_pos
        
        max_gain_true = np.max(gain_true)
        max_gain_false = np.max(gain_false)
        
        best_gain = -np.inf
        best_var = -1
        best_val = False
        
        if max_gain_true > max_gain_false:
            best_gain = max_gain_true
            candidates = np.where(gain_true == max_gain_true)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = True
        elif max_gain_false > max_gain_true:
            best_gain = max_gain_false
            candidates = np.where(gain_false == max_gain_false)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = False
        else:
            # Tie
            if max_gain_true == max_gain_false:
                if rng.rand() < 0.5:
                    candidates = np.where(gain_true == max_gain_true)[0]
                    best_var = candidates[rng.randint(len(candidates))]
                    best_val = True
                    best_gain = max_gain_true
                else:
                    candidates = np.where(gain_false == max_gain_false)[0]
                    best_var = candidates[rng.randint(len(candidates))]
                    best_val = False
                    best_gain = max_gain_false
            else:
                best_var = int(rng.randint(0, n))
                best_val = not assignment[best_var]
                best_gain = 0

        # Escape mechanism
        if best_gain <= 0 and rng.rand() < p_walk:
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                break
            ci = unsat_indices[rng.randint(len(unsat_indices))]
            j = rng.randint(3)
            v = int(clause_vars[ci, j])
            best_var = v
            if rng.rand() < 0.5:
                best_val = (clause_signs[ci, j] == 1)
            else:
                best_val = not assignment[v]

        # Apply flip
        if best_val != assignment[best_var]:
            assignment[best_var] = best_val
            
            # Update clause_sat for clauses containing best_var
            pos_affected = M_pos[best_var].indices
            neg_affected = M_neg[best_var].indices
            
            if len(pos_affected) > 0 or len(neg_affected) > 0:
                affected_clauses = np.unique(np.concatenate([pos_affected, neg_affected]))
            else:
                continue

            for ci in affected_clauses:
                is_sat = False
                for j in range(3):
                    v = int(clause_vars[ci, j])
                    s = (clause_signs[ci, j] == 1)
                    if assignment[v] == s:
                        is_sat = True
                        break
                clause_sat[ci] = is_sat

    return assignment.tolist()
