# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def _build_masks(clauses, n):
    pos = []
    neg = []
    for c in clauses:
        p = 0
        q = 0
        for l in c:
            if l > 0:
                p |= (1 << (l - 1))
            else:
                q |= (1 << (-l - 1))
        pos.append(p)
        neg.append(q)
    return pos, neg


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 295.0  # safety margin before 330s kill
    pos, neg = _build_masks(clauses, n)
    pos = tuple(pos)
    neg = tuple(neg)

    # Precompute clause literal index lists for fast flip
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])

    # Build numpy arrays for vectorized satisfaction checking
    # For each clause, we need: pos_mask, neg_mask
    # A clause is satisfied if (assign & pos_mask) != 0 OR ((~assign) & neg_mask) != 0
    # In numpy: assign is a bit array, but for 50 vars we can use uint64
    
    # Precompute as arrays for faster access
    pos_arr = np.array(pos, dtype=np.uint64)
    neg_arr = np.array(neg, dtype=np.uint64)
    
    # Build a matrix: for each clause, for each variable in that clause, store (var_idx, is_pos)
    # This helps with fast variable selection
    clause_var_matrix = []
    for cv in clause_vars:
        clause_var_matrix.append(cv)

    best_assign = None
    best_sat = -1

    def sat_count_numpy(assign_int):
        # assign_int: bit i set means var i is True
        # For each clause: satisfied if (assign & pos) != 0 OR ((~assign) & neg) != 0
        # ~assign needs to be masked to n bits
        mask = (1 << n) - 1
        assign_np = np.uint64(assign_int)
        pos_satisfied = (assign_np & pos_arr) != 0
        neg_satisfied = ((~assign_np) & mask & neg_arr) != 0
        satisfied = pos_satisfied | neg_satisfied
        return int(np.sum(satisfied))

    def assign_to_int(assign):
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)
        return v

    def int_to_assign(v):
        return [bool(v & (1 << i)) for i in range(n)]

    rng = random.Random(12345)

    # Phase 0: quick pure-literal / unit-style greedy initializations
    for attempt_seed in range(8):
        r = random.Random(attempt_seed)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        s = sat_count_numpy(v)
        if s > best_sat:
            best_sat = s
            best_assign = assign
        if s == m:
            return assign
        if time.time() > deadline:
            return int_to_assign(assign_to_int(best_assign)) if best_assign else [False]*n

    # Phase 1: Multi-restart WalkSAT with improved heuristics
    for restart in range(4000):
        if time.time() > deadline:
            break
        r = random.Random(restart)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = assign_to_int(assign)
        steps_budget = 6000
        for step in range(steps_budget):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses using numpy
            mask = (1 << n) - 1
            assign_np = np.uint64(v)
            pos_satisfied = (assign_np & pos_arr) != 0
            neg_satisfied = ((~assign_np) & mask & neg_arr) != 0
            satisfied = pos_satisfied | neg_satisfied
            unsat_indices = np.where(~satisfied)[0]
            
            if len(unsat_indices) == 0:
                best_assign = int_to_assign(v)
                best_sat = m
                return best_assign
            
            # Pick a random unsatisfied clause
            ci = int(unsat_indices[r.randrange(len(unsat_indices))])
            
            # For this clause, evaluate each literal: count how many other unsat clauses it would satisfy
            cv = clause_var_matrix[ci]
            best_var = None
            best_score = -1
            
            # Precompute: for each variable, how many unsat clauses contain it (with correct polarity)
            # This is expensive, so sample if too many unsat
            if len(unsat_indices) > 50:
                sample_unsat = r.sample(list(unsat_indices), 50)
            else:
                sample_unsat = list(unsat_indices)
            
            var_scores = {}
            for var, is_pos in cv:
                # Count how many sample unsat clauses this variable appears in with helpful polarity
                help_count = 0
                for uci in sample_unsat:
                    for uvar, uis_pos in clause_var_matrix[uci]:
                        if uvar == var:
                            if is_pos and not uis_pos:
                                help_count += 1
                            elif not is_pos and uis_pos:
                                help_count += 1
                            else:
                                # Same polarity - flipping might hurt
                                help_count -= 0.5
                if var not in var_scores:
                    var_scores[var] = help_count
                else:
                    var_scores[var] = max(var_scores[var], help_count)
            
            # Add noise and pick best
            scores = []
            for var, is_pos in cv:
                base = var_scores.get(var, 0)
                noise = r.random() * 2.0
                scores.append((base + noise, var, is_pos))
            
            scores.sort(reverse=True)
            var, is_pos = scores[0][1], scores[0][2]
            
            bit = 1 << var
            v ^= bit
        
        # after inner loop, track best
        s = sat_count_numpy(v)
        if s > best_sat:
            best_sat = s
            best_assign = int_to_assign(v)
        if best_sat == m:
            return best_assign
        if time.time() > deadline:
            break

    # Phase 2: Final polish - try to fix remaining unsat clauses with targeted search
    if best_assign is not None and best_sat < m:
        v = assign_to_int(best_assign)
        mask = (1 << n) - 1
        remaining = 30  # seconds for polish
        polish_deadline = time.time() + remaining
        
        for polish_restart in range(100):
            if time.time() > polish_deadline:
                break
            
            # Start from best known assignment
            v_cur = v
            for pstep in range(2000):
                if time.time() > polish_deadline:
                    break
                
                assign_np = np.uint64(v_cur)
                pos_satisfied = (assign_np & pos_arr) != 0
                neg_satisfied = ((~assign_np) & mask & neg_arr) != 0
                satisfied = pos_satisfied | neg_satisfied
                unsat_indices = np.where(~satisfied)[0]
                
                if len(unsat_indices) == 0:
                    best_assign = int_to_assign(v_cur)
                    best_sat = m
                    return best_assign
                
                # Pick random unsat clause
                ci = int(unsat_indices[polish_restart % len(unsat_indices) if len(unsat_indices) > 0 else 0])
                if len(unsat_indices) > 0:
                    ci = int(unsat_indices[r.randrange(len(unsat_indices))])
                
                cv = clause_var_matrix[ci]
                # Try each literal, pick the one that satisfies most other unsat clauses
                best_var = None
                best_help = -1
                for var, is_pos in cv:
                    help_count = 0
                    for uci in unsat_indices[:20]:  # limit for speed
                        for uvar, uis_pos in clause_var_matrix[uci]:
                            if uvar == var:
                                if (is_pos and not uis_pos) or (not is_pos and uis_pos):
                                    help_count += 1
                    if help_count > best_help:
                        best_help = help_count
                        best_var = var
                
                if best_var is not None:
                    v_cur ^= (1 << best_var)
            
            s = sat_count_numpy(v_cur)
            if s > best_sat:
                best_sat = s
                best_assign = int_to_assign(v_cur)
                v = v_cur
            if best_sat == m:
                return best_assign

    if best_assign is None:
        return [False] * n
    return best_assign
# EVOLVE-BLOCK-END
