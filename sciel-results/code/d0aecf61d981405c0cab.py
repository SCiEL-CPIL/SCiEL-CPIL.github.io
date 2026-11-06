"""Planted 3-SAT solver (50 vars) — fast incremental WalkSAT with smooth surrogate, adaptive weights, polished finish."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285.0  # safety margin under 300s per-instance budget

    # Precompute clause literal indices for speed
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]
    clause_signs = [[(1 if l > 0 else -1) for l in c] for c in clauses]

    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i][j]].append(i)

    # Precompute for each clause, the position of each variable (for fast lookup)
    # clause_var_pos[ci][v] = position j in clause ci where variable v appears, or -1
    clause_var_pos = [[-1] * n for _ in range(m)]
    for i in range(m):
        for j in range(3):
            clause_var_pos[i][clause_vars[i][j]] = j

    def clause_satisfied(assign, ci):
        cv = clause_vars[ci]
        cs = clause_signs[ci]
        for j in range(3):
            if assign[cv[j]] == (cs[j] > 0):
                return True
        return False

    def full_unsat_count(assign):
        cnt = 0
        for i in range(m):
            if not clause_satisfied(assign, i):
                cnt += 1
        return cnt

    def is_satisfied(assign):
        for i in range(m):
            if not clause_satisfied(assign, i):
                return False
        return True

    def get_unsat_clauses(assign):
        unsat = []
        for i in range(m):
            if not clause_satisfied(assign, i):
                unsat.append(i)
        return unsat

    def compute_flip_delta(assign, v, new_val, unsat_set):
        """
        Compute change in unsat count if we flip variable v to new_val.
        Only considers clauses in unsat_set plus clauses that might become unsat.
        Returns (delta_unsat, list_of_clauses_affected)
        """
        delta = 0
        for ci in var_to_clauses[v]:
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            pos = clause_var_pos[ci][v]
            if pos < 0:
                continue
            cur_val = assign[v]
            # Currently: is clause satisfied?
            cur_sat = False
            for j in range(3):
                if j == pos:
                    continue
                if assign[cv[j]] == (cs[j] > 0):
                    cur_sat = True
                    break
            # After flip: is clause satisfied?
            new_sat = False
            for j in range(3):
                if j == pos:
                    if new_val == (cs[j] > 0):
                        new_sat = True
                        break
                else:
                    if assign[cv[j]] == (cs[j] > 0):
                        new_sat = True
                        break
            if cur_sat and not new_sat:
                delta += 1  # was satisfied, now unsat
            elif not cur_sat and new_sat:
                delta -= 1  # was unsat, now satisfied
        return delta

    def polish(assign, max_iters=200):
        """Deterministic sweep: flip any variable that strictly reduces unsat count. Uses incremental delta."""
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            unsat_cnt = full_unsat_count(assign)
            if unsat_cnt == 0:
                break
            changed = False
            for v in range(n):
                new_val = not assign[v]
                delta = compute_flip_delta(assign, v, new_val, set())
                if delta < 0:
                    assign[v] = new_val
                    unsat_cnt += delta
                    changed = True
                # else: don't flip (keep current)
            if not changed:
                break
        return assign

    def unit_propagate(assign):
        """Apply unit propagation. Returns (assign, solved_bool, conflict_bool)."""
        sat = [False] * m
        for i in range(m):
            if clause_satisfied(assign, i):
                sat[i] = True
        
        unit_queue = []
        for i in range(m):
            if sat[i]:
                continue
            false_cnt = 0
            for j in range(3):
                if not assign[clause_vars[i][j]] == (clause_signs[i][j] > 0):
                    false_cnt += 1
            if false_cnt == 2:
                unit_queue.append(i)
        
        while unit_queue:
            ci = unit_queue.pop()
            if sat[ci]:
                continue
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            target_v = -1
            target_s = 0
            for j in range(3):
                if not assign[cv[j]] == (cs[j] > 0):
                    target_v = cv[j]
                    target_s = cs[j]
                    break
            if target_v == -1:
                continue
            if assign[target_v] == (target_s > 0):
                sat[ci] = True
                continue
            if assign[target_v] != (target_s > 0):
                still_unsat = True
                for j in range(3):
                    if assign[cv[j]] == (cs[j] > 0):
                        still_unsat = False
                        break
                if still_unsat:
                    return assign, False, True
                else:
                    sat[ci] = True
                    continue
            
            assign[target_v] = (target_s > 0)
            sat[ci] = True
            
            for ui in var_to_clauses[target_v]:
                if sat[ui]:
                    continue
                if clause_satisfied(assign, ui):
                    sat[ui] = True
                    continue
                false_cnt = 0
                for k in range(3):
                    if not assign[clause_vars[ui][k]] == (clause_signs[ui][k] > 0):
                        false_cnt += 1
                if false_cnt == 2:
                    unit_queue.append(ui)
        
        for i in range(m):
            if not sat[i]:
                return assign, False, False
        return assign, True, False

    def spectral_estimate():
        """Count positive vs negative appearances to guide initialization."""
        if m == 0:
            return [True] * n
        pos_count = [0] * n
        neg_count = [0] * n
        for i in range(m):
            for j in range(3):
                v = clause_vars[i][j]
                s = clause_signs[i][j]
                if s > 0:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        scores = []
        for v in range(n):
            total = pos_count[v] + neg_count[v]
            if total == 0:
                scores.append(0.0)
            else:
                scores.append((pos_count[v] - neg_count[v]) / total)
        return scores

    def soft_unsat_score(assign, unsat_list, temperature=0.5):
        """
        Smooth surrogate: instead of hard count, use a soft penalty.
        For each unsat clause, weight by how 'close' it is to being satisfied.
        A clause with 1 literal satisfied is 'closer' than one with 0.
        This gives a smoother landscape for local search.
        """
        if not unsat_list:
            return 0.0
        score = 0.0
        for ci in unsat_list:
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            satisfied_lits = 0
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    satisfied_lits += 1
            # 0 satisfied -> penalty 1.0, 1 satisfied -> penalty exp(-1/temp), 2 -> even less
            # Since clause is unsat, satisfied_lits is 0 or 1 (for 3-clauses)
            penalty = math.exp(-satisfied_lits / temperature)
            score += penalty
        return score

    def walk_sat_fast(assign, rng, flip_budget, clause_weights):
        """
        Fast WalkSAT with incremental tracking and smooth surrogate guidance.
        clause_weights: list of weights per clause (adaptive).
        """
        unsat_cnt = full_unsat_count(assign)
        
        for step in range(flip_budget):
            if time.time() > deadline - 5.0:
                return assign, False
            if unsat_cnt == 0:
                return assign, True
            
            # Rebuild unsat list periodically (every 50 steps or when needed)
            if step % 50 == 0:
                unsat_list = get_unsat_clauses(assign)
                unsat_cnt = len(unsat_list)
                if unsat_cnt == 0:
                    return assign, True
            
            # Pick a random unsat clause (weighted)
            if step % 50 == 0 or not unsat_list:
                unsat_list = get_unsat_clauses(assign)
                unsat_cnt = len(unsat_list)
                if unsat_cnt == 0:
                    return assign, True
            
            # Weighted clause selection
            total_w = sum(clause_weights[ci] for ci in unsat_list)
            if total_w > 0:
                r = rng.random() * total_w
                cum = 0
                ci = unsat_list[0]
                for uci in unsat_list:
                    cum += clause_weights[uci]
                    if r <= cum:
                        ci = uci
                        break
            else:
                ci = unsat_list[rng.randrange(len(unsat_list))]
            
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            
            # Score each literal in the clause using incremental delta + smooth surrogate
            best_score = -1e9
            best_j = -1
            for j in range(3):
                v = cv[j]
                new_val = (cs[j] > 0)
                # Compute delta in unsat count
                delta = compute_flip_delta(assign, v, new_val, set())
                # Smooth surrogate bonus: prefer flips that make more literals in unsat clauses true
                smooth_bonus = 0.0
                for uci in unsat_list:
                    if uci == ci:
                        continue
                    pos = clause_var_pos[uci][v]
                    if pos >= 0:
                        if new_val == (clause_signs[uci][pos] > 0):
                            smooth_bonus += 0.1
                # Combined score
                score = -delta + smooth_bonus + rng.random() * 0.05
                if score > best_score:
                    best_score = score
                    best_j = j
            
            v = cv[best_j]
            old_val = assign[v]
            assign[v] = (cs[best_j] > 0)
            
            # Update clause weights: increase weight of unsat clauses containing v
            for uci in var_to_clauses[v]:
                if not clause_satisfied(assign, uci):
                    clause_weights[uci] = min(clause_weights[uci] * 1.1, 10.0)
            
            # Recompute unsat count incrementally
            unsat_cnt = full_unsat_count(assign)
            if unsat_cnt == 0:
                return assign, True
        
        return assign, is_satisfied(assign)

    # --- Initialization: try several seeds, keep best ---
    rng = random.Random(12345)
    best_assign = None
    best_unsat = m + 1

    # Generate candidate initial assignments
    candidates = []
    for _ in range(15):
        a = [rng.random() < 0.5 for _ in range(n)]
        candidates.append(a)
    candidates.append([True] * n)
    candidates.append([False] * n)
    candidates.append([i % 2 == 0 for i in range(n)])
    candidates.append([i % 3 == 0 for i in range(n)])
    candidates.append([i % 5 == 0 for i in range(n)])
    
    # Add spectral-based estimate
    scores = spectral_estimate()
    spectral_init = [s > 0.05 for s in scores]
    candidates.append(spectral_init)
    candidates.append([s < -0.05 for s in scores])  # Inverted spectral
    candidates.append([s >= 0 for s in scores])
    candidates.append([s < 0 for s in scores])

    for a in candidates:
        a2 = list(a)
        a2, solved, conflict = unit_propagate(a2)
        if solved:
            return a2
        u = full_unsat_count(a2)
        if u < best_unsat:
            best_unsat = u
            best_assign = list(a2)

    if best_unsat == 0:
        return best_assign

    # --- Main search: WalkSAT with restarts and adaptive clause weights ---
    current = list(best_assign)
    restarts = 0
    max_restarts = 50000  # More restarts possible due to faster steps
    clause_weights = [1.0] * m
    
    while time.time() < deadline - 10.0 and restarts < max_restarts:
        restarts += 1
        # Adaptive flip budget: shorter early, longer as we converge
        if best_unsat <= 3:
            flip_budget = 500 + rng.randrange(500)
        elif best_unsat <= 10:
            flip_budget = 300 + rng.randrange(400)
        else:
            flip_budget = 200 + rng.randrange(300)
        
        # Reset clause weights partially on restart
        if restarts % 10 == 0:
            for i in range(m):
                clause_weights[i] = max(1.0, clause_weights[i] * 0.8)
        
        res, solved = walk_sat_fast(current, rng, flip_budget, clause_weights)
        u = full_unsat_count(res)
        if u < best_unsat:
            best_unsat = u
            best_assign = list(res)
        if solved:
            return res
        
        # Occasionally re-randomize a few variables to escape local minima
        if restarts % 3 == 0:
            n_perturb = 1 + rng.randrange(3)
            for _ in range(n_perturb):
                v = rng.randrange(n)
                current[v] = not current[v]
        current = list(res)

    # Final polish on best found
    final = list(best_assign)
    polish(final, max_iters=300)
    if is_satisfied(final):
        return final
    
    # Try more polish passes with random perturbations
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
        perturbed = list(final)
        n_pert = 1 + rng.randrange(4)
        for _ in range(n_pert):
            v = rng.randrange(n)
            perturbed[v] = not perturbed[v]
        polish(perturbed, max_iters=150)
        if is_satisfied(perturbed):
            return perturbed
        u = full_unsat_count(perturbed)
        if u < best_unsat:
            best_unsat = u
            final = list(perturbed)
    
    return final


# EVOLVE-BLOCK-END
