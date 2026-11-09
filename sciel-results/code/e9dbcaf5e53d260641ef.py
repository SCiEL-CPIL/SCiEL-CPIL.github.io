# EVOLVE-BLOCK-START
import time
import random
import heapq


def _satisfied(cl, ass):
    for v, p in cl:
        if ass[v] == p:
            return True
    return False


def _dpll_solve(clauses_opt, var_clauses, n, m, deadline):
    assign = [-1] * n
    trail = []
    level = [0] * n
    learnt = []
    
    activity = [1.0] * n
    var_heap = [(-activity[v], v) for v in range(n)]
    heapq.heapify(var_heap)
    phase = [True] * n
    
    def propagate(queue):
        while queue:
            ci, is_learnt = queue.pop()
            if is_learnt:
                if ci >= len(learnt): continue
                cl = learnt[ci]
            else:
                if ci >= m: continue
                cl = clauses_opt[ci]
            
            satisfied = False
            unfixed = []
            for v, p in cl:
                if assign[v] != -1:
                    if assign[v] == (1 if p else 0):
                        satisfied = True
                        break
                else:
                    unfixed.append((v, p))
            
            if satisfied:
                continue
            if not unfixed:
                return False, (ci, is_learnt)
            if len(unfixed) == 1:
                v, p = unfixed[0]
                if assign[v] == -1:
                    assign[v] = 1 if p else 0
                    trail.append(v)
                    level[v] = len(trail)
                    for ci2, p2 in var_clauses[v]:
                        queue.append((ci2, False))
                    for li in range(len(learnt)):
                        for v2, p2 in learnt[li]:
                            if v2 == v:
                                queue.append((li, True))
                                break
        return True, None

    def analyze_conflict(conf_clause):
        ci, is_learnt = conf_clause
        if is_learnt:
            cl = learnt[ci]
        else:
            cl = clauses_opt[ci]
        
        learnt_clause = []
        max_level = 0
        for v, p in cl:
            if assign[v] != -1:
                learnt_clause.append((v, p))
                if level[v] > max_level:
                    max_level = level[v]
        
        if len(learnt_clause) > 1:
            seen = set()
            unique = []
            for v, p in learnt_clause:
                if (v, p) not in seen:
                    seen.add((v, p))
                    unique.append((v, p))
            learnt.append(unique)
            for v, p in unique:
                activity[v] += 1.0
            return unique, max_level
        return None, 0

    q = [(ci, False) for ci in range(m)]
    ok, conf = propagate(q)
    if not ok:
        return None

    while time.time() < deadline:
        v = -1
        while var_heap:
            neg_act, cand = var_heap[0]
            if assign[cand] == -1:
                v = cand
                heapq.heappop(var_heap)
                break
            else:
                heapq.heappop(var_heap)
        
        if v == -1:
            for ci in range(m):
                cl = clauses_opt[ci]
                for vi, pi in cl:
                    if assign[vi] == (1 if pi else 0):
                        break
                else:
                    return None
            return [bool(assign[i]) for i in range(n)]
        
        val = 1 if phase[v] else 0
        assign[v] = val
        trail.append(v)
        level[v] = len(trail)
        
        q = [(ci2, False) for ci2, p2 in var_clauses[v]]
        for li in range(len(learnt)):
            for v2, p2 in learnt[li]:
                if v2 == v:
                    q.append((li, True))
                    break
        
        ok, conf = propagate(q)
        
        if ok:
            solved = True
            for ci in range(m):
                cl = clauses_opt[ci]
                for vi, pi in cl:
                    if assign[vi] == (1 if pi else 0):
                        break
                else:
                    solved = False
                    break
            if solved:
                return [bool(assign[i]) for i in range(n)]
        else:
            learnt_clause, bt_level = analyze_conflict(conf)
            while trail and level[trail[-1]] >= bt_level:
                vv = trail.pop()
                assign[vv] = -1
                level[vv] = 0
            if not trail:
                return None
        
        val2 = 1 - val
        assign[v] = val2
        trail.append(v)
        level[v] = len(trail)
        phase[v] = not phase[v]
        
        q = [(ci2, False) for ci2, p2 in var_clauses[v]]
        for li in range(len(learnt)):
            for v2, p2 in learnt[li]:
                if v2 == v:
                    q.append((li, True))
                    break
        
        ok, conf = propagate(q)
        
        if ok:
            solved = True
            for ci in range(m):
                cl = clauses_opt[ci]
                for vi, pi in cl:
                    if assign[vi] == (1 if pi else 0):
                        break
                else:
                    solved = False
                    break
            if solved:
                return [bool(assign[i]) for i in range(n)]
        else:
            learnt_clause, bt_level = analyze_conflict(conf)
            while trail and level[trail[-1]] >= bt_level:
                vv = trail.pop()
                assign[vv] = -1
                level[vv] = 0
            if not trail:
                return None

    return None


def _find_components(clauses_opt, n):
    """Find weakly-connected components of the clause-variable incidence graph."""
    # Build adjacency: variable -> variables sharing a clause
    adj = [[] for _ in range(n)]
    for cl in clauses_opt:
        vars_in_clause = [v for v, p in cl]
        for i in range(len(vars_in_clause)):
            for j in range(i + 1, len(vars_in_clause)):
                a, b = vars_in_clause[i], vars_in_clause[j]
                adj[a].append(b)
                adj[b].append(a)
    
    # BFS/DFS to find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in adj[v]:
                    if not visited[u]:
                        visited[u] = True
                        stack.append(u)
            components.append(comp)
    return components


def _solve_component(comp_vars, comp_clauses, comp_var_clauses, comp_n, comp_m, deadline, seed_offset=0):
    """Solve a single connected component using DPLL + SLS hybrid."""
    if comp_m == 0:
        # No clauses: any assignment works
        return [False] * comp_n
    
    # Try DPLL first with a shorter budget for small components
    dpll_time = min(5.0, max(1.0, comp_n * 0.05))
    dpll_deadline = time.time() + dpll_time
    
    result = _dpll_solve(comp_clauses, comp_var_clauses, comp_n, comp_m, dpll_deadline)
    if result is not None:
        # Verify
        for cl in comp_clauses:
            for v, p in cl:
                if result[v] == p:
                    break
            else:
                result = None
                break
        if result is not None:
            return result
    
    # SLS fallback for this component
    def count_unsat(ass):
        cnt = 0
        for cl in comp_clauses:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                cnt += 1
        return cnt
    
    def build_state(ass):
        sat = [0] * comp_m
        for ci, cl in enumerate(comp_clauses):
            s = 0
            for v, p in cl:
                if ass[v] == p:
                    s += 1
            sat[ci] = s
        return sat
    
    def flip_var(ass, var, sat):
        new = not ass[var]
        ass[var] = new
        for ci, p in comp_var_clauses[var]:
            if p == new:
                sat[ci] += 1
            else:
                sat[ci] -= 1
    
    def unit_propagate(ass, remaining_vars, n_remaining, sat):
        changed = True
        while changed and n_remaining > 0:
            changed = False
            for ci in range(comp_m):
                if sat[ci] > 0:
                    continue
                cl = comp_clauses[ci]
                unfixed = []
                for v, p in cl:
                    if v not in remaining_vars:
                        continue
                    unfixed.append((v, p))
                if len(unfixed) == 1:
                    v, p = unfixed[0]
                    if ass[v] != p:
                        flip_var(ass, v, sat)
                    remaining_vars.discard(v)
                    n_remaining -= 1
                    changed = True
        return True
    
    best_assignment = None
    best_unsat = float('inf')
    
    max_restarts = min(500, max(50, comp_n * 5))
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 131 + seed_offset)
        ass = [rng.random() < 0.5 for _ in range(comp_n)]
        
        # Phase bias
        if restart % 5 == 0:
            for v in range(comp_n):
                pos = sum(1 for ci, p in comp_var_clauses[v] if p)
                ass[v] = (pos >= len(comp_var_clauses[v]) - pos) if len(comp_var_clauses[v]) else False
        
        u = count_unsat(ass)
        if u == 0:
            return ass
        if u < best_unsat:
            best_unsat = u
            best_assignment = ass[:]
        
        # Unit propagation
        remaining_vars = set(range(comp_n))
        n_remaining = comp_n
        sat = build_state(ass)
        unit_propagate(ass, remaining_vars, n_remaining, sat)
        u = count_unsat(ass)
        if u == 0:
            return ass
        if u < best_unsat:
            best_unsat = u
            best_assignment = ass[:]
        
        mode = restart % 3
        noise = 0.2 + 0.1 * (restart % 4)
        steps = min(10000, max(1000, comp_n * 20))
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clause
            best_ci = -1
            for ci in range(comp_m):
                if sat[ci] == 0:
                    best_ci = ci
                    break
            if best_ci < 0:
                return ass
            
            clause = comp_clauses[best_ci]
            
            if rng.random() < noise:
                v, p = clause[rng.randrange(len(clause))]
            else:
                best_v = clause[0][0]
                best_cost = float('inf')
                for v, p in clause:
                    if ass[v] == p:
                        continue
                    cost = 0
                    for ci2, p2 in comp_var_clauses[v]:
                        if p2 == ass[v] and sat[ci2] == 1:
                            cost += 1
                    if mode == 1:
                        pos = sum(1 for ci2, p2 in comp_var_clauses[v] if p2)
                        phase_match = 1 if (p == (pos >= len(comp_var_clauses[v]) - pos)) else 0
                        cost_adj = cost - phase_match * 0.5
                    else:
                        cost_adj = cost
                    if cost_adj < best_cost:
                        best_cost = cost_adj
                        best_v = v
                v = best_v
            flip_var(ass, v, sat)
            
            all_sat = True
            for s in sat:
                if s == 0:
                    all_sat = False
                    break
            if all_sat:
                return ass
        
        u = count_unsat(ass)
        if u < best_unsat:
            best_unsat = u
            best_assignment = ass[:]
    
    return best_assignment


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 318  # safety margin before 330s kill

    # Precompute clauses as (var_idx 0-based, polarity bool)
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_opt)

    # Precompute var -> list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses_opt):
        for v, p in cl:
            var_clauses[v].append((ci, p))

    # Find connected components
    components = _find_components(clauses_opt, n)
    
    # If single component, use the original full-problem approach
    if len(components) == 1:
        # Use the full problem DPLL + SLS as before
        dpll_deadline = start_time + 20
        result = _dpll_solve(clauses_opt, var_clauses, n, m, dpll_deadline)
        if result is not None:
            for cl in clauses_opt:
                for v, p in cl:
                    if result[v] == p:
                        break
                else:
                    result = None
                    break
            if result is not None:
                return result
        
        # SLS on full problem
        def count_unsat(ass):
            cnt = 0
            for cl in clauses_opt:
                for v, p in cl:
                    if ass[v] == p:
                        break
                else:
                    cnt += 1
            return cnt
        
        def build_state(ass):
            sat = [0] * m
            for ci, cl in enumerate(clauses_opt):
                s = 0
                for v, p in cl:
                    if ass[v] == p:
                        s += 1
                sat[ci] = s
            return sat
        
        def flip_var(ass, var, sat):
            new = not ass[var]
            ass[var] = new
            for ci, p in var_clauses[var]:
                if p == new:
                    sat[ci] += 1
                else:
                    sat[ci] -= 1
        
        def unit_propagate(ass, remaining_vars, n_remaining, sat):
            changed = True
            while changed and n_remaining > 0:
                changed = False
                for ci in range(m):
                    if sat[ci] > 0:
                        continue
                    cl = clauses_opt[ci]
                    unfixed = []
                    for v, p in cl:
                        if v not in remaining_vars:
                            continue
                        unfixed.append((v, p))
                    if len(unfixed) == 1:
                        v, p = unfixed[0]
                        if ass[v] != p:
                            flip_var(ass, v, sat)
                        remaining_vars.discard(v)
                        n_remaining -= 1
                        changed = True
            return True
        
        best_assignment = None
        best_unsat = float('inf')
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = random.Random(restart * 7919 + 131)
            ass = [rng.random() < 0.5 for _ in range(n)]
            
            if restart % 5 == 0:
                for v in range(n):
                    pos = sum(1 for ci, p in var_clauses[v] if p)
                    ass[v] = (pos >= len(var_clauses[v]) - pos) if len(var_clauses[v]) else False
            
            u = count_unsat(ass)
            if u == 0:
                return ass
            if u < best_unsat:
                best_unsat = u
                best_assignment = ass[:]
            
            remaining_vars = set(range(n))
            n_remaining = n
            sat = build_state(ass)
            unit_propagate(ass, remaining_vars, n_remaining, sat)
            u = count_unsat(ass)
            if u == 0:
                return ass
            if u < best_unsat:
                best_unsat = u
                best_assignment = ass[:]
            
            mode = restart % 3
            noise = 0.2 + 0.1 * (restart % 4)
            steps = 20000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                best_ci = -1
                for ci in range(m):
                    if sat[ci] == 0:
                        best_ci = ci
                        break
                if best_ci < 0:
                    return ass
                
                clause = clauses_opt[best_ci]
                
                if rng.random() < noise:
                    v, p = clause[rng.randrange(len(clause))]
                else:
                    best_v = clause[0][0]
                    best_cost = float('inf')
                    for v, p in clause:
                        if ass[v] == p:
                            continue
                        cost = 0
                        for ci2, p2 in var_clauses[v]:
                            if p2 == ass[v] and sat[ci2] == 1:
                                cost += 1
                        if mode == 1:
                            pos = sum(1 for ci2, p2 in var_clauses[v] if p2)
                            phase_match = 1 if (p == (pos >= len(var_clauses[v]) - pos)) else 0
                            cost_adj = cost - phase_match * 0.5
                        else:
                            cost_adj = cost
                        if cost_adj < best_cost:
                            best_cost = cost_adj
                            best_v = v
                    v = best_v
                flip_var(ass, v, sat)
                
                all_sat = True
                for s in sat:
                    if s == 0:
                        all_sat = False
                        break
                if all_sat:
                    return ass
            
            u = count_unsat(ass)
            if u < best_unsat:
                best_unsat = u
                best_assignment = ass[:]
        
        if best_assignment is not None:
            return best_assignment
        return ass
    
    # Multiple components: solve each independently
    full_assignment = [False] * n
    
    # Sort components by size (largest first for time allocation)
    components.sort(key=len, reverse=True)
    
    # Allocate time proportionally to component size
    total_vars = sum(len(c) for c in components)
    remaining_time_budget = deadline - time.time() - 5  # leave 5s safety
    
    for comp_idx, comp_vars in enumerate(components):
        if time.time() > deadline - 2:
            break
        
        comp_n = len(comp_vars)
        
        # Build mapping: original var index -> local index
        var_map = {}
        for local_i, orig_v in enumerate(comp_vars):
            var_map[orig_v] = local_i
        
        # Find clauses belonging to this component
        comp_clause_set = set()
        for ci, cl in enumerate(clauses_opt):
            if all(v in var_map for v, p in cl):
                comp_clause_set.add(ci)
        
        comp_clauses_orig = [clauses_opt[ci] for ci in comp_clause_set]
        comp_m = len(comp_clauses_orig)
        
        if comp_m == 0:
            # No clauses in this component: all false is fine
            for v in comp_vars:
                full_assignment[v] = False
            continue
        
        # Remap to local indices
        comp_clauses = [[(var_map[v], p) for v, p in cl] for cl in comp_clauses_orig]
        
        # Build local var_clauses
        comp_var_clauses = [[] for _ in range(comp_n)]
        for ci, cl in enumerate(comp_clauses):
            for v, p in cl:
                comp_var_clauses[v].append((ci, p))
        
        # Allocate time: proportional to size, with minimum 1s
        comp_time = max(1.0, remaining_time_budget * comp_n / total_vars)
        comp_deadline = time.time() + comp_time
        
        comp_result = _solve_component(
            comp_vars, comp_clauses, comp_var_clauses, comp_n, comp_m, comp_deadline, seed_offset=comp_idx * 1000
        )
        
        # Map back to full assignment
        if comp_result is not None:
            for local_i, orig_v in enumerate(comp_vars):
                full_assignment[orig_v] = comp_result[local_i]
        else:
            # Fallback: random assignment
            rng_fb = random.Random(comp_idx * 42 + 7)
            for v in comp_vars:
                full_assignment[v] = rng_fb.random() < 0.5
    
    # Final verification
    for cl in clauses_opt:
        for v, p in cl:
            if full_assignment[v] == p:
                break
        else:
            # Component approach failed; fall back to global SLS
            # Try a quick global SLS with remaining time
            remaining = deadline - time.time()
            if remaining > 5:
                rng_global = random.Random(99999)
                ass = full_assignment[:]
                sat = [0] * m
                for ci, cl in enumerate(clauses_opt):
                    s = 0
                    for v, p in cl:
                        if ass[v] == p:
                            s += 1
                    sat[ci] = s
                
                for step in range(int(remaining * 10000)):
                    if time.time() > deadline - 1:
                        break
                    best_ci = -1
                    for ci in range(m):
                        if sat[ci] == 0:
                            best_ci = ci
                            break
                    if best_ci < 0:
                        return ass
                    clause = clauses_opt[best_ci]
                    v, p = clause[rng_global.randrange(len(clause))]
                    new = not ass[v]
                    ass[v] = new
                    for ci2, p2 in var_clauses[v]:
                        if p2 == new:
                            sat[ci2] += 1
                        else:
                            sat[ci2] -= 1
                    all_sat = True
                    for s in sat:
                        if s == 0:
                            all_sat = False
                            break
                    if all_sat:
                        return ass
                return ass
            return full_assignment
    
    return full_assignment
# EVOLVE-BLOCK-END
