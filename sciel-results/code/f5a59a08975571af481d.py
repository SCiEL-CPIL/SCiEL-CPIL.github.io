# EVOLVE-BLOCK-START
import time
import random
import heapq
import sys

sys.setrecursionlimit(100000)


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 325  # safety margin before 330s kill

    # Precompute clauses as (var_idx 0-based, polarity bool)
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_opt)

    # Precompute var -> list of (clause_idx, polarity)
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses_opt):
        for v, p in cl:
            var_clauses[v].append((ci, p))

    # Weights per variable
    weights = [len(l) for l in var_clauses]

    def quick_check(ass):
        for cl in clauses_opt:
            for v, p in cl:
                if ass[v] == p:
                    break
            else:
                return False
        return True

    # ===================== DPLL with watched literals =====================
    def _dpll_solve(n, clauses_opt, var_clauses, deadline):
        """
        DPLL with:
        - Watched literals for efficient unit propagation
        - VSIDS variable ordering with heap
        - Trail-based backtracking
        """
        assignment = [None] * n
        phase = [False] * n  # phase save

        m = len(clauses_opt)
        
        # Watched literals: for each clause, track which 2 literals are "watched"
        # watches[var] = list of (clause_idx, polarity_of_other_watched_literal)
        watches = [[] for _ in range(n)]
        
        # Initialize watches: for each clause, watch the first two literals
        for ci, cl in enumerate(clauses_opt):
            if len(cl) >= 2:
                # Watch first two literals
                v1, p1 = cl[0]
                v2, p2 = cl[1]
                watches[v1].append((ci, p2))  # clause ci, other watch is p2
                watches[v2].append((ci, p1))  # clause ci, other watch is p1
            elif len(cl) == 1:
                # Unit clause
                v1, p1 = cl[0]
                watches[v1].append((ci, None))  # No other watch
        
        # Activity tracking
        activity = [1.0] * n
        clock = 0
        decay = 0.95
        
        # Heap for variable selection: (negative_activity, var)
        var_heap = [(-1.0, v) for v in range(n)]
        heapq.heapify(var_heap)
        in_heap = [True] * n
        
        def bump(v):
            nonlocal clock
            activity[v] = activity[v] * 2.0 + 1.0
            clock += 1
            if in_heap[v]:
                # Push new entry, old one will be ignored
                heapq.heappush(var_heap, (-activity[v], v))
        
        def decay_activities():
            nonlocal clock
            if clock > 1000:
                for v in range(n):
                    activity[v] *= decay
                clock = 0
        
        # Trail for backtracking
        trail = []
        
        def propagate():
            """
            Unit propagation using watched literals.
            Returns list of (var, value) assignments made, or None on conflict.
            """
            made = []
            queue = []
            
            # Find initial unit clauses
            for ci, cl in enumerate(clauses_opt):
                unassigned = [(v, p) for v, p in cl if assignment[v] is None]
                if len(unassigned) == 1:
                    v, p = unassigned[0]
                    queue.append((v, p))
            
            while queue:
                v, p = queue.pop()
                if assignment[v] is not None:
                    if assignment[v] != p:
                        return None  # conflict
                    continue
                
                # Assign v = p
                assignment[v] = p
                made.append((v, p))
                phase[v] = p
                trail.append(v)
                
                # Process watches for variable v
                new_watches = []
                for ci, other_p in watches[v]:
                    cl = clauses_opt[ci]
                    
                    # Check if the other watched literal is satisfied
                    if other_p is not None:
                        # Find the variable of the other watched literal
                        other_v = None
                        for vv, pp in cl:
                            if vv != v and pp == other_p:
                                other_v = vv
                                break
                        if other_v is not None and assignment[other_v] == other_p:
                            # Other literal satisfied, keep this watch
                            new_watches.append((ci, other_p))
                            continue
                    
                    # Check if clause is satisfied by any assigned literal
                    sat = False
                    for vv, pp in cl:
                        if assignment[vv] is not None and assignment[vv] == pp:
                            sat = True
                            break
                    
                    if sat:
                        # Clause satisfied, no need to watch
                        continue
                    
                    # Find unassigned literals to watch
                    unassigned = [(vv, pp) for vv, pp in cl if assignment[vv] is None]
                    
                    if len(unassigned) == 0:
                        return None  # Conflict: all literals false
                    elif len(unassigned) == 1:
                        # Unit clause
                        uv, up = unassigned[0]
                        queue.append((uv, up))
                        # Set up watch on this literal
                        new_watches.append((ci, None))
                        # Remove this watch from the unassigned variable's list later
                    else:
                        # Pick first unassigned literal to watch
                        uv, up = unassigned[0]
                        new_watches.append((ci, up))
                        # Remove watch from uv's list
                        # We'll handle this by rebuilding watches below
                
                # Rebuild watches[v]
                watches[v] = new_watches
            
            return made
        
        def select_variable():
            """Select variable using VSIDS with heap."""
            while var_heap:
                neg_act, v = heapq.heappop(var_heap)
                if not in_heap[v]:
                    continue
                if assignment[v] is None:
                    in_heap[v] = False
                    return v
            return -1
        
        def all_satisfied():
            """Check if all clauses are satisfied."""
            for ci, cl in enumerate(clauses_opt):
                for v, p in cl:
                    if assignment[v] == p:
                        break
                else:
                    return False
            return True
        
        def backtrack_to(trail_len):
            """Backtrack to given trail length."""
            while len(trail) > trail_len:
                v = trail.pop()
                assignment[v] = None
                in_heap[v] = True
                heapq.heappush(var_heap, (-activity[v], v))
        
        # DPLL search
        def dpll(depth):
            nonlocal clock
            
            if time.time() > deadline:
                return None
            
            # Unit propagation
            made = propagate()
            if made is None:
                return None
            
            if all_satisfied():
                return assignment[:]
            
            v = select_variable()
            if v < 0:
                if all_satisfied():
                    return assignment[:]
                return None
            
            # Phase-based ordering
            if phase[v]:
                order = [True, False]
            else:
                order = [False, True]
            
            for val in order:
                if time.time() > deadline:
                    return None
                
                old_trail_len = len(trail)
                
                assignment[v] = val
                phase[v] = val
                trail.append(v)
                in_heap[v] = False
                
                bump(v)
                if clock % 100 == 0:
                    decay_activities()
                
                result = dpll(depth + 1)
                if result is not None:
                    return result
                
                # Backtrack
                backtrack_to(old_trail_len)
                assignment[v] = None
            
            return None
        
        # Run
        result = dpll(0)
        if result is not None:
            return [bool(x) if x is not None else False for x in result]
        return None

    # ===================== WalkSAT / SLS =====================
    def _sls_solve(n, clauses_opt, var_clauses, weights, deadline):
        """
        WalkSAT with:
        - Min-break heuristic with noise
        - Phase saving
        - Multiple restarts with increasing noise
        - Best-so-far tracking
        """
        best_assignment = None
        best_unsat = float('inf')

        def count_unsat(ass):
            cnt = 0
            for cl in clauses_opt:
                for v, p in cl:
                    if ass[v] == p:
                        break
                else:
                    cnt += 1
            return cnt

        def build_sat(ass):
            sat = [0] * m
            for ci, cl in enumerate(clauses_opt):
                s = 0
                for v, p in cl:
                    if ass[v] == p:
                        s += 1
                sat[ci] = s
            return sat

        def flip_var(ass, var, sat):
            old = ass[var]
            new = not old
            ass[var] = new
            for ci, p in var_clauses[var]:
                if p == new:
                    sat[ci] += 1
                else:
                    sat[ci] -= 1

        rng_master = random.Random(42)
        max_restarts = 500
        phase = [False] * n

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            rng = random.Random(restart * 7919 + 131 + int(time.time() * 1000))

            # Initialize
            if restart == 0:
                ass = [rng.random() < 0.5 for _ in range(n)]
            elif restart % 10 == 0 and best_assignment is not None:
                # Perturb best
                ass = best_assignment[:]
                for _ in range(n // 10):
                    ass[rng.randrange(n)] = not ass[rng.randrange(n)]
            else:
                ass = [rng.random() < 0.5 for _ in range(n)]

            # Phase bias
            for v in range(n):
                pos = sum(1 for ci, p in var_clauses[v] if p)
                if pos > len(var_clauses[v]) - pos:
                    phase[v] = True

            u = count_unsat(ass)
            if u == 0:
                return ass
            if u < best_unsat:
                best_unsat = u
                best_assignment = ass[:]

            sat = build_sat(ass)

            # Noise increases with restarts
            noise = 0.15 + 0.05 * min(restart // 20, 4)
            steps = 50000

            for step in range(steps):
                if time.time() > deadline - 5:
                    break

                # Find unsat clause
                best_ci = -1
                for _ in range(3):  # try 3 random clauses
                    ci = rng.randrange(m)
                    if sat[ci] == 0:
                        best_ci = ci
                        break
                if best_ci < 0:
                    # Full check
                    if all(s >= 1 for s in sat):
                        return ass
                    # Find one
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
                    # Min-break
                    best_v = clause[0][0]
                    best_cost = float('inf')
                    for v, p in clause:
                        if ass[v] == p:
                            continue
                        cost = 0
                        for ci2, p2 in var_clauses[v]:
                            if p2 == ass[v] and sat[ci2] == 1:
                                cost += 1
                        # Tie-break by weight
                        if cost < best_cost or (cost == best_cost and weights[v] > weights[best_v]):
                            best_cost = cost
                            best_v = v
                    v = best_v

                flip_var(ass, v, sat)
                phase[v] = ass[v]

                # Quick check
                unsat_count = 0
                for s in sat:
                    if s == 0:
                        unsat_count += 1
                if unsat_count == 0:
                    return ass
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = ass[:]

        if best_assignment is not None:
            return best_assignment
        return ass

    # ===================== Main solve =====================
    # Phase 0: Try DPLL with time limit
    dpll_deadline = start_time + 30
    dpll_result = _dpll_solve(n, clauses_opt, var_clauses, dpll_deadline)
    if dpll_result is not None and quick_check(dpll_result):
        return dpll_result

    # Phase 1: SLS
    sls_deadline = deadline
    sls_result = _sls_solve(n, clauses_opt, var_clauses, weights, sls_deadline)
    if sls_result is not None and quick_check(sls_result):
        return sls_result

    # Phase 2: If DPLL didn't finish, try again with remaining time
    remaining = deadline - time.time()
    if remaining > 10:
        dpll_result2 = _dpll_solve(n, clauses_opt, var_clauses, start_time + remaining)
        if dpll_result2 is not None and quick_check(dpll_result2):
            return dpll_result2

    # Return best effort
    if sls_result is not None:
        return sls_result
    return [False] * n

# EVOLVE-BLOCK-END
