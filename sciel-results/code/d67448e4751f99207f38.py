import random
import time
import math
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    # 300s limit, keep 10s margin for safety
    deadline = time.time() + 290 

    if m == 0:
        return [False] * n

    # Precompute variable occurrences as two parallel lists for speed
    # var_occ_clauses[var] = list of clause indices
    # var_occ_pos[var] = list of booleans (True if positive literal)
    var_occ_clauses = [[] for _ in range(n)]
    var_occ_pos = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occ_clauses[var].append(ci)
            var_occ_pos[var].append(lit > 0)

    def count_unsatisfied(assignment):
        count = 0
        for c in clauses:
            sat = False
            for lit in c:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def polarity_init():
        """Compute polarity-based initial assignment: for each variable, count positive vs negative occurrences.
        Assign True if more positive occurrences (planted instances tend to have biased polarity)."""
        pos_count = [0] * n
        neg_count = [0] * n
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if lit > 0:
                    pos_count[var] += 1
                else:
                    neg_count[var] += 1
        polarity = []
        for var in range(n):
            if pos_count[var] > neg_count[var]:
                polarity.append(True)
            elif neg_count[var] > pos_count[var]:
                polarity.append(False)
            else:
                polarity.append(True)
        return polarity

    def dpll_with_unit_prop(seed, time_budget=10):
        """DPLL with unit propagation, VSIDS-style variable ordering, and randomized restarts."""
        rng = random.Random(seed)
        start_time = time.time()

        # Compute polarity for better initial assignments
        pol = polarity_init()

        def propagate(clauses_list, assignment):
            """Unit propagation. Returns (new_assignment, new_clauses_list, contradiction)"""
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses_list:
                    sat = False
                    unit_lit = None
                    open_lits = []
                    for lit in clause:
                        var = abs(lit) - 1
                        if var in assignment:
                            val = assignment[var]
                            if (lit > 0 and val) or (lit < 0 and not val):
                                sat = True
                                break
                        else:
                            open_lits.append(lit)

                    if sat:
                        continue
                    elif len(open_lits) == 0:
                        return assignment, clauses_list, True
                    elif len(open_lits) == 1:
                        unit_lit = open_lits[0]
                        var = abs(unit_lit) - 1
                        val = unit_lit > 0
                        if var in assignment:
                            if assignment[var] != val:
                                return assignment, clauses_list, True
                        else:
                            assignment[var] = val
                            changed = True
                        continue
                    else:
                        new_clauses.append(open_lits)

            return assignment, new_clauses, False

        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break

            assignment = {}
            # Use polarity-based initialization with some randomization
            num_initial = min(n // 4, 40)
            for var in rng.sample(range(n), num_initial):
                # 70% chance follow polarity, 30% random
                if rng.random() < 0.7:
                    assignment[var] = pol[var]
                else:
                    assignment[var] = rng.random() < 0.5

            clauses_copy = [list(c) for c in clauses]
            assignment, clauses_copy, contradiction = propagate(clauses_copy, assignment)

            if contradiction:
                continue

            if not clauses_copy:
                result = [False] * n
                for var in range(n):
                    result[var] = assignment.get(var, pol[var])
                return result

            # If DPLL doesn't finish, return a partial assignment filled with polarity
            result = [assignment.get(var, pol[var]) for var in range(n)]
            unsat_count = count_unsatisfied(result)
            if unsat_count == 0:
                return result
            return result

        # Fallback: polarity-based assignment
        return pol[:]

    def local_dpll_refine(assignment, time_budget=5.0):
        """
        Local DPLL refinement: fix variables in unsatisfied clauses, run unit propagation,
        then backtrack if contradiction. Optimized with incremental sat_count updates.
        """
        rng = random.Random(12345)
        start_time = time.time()
        
        # Identify unsatisfied clauses
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1
        
        unsat_indices = [i for i in range(m) if sat_count[i] == 0]
        
        if not unsat_indices:
            return assignment

        best_assignment = assignment[:]
        best_unsat = len(unsat_indices)
        pol = polarity_init()

        max_rounds = 50
        for rnd in range(max_rounds):
            if time.time() - start_time > time_budget:
                break
            if not unsat_indices:
                break

            # Pick a random unsatisfied clause
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = clauses[ci]

            # Try flipping each variable in the clause to satisfy it
            found_solution = False
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0

                if assignment[var] == want_val:
                    continue

                # Save current state
                old_val = assignment[var]
                assignment[var] = want_val

                # Run unit propagation on the modified assignment with incremental updates
                prop_assignment = assignment.copy()
                prop_sat_count = sat_count[:]

                # Update sat_count for the flipped variable
                occ_clauses = var_occ_clauses[var]
                occ_pos = var_occ_pos[var]
                for idx in range(len(occ_clauses)):
                    oc = occ_clauses[idx]
                    lit_val = occ_pos[idx]
                    if lit_val == want_val:
                        prop_sat_count[oc] += 1
                    else:
                        prop_sat_count[oc] -= 1

                # Unit propagation
                contradiction = False
                changed = True
                while changed and not contradiction:
                    changed = False
                    for pi, pclause in enumerate(clauses):
                        if prop_sat_count[pi] > 0:
                            continue
                        open_lits = []
                        for plit in pclause:
                            pvar = abs(plit) - 1
                            if pvar not in prop_assignment:
                                open_lits.append(plit)
                        if len(open_lits) == 0:
                            contradiction = True
                            break
                        elif len(open_lits) == 1:
                            ulit = open_lits[0]
                            uvar = abs(ulit) - 1
                            uval = ulit > 0
                            if uvar in prop_assignment:
                                if prop_assignment[uvar] != uval:
                                    contradiction = True
                                    break
                            else:
                                prop_assignment[uvar] = uval
                                changed = True
                                # Update sat counts incrementally
                                u_occ_clauses = var_occ_clauses[uvar]
                                u_occ_pos = var_occ_pos[uvar]
                                for uidx in range(len(u_occ_clauses)):
                                    uoc = u_occ_clauses[uidx]
                                    ulit_val = u_occ_pos[uidx]
                                    if ulit_val == uval:
                                        prop_sat_count[uoc] += 1
                                    else:
                                        prop_sat_count[uoc] -= 1

                if contradiction:
                    # Backtrack
                    assignment[var] = old_val
                    continue

                # Count unsatisfied in propagated assignment
                new_unsat = sum(1 for c in prop_sat_count if c == 0)
                if new_unsat < best_unsat:
                    best_unsat = new_unsat
                    # Fill in unassigned variables using polarity
                    for v in range(n):
                        if v not in prop_assignment:
                            prop_assignment[v] = pol[v]
                    best_assignment = prop_assignment[:]
                    found_solution = True
                    break
                else:
                    # Backtrack
                    assignment[var] = old_val

            if not found_solution:
                # No improvement found this round, try different clause
                continue

        return best_assignment

    def walksat(seed, init_assignment=None, noise=0.3, max_steps=500000):
        """
        Optimized WalkSAT with polarity-aware initialization and adaptive noise.
        Maintains sat_count for each clause.
        Flips variable in a random unsatisfied clause.
        Variable selection: with prob 'noise' random, else max gain.
        """
        rng = random.Random(seed)
        pol = polarity_init()
        
        if init_assignment is not None:
            assignment = init_assignment[:]
        else:
            # Polarity-based initialization with slight randomization
            assignment = [pol[i] if rng.random() < 0.8 else (rng.random() < 0.5) for i in range(n)]

        # Initialize satisfaction counts
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1
        
        unsat_set = set(i for i in range(m) if sat_count[i] == 0)

        if not unsat_set:
            return assignment

        start_time = time.time()
        prev_unsat = len(unsat_set)
        stall_counter = 0
        best_unsat = len(unsat_set)
        best_assignment = assignment[:]

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if not unsat_set:
                break

            current_unsat = len(unsat_set)
            if current_unsat >= prev_unsat:
                stall_counter += 1
            else:
                stall_counter = 0
            prev_unsat = current_unsat

            # Adaptive noise: increase when stalled, but cap it
            adaptive_noise = noise + min(0.5, stall_counter * 0.015)

            # Select a random unsatisfied clause
            unsat_list = list(unsat_set)
            ci = unsat_list[rng.randrange(len(unsat_list))]
            clause = clauses[ci]

            # Evaluate flips for variables in this clause
            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                gain = 0
                loss = 0
                
                # Check impact on other clauses using precomputed parallel lists
                occ_clauses = var_occ_clauses[var]
                occ_pos = var_occ_pos[var]
                for idx in range(len(occ_clauses)):
                    oc = occ_clauses[idx]
                    if oc == ci:
                        continue
                    lit_val = occ_pos[idx]
                    if lit_val == want_val:
                        # Flipping var to want_val satisfies this occurrence in clause oc
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        # Flipping var to want_val unsatisfies this occurrence in clause oc
                        if sat_count[oc] == 1:
                            loss += 1
                
                score = gain - loss
                candidates.append((score, var, want_val))

            # Choose flip with slight preference for polarity-consistent flips
            if rng.random() < adaptive_noise:
                _, var, want_val = rng.choice(candidates)
            else:
                # Pick max score, break ties by polarity preference
                best_score = max(c[0] for c in candidates)
                best_candidates = [c for c in candidates if c[0] == best_score]
                if len(best_candidates) > 1:
                    # Prefer polarity-consistent flip
                    pol_consistent = [c for c in best_candidates if c[2] == pol[c[1]]]
                    if pol_consistent:
                        var, want_val = pol_consistent[0][1], pol_consistent[0][2]
                    else:
                        var, want_val = best_candidates[0][1], best_candidates[0][2]
                else:
                    var, want_val = best_candidates[0][1], best_candidates[0][2]

            old_val = assignment[var]
            if old_val == want_val:
                continue

            assignment[var] = want_val

            # Update sat_counts using precomputed parallel lists
            occ_clauses = var_occ_clauses[var]
            occ_pos = var_occ_pos[var]
            for idx in range(len(occ_clauses)):
                oc = occ_clauses[idx]
                lit_val = occ_pos[idx]
                if lit_val == want_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)

            if not unsat_set:
                return assignment
            
            # Track best
            if len(unsat_set) < best_unsat:
                best_unsat = len(unsat_set)
                best_assignment = assignment[:]

        return best_assignment

    def targeted_repair(assignment, max_rounds=1000):
        """
        Targeted repair: only flip variables involved in currently unsatisfied clauses.
        Uses greedy min-conflict approach with limited restarts.
        """
        rng = random.Random(42)
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1
        
        unsat_set = set(i for i in range(m) if sat_count[i] == 0)
        
        if not unsat_set:
            return assignment

        best_unsat = len(unsat_set)
        best_assignment = assignment[:]

        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            if not unsat_set:
                break

            # Pick a random unsatisfied clause
            unsat_list = list(unsat_set)
            ci = unsat_list[rng.randrange(len(unsat_list))]
            clause = clauses[ci]

            # Find best flip among variables in this clause
            best_score = -999
            best_var = -1
            best_val = False

            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                gain = 0
                loss = 0

                occ_clauses = var_occ_clauses[var]
                occ_pos = var_occ_pos[var]
                for idx in range(len(occ_clauses)):
                    oc = occ_clauses[idx]
                    if oc == ci:
                        continue
                    lit_val = occ_pos[idx]
                    if lit_val == want_val:
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        if sat_count[oc] == 1:
                            loss += 1

                score = gain - loss
                if score > best_score:
                    best_score = score
                    best_var = var
                    best_val = want_val

            if best_var < 0:
                break

            old_val = assignment[best_var]
            if old_val == best_val:
                continue

            assignment[best_var] = best_val

            # Update sat_counts
            occ_clauses = var_occ_clauses[best_var]
            occ_pos = var_occ_pos[best_var]
            for idx in range(len(occ_clauses)):
                oc = occ_clauses[idx]
                lit_val = occ_pos[idx]
                if lit_val == best_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)

            if not unsat_set:
                return assignment

            if len(unsat_set) < best_unsat:
                best_unsat = len(unsat_set)
                best_assignment = assignment[:]

        return best_assignment

    # Phase 1: DPLL with polarity-based initialization
    result = dpll_with_unit_prop(0, time_budget=30)
    unsat_count = count_unsatisfied(result)
    if unsat_count == 0:
        return result

    best_assignment = result
    best_unsat = unsat_count

    # Phase 1b: Try polarity-based assignment directly as a starting point
    pol = polarity_init()
    pol_unsat = count_unsatisfied(pol)
    if pol_unsat == 0:
        return pol
    if pol_unsat < best_unsat:
        best_unsat = pol_unsat
        best_assignment = pol[:]

    # Phase 2: WalkSAT from best initial result
    if time.time() < deadline - 15:
        result = walksat(0, init_assignment=best_assignment, noise=0.3, max_steps=500000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 2b: Local DPLL refinement after WalkSAT
    if time.time() < deadline - 12 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=5.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 2c: Targeted repair on best assignment
    if time.time() < deadline - 10 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=5000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 3: More DPLL restarts with local refinement
    for seed in range(1, 20):
        if time.time() > deadline - 8:
            break
        result = dpll_with_unit_prop(seed, time_budget=5)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result
            # Local DPLL refinement on improved solution
            if time.time() < deadline - 6 and best_unsat > 0:
                refined = local_dpll_refine(best_assignment, time_budget=3.0)
                refined_unsat = count_unsatisfied(refined)
                if refined_unsat < best_unsat:
                    best_unsat = refined_unsat
                    best_assignment = refined

    # Phase 4: Multiple WalkSAT restarts with local DPLL refinement
    for seed in range(1, 50):
        if time.time() > deadline - 5:
            break
        if seed % 2 == 0:
            result = walksat(seed, init_assignment=best_assignment, noise=0.2, max_steps=300000)
        else:
            result = walksat(seed, init_assignment=None, noise=0.3, max_steps=300000)
        
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result
            # Local DPLL refinement on improved solution
            if time.time() < deadline - 4 and best_unsat > 0:
                refined = local_dpll_refine(best_assignment, time_budget=2.0)
                refined_unsat = count_unsatisfied(refined)
                if refined_unsat < best_unsat:
                    best_unsat = refined_unsat
                    best_assignment = refined

    # Phase 5: Final targeted repair polish
    if time.time() < deadline - 2 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=10000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 5b: Final local DPLL refinement
    if time.time() < deadline - 1.5 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=2.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 6: Final WalkSAT polish
    if time.time() < deadline - 1 and best_unsat > 0:
        result = walksat(999, init_assignment=best_assignment, noise=0.1, max_steps=200000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 7: Final local DPLL polish
    if time.time() < deadline - 0.5 and best_unsat > 0:
        result = local_dpll_refine(best_assignment, time_budget=1.0)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment
