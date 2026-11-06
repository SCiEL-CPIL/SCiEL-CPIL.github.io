def solve(instance):
        import time
        import random
        
        n = instance.get("n", 0)
        raw_clauses = instance.get("clauses", [])
        
        # Robust parsing to avoid TypeError
        clauses = []
        if isinstance(raw_clauses, (list, tuple)):
            if len(raw_clauses) > 0:
                if isinstance(raw_clauses[0], (list, tuple)):
                    clauses = [list(c) for c in raw_clauses]
                else:
                    # Assume flat list of 3-literal clauses
                    clauses = [list(raw_clauses[i:i+3]) for i in range(0, len(raw_clauses), 3)]
            else:
                clauses = []
        elif isinstance(raw_clauses, int):
            # If clauses is an int, we can't proceed with standard parsing, 
            # but we avoid the crash. Return random assignment.
            clauses = []
        else:
            clauses = []

        if n == 0 or not clauses:
            return [False] * n

        deadline = time.time() + 290 # Safety margin
        m = len(clauses)

        def evaluate(assignment):
            unsat = []
            for ci, clause in enumerate(clauses):
                sat = False
                for lit in clause:
                    v = abs(lit) - 1
                    if 0 <= v < n:
                        if assignment[v] == (lit > 0):
                            sat = True
                            break
                if not sat:
                    unsat.append(ci)
            return unsat

        # Precompute for fast delta evaluation
        # var_to_clauses[v] = list of clause indices containing variable v
        # clause_satisfaction_count[ci] = number of satisfied literals in clause ci
        var_to_clauses = [[] for _ in range(n)]
        for ci, clause in enumerate(clauses):
            for lit in clause:
                v = abs(lit) - 1
                if 0 <= v < n:
                    var_to_clauses[v].append(ci)

        def init_sat_counts(assignment):
            """Compute number of satisfied literals per clause."""
            sat_counts = [0] * m
            for ci, clause in enumerate(clauses):
                cnt = 0
                for lit in clause:
                    v = abs(lit) - 1
                    if 0 <= v < n and assignment[v] == (lit > 0):
                        cnt += 1
                sat_counts[ci] = cnt
            return sat_counts

        def get_unsat_from_counts(sat_counts):
            return [ci for ci in range(m) if sat_counts[ci] == 0]

        # 1. Try Gaussian Elimination (Linear/XOR assumption)
        try:
            if time.time() < deadline:
                A = [[0] * n for _ in range(m)]
                for i, clause in enumerate(clauses):
                    for lit in clause:
                        v = abs(lit) - 1
                        if 0 <= v < n:
                            A[i][v] = 1
                
                aug = [row[:] + [1] for row in A]
                
                rank = 0
                for col in range(n):
                    pivot = -1
                    for r in range(rank, m):
                        if aug[r][col] == 1:
                            pivot = r
                            break
                    if pivot == -1:
                        continue
                    aug[rank], aug[pivot] = aug[pivot], aug[rank]
                    for r in range(m):
                        if r != rank and aug[r][col] == 1:
                            for c in range(n + 1):
                                aug[r][c] ^= aug[rank][c]
                    rank += 1
                    if rank == n:
                        break

                consistent = True
                for r in range(rank, m):
                    if aug[r][n] == 1:
                        consistent = False
                        break
                
                if consistent:
                    assignment = [False] * n
                    for r in range(rank):
                        pivot_col = -1
                        for c in range(n):
                            if aug[r][c] == 1:
                                pivot_col = c
                                break
                        if pivot_col != -1:
                            assignment[pivot_col] = bool(aug[r][n])
                    
                    if not evaluate(assignment):
                        return assignment
        except Exception:
            pass

        # 2. Fallback: WalkSAT with noise heuristic and fast delta evaluation
        best_assignment = None
        best_unsat_count = float('inf')
        
        # Try multiple noise values to explore different behaviors
        noise_values = [0.3, 0.5, 0.2, 0.7, 0.4, 0.6, 0.1, 0.8]
        
        total_attempts = 200
        for attempt in range(total_attempts):
            if time.time() > deadline:
                break
            
            noise = noise_values[attempt % len(noise_values)]
            rng = random.Random(attempt * 7919 + 13)
            assignment = [rng.random() < 0.5 for _ in range(n)]
            
            sat_counts = init_sat_counts(assignment)
            unsat = get_unsat_from_counts(sat_counts)
            if not unsat:
                return assignment
            
            current_unsat = len(unsat)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment.copy()
            
            # WalkSAT loop with noise-based flip heuristic
            max_steps = 3000
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                if not unsat:
                    return assignment
                    
                # Pick random unsat clause
                ci = rng.choice(unsat)
                clause = clauses[ci]
                
                # Noise heuristic:
                # With probability (1-noise), pick a literal that is currently FALSE 
                # in this clause (all literals are false since clause is unsat),
                # but prefer ones that satisfy more other clauses (greedy).
                # With probability noise, pick a random literal (random walk).
                
                if rng.random() < noise:
                    # Random flip
                    lit = rng.choice(clause)
                else:
                    # Greedy: pick the literal whose flip satisfies the most other clauses
                    best_score = -1
                    best_lit = clause[0]
                    for lit in clause:
                        v = abs(lit) - 1
                        # Count how many other clauses this flip would satisfy
                        score = 0
                        for oci in var_to_clauses[v]:
                            if oci == ci:
                                continue
                            # Check if flipping v would satisfy this clause
                            # The literal in oci for variable v
                            for olit in clauses[oci]:
                                ov = abs(olit) - 1
                                if ov == v:
                                    # After flip, assignment[v] is inverted
                                    new_val = not assignment[v]
                                    if new_val == (olit > 0):
                                        score += 1
                                    break
                        if score > best_score:
                            best_score = score
                            best_lit = lit
                    lit = best_lit
                
                v = abs(lit) - 1
                if v < 0 or v >= n:
                    # Invalid literal, skip
                    continue
                    
                old_val = assignment[v]
                new_val = not old_val
                assignment[v] = new_val
                
                # Update sat_counts for affected clauses
                for oci in var_to_clauses[v]:
                    # Determine how this flip affects clause oci
                    # Find the literal in oci corresponding to variable v
                    for olit in clauses[oci]:
                        ov = abs(olit) - 1
                        if ov == v:
                            if new_val == (olit > 0):
                                # Now satisfied
                                sat_counts[oci] += 1
                            else:
                                # Now unsatisfied
                                sat_counts[oci] -= 1
                            break
                
                unsat = get_unsat_from_counts(sat_counts)
                cur_unsat = len(unsat)
                if cur_unsat < best_unsat_count:
                    best_unsat_count = cur_unsat
                    best_assignment = assignment.copy()

        if best_assignment is not None:
            return best_assignment
            
        # Final fallback
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]
