def solve(instance):
    import time
    import random

    n = instance.get("n", 0)
    raw_clauses = instance.get("clauses", [])

    # Robust parsing
    clauses = []
    if isinstance(raw_clauses, (list, tuple)):
        if len(raw_clauses) > 0:
            if isinstance(raw_clauses[0], (list, tuple)):
                clauses = [list(c) for c in raw_clauses]
            else:
                clauses = [list(raw_clauses[i:i+3]) for i in range(0, len(raw_clauses), 3)]
        else:
            clauses = []
    elif isinstance(raw_clauses, int):
        clauses = []
    else:
        clauses = []

    if n == 0 or not clauses:
        return [False] * n

    deadline = time.time() + 290  # Safety margin

    # Pre-compute clauses as 0-based (idx, pos) tuples
    clauses_idx = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_idx)

    # Build variable -> clauses index for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_idx):
        for idx, _pos in c:
            if 0 <= idx < n:
                var_to_clauses[idx].append(ci)

    def sat_mask(assignment, ci):
        c = clauses_idx[ci]
        mask = 0
        for k, (idx, pos) in enumerate(c):
            if 0 <= idx < n and assignment[idx] == pos:
                mask |= (1 << k)
        return mask

    def evaluate(assignment):
        unsat = []
        for ci in range(m):
            if sat_mask(assignment, ci) == 0:
                unsat.append(ci)
        return unsat

    # 1. Try Gaussian Elimination (Linear/XOR assumption)
    try:
        if time.time() < deadline:
            A = [[0] * n for _ in range(m)]
            for i, clause in enumerate(clauses_idx):
                for idx, _pos in clause:
                    if 0 <= idx < n:
                        A[i][idx] = 1

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

    # 2. WalkSAT with incremental updates, adaptive noise, and restarts
    best_assignment = None
    best_unsat = m + 1

    # Track clause frequencies for smarter clause selection
    clause_freq = [0] * m
    # Track variable flip counts for diversity
    var_flip_count = [0] * n

    for attempt in range(300):
        if time.time() > deadline:
            break

        rng = random.Random(attempt * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        masks = [sat_mask(assignment, ci) for ci in range(m)]
        unsat = set(ci for ci in range(m) if masks[ci] == 0)

        if not unsat:
            return assignment

        if len(unsat) < best_unsat:
            best_unsat = len(unsat)
            best_assignment = assignment[:]

        # Adaptive noise: start lower, increase when stuck
        base_noise = 0.15
        noise = base_noise
        max_flips = 6000
        last_unsat_count = len(unsat)
        no_improve = 0
        consecutive_worse = 0

        for _flip in range(max_flips):
            if time.time() > deadline:
                break
            if not unsat:
                return assignment

            cur_unsat = len(unsat)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            if cur_unsat < last_unsat_count:
                no_improve = 0
                consecutive_worse = 0
                last_unsat_count = cur_unsat
            elif cur_unsat > last_unsat_count:
                consecutive_worse += 1
                no_improve += 1
            else:
                no_improve += 1

            # Increase noise when stuck
            if no_improve > 50:
                noise = min(0.6, noise + 0.03)
                no_improve = 0
            elif no_improve > 20:
                noise = min(0.5, noise + 0.01)

            # Select clause: weighted by frequency (more often unsat = more important)
            if len(unsat) > 1:
                weights = [clause_freq[ci] + 1 for ci in unsat]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                ci = list(unsat)[0]
                for c_idx, w in zip(unsat, weights):
                    cumulative += w
                    if r <= cumulative:
                        ci = c_idx
                        break
            else:
                ci = next(iter(unsat))

            c = clauses_idx[ci]

            # Score each literal flip by how many clauses it would satisfy
            best_score = -1
            best_k = -1
            for k, (idx, pos) in enumerate(c):
                if not (0 <= idx < n):
                    continue
                # Count how many unsatisfied clauses this flip would satisfy
                score = 0
                for cj in var_to_clauses[idx]:
                    if masks[cj] == 0:  # currently unsatisfied
                        # Check if flipping idx would satisfy this clause
                        cj_data = clauses_idx[cj]
                        for k2, (idx2, pos2) in enumerate(cj_data):
                            if idx2 == idx and pos2 == (not assignment[idx]):
                                # This flip would satisfy cj
                                score += 1
                                break
                # Add small random perturbation for diversity
                score += rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_k = k

            if best_k == -1:
                continue

            # Decide whether to use greedy or random flip
            if rng.random() > noise:
                k = best_k
            else:
                # Random flip: prefer unsatisfied literals in the clause
                unsat_positions = [k2 for k2, (idx2, pos2) in enumerate(c)
                                  if 0 <= idx2 < n and not (assignment[idx2] == pos2)]
                if unsat_positions and rng.random() > 0.3:
                    k = rng.choice(unsat_positions)
                else:
                    k = rng.randrange(3)

            idx, pos = c[k]

            if not (0 <= idx < n):
                continue

            # Update clause frequency before flipping
            clause_freq[ci] += 1
            var_flip_count[idx] += 1

            assignment[idx] = not assignment[idx]

            # Incrementally update masks for clauses touching idx
            for cj in var_to_clauses[idx]:
                new_mask = sat_mask(assignment, cj)
                old_mask = masks[cj]
                if old_mask == 0 and new_mask != 0:
                    unsat.discard(cj)
                elif old_mask != 0 and new_mask == 0:
                    unsat.add(cj)
                masks[cj] = new_mask

    if best_assignment is not None:
        return best_assignment

    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
