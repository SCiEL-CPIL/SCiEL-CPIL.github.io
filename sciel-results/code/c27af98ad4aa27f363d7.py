def construct():
    import time
    import random
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill
    
    N = 180
    
    # Precompute all forbidden triples: (a, b, c) with a < b < c, b*c = a*(b+c)
    # For each element x, store list of (other1, other2) pairs that form a forbidden triple with x
    constraint = [[] for _ in range(N + 1)]
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    # a < b < c always holds
                    constraint[a].append((b, c))
                    constraint[b].append((a, c))
                    constraint[c].append((a, b))
    
    # Also build a set for fast lookup of forbidden triples
    forbidden_triples_set = set()
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    forbidden_triples_set.add((a, b, c))
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a forbidden triple."""
        for (b, c) in constraint[x]:
            if b in S and c in S:
                return False
        return True
    
    def count_blocked_pairs(x, S):
        """Count how many constraint pairs of x are both in S."""
        cnt = 0
        for (b, c) in constraint[x]:
            if b in S and c in S:
                cnt += 1
        return cnt
    
    def get_blockers(x, S):
        """Get all elements in S that block x (i.e., pairs (b,c) where both b,c in S)."""
        blockers = set()
        for (b, c) in constraint[x]:
            if b in S and c in S:
                blockers.add(b)
                blockers.add(c)
        return blockers
    
    def full_validate(S):
        """Full validation: check no forbidden triple is fully in S."""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (S_list[i], S_list[j], S_list[k]) in forbidden_triples_set:
                        return False
        return True
    
    def greedy_build(seed_elements=None):
        """Build a set greedily, optionally starting with seed elements."""
        S = set()
        if seed_elements:
            for e in seed_elements:
                if can_add(e, S):
                    S.add(e)
        
        # Sort candidates by number of constraints (fewer constraints = easier to add)
        candidates = list(range(1, N + 1))
        candidates.sort(key=lambda x: (len(constraint[x]), -x))
        
        for x in candidates:
            if x in S:
                continue
            if can_add(x, S):
                S.add(x)
        
        return S
    
    def repair_local_search(S, max_iter=50000):
        """Repair-based local search: try to add blocked elements by removing blockers."""
        S = set(S)
        best = set(S)
        best_size = len(S)
        iter_count = 0
        
        while iter_count < max_iter:
            if time.time() - start_time > DEADLINE:
                break
            iter_count += 1
            
            # Find blocked elements (elements not in S that are blocked)
            blocked = []
            for x in range(1, N + 1):
                if x not in S:
                    blockers = get_blockers(x, S)
                    if blockers:
                        blocked.append((x, blockers))
            
            if not blocked:
                break
            
            # For each blocked element, try removing one blocker to add it
            # Prefer removing the larger blocker (large elements have fewer constraints)
            found_improvement = False
            random.shuffle(blocked)
            
            for x, blockers in blocked:
                if time.time() - start_time > DEADLINE:
                    break
                
                # Sort blockers: prefer removing larger elements (less constrained)
                sorted_blockers = sorted(blockers, key=lambda b: -b)
                
                for rem in sorted_blockers:
                    trial = S - {rem}
                    if can_add(x, trial):
                        # Check: is removing rem and adding x a net gain or neutral?
                        # We gain x, lose rem. Net 0. But maybe x allows more additions later.
                        # Accept if it doesn't decrease size (it's equal) and x < rem (prefer smaller)
                        # Or just accept to explore
                        new_S = trial | {x}
                        if len(new_S) >= len(S):
                            # Verify no triple created
                            if full_validate(new_S):
                                S = new_S
                                found_improvement = True
                                if len(S) > best_size:
                                    best = set(S)
                                    best_size = len(S)
                                break
            
            if not found_improvement:
                # Try single-element swaps: remove one element, add another
                improved = False
                elements_list = list(S)
                random.shuffle(elements_list)
                
                for rem in elements_list:
                    if time.time() - start_time > DEADLINE:
                        break
                    
                    trial = S - {rem}
                    
                    # Try adding any element not in trial
                    candidates_to_add = [x for x in range(1, N + 1) if x not in trial]
                    random.shuffle(candidates_to_add)
                    
                    for add in candidates_to_add:
                        if can_add(add, trial):
                            new_S = trial | {add}
                            if len(new_S) == len(S) and full_validate(new_S):
                                # Accept swap (same size, different composition)
                                # Prefer if add < rem (getting smaller elements)
                                if add < rem or random.random() < 0.3:
                                    S = new_S
                                    improved = True
                                    break
                    if improved:
                        break
                
                if not improved:
                    break
        
        return best
    
    def dynamic_weighted_search(S, max_iter=30000):
        """Max-SAT style local search with dynamic clause weighting."""
        S = set(S)
        best = set(S)
        best_size = len(S)
        
        # Track clause weights: for each forbidden triple, how "close" it is to being violated
        # A triple (a,b,c) is violated if all three are in S
        # "Close" means 2 of 3 are in S
        
        # Weights for each element: how many near-violated triples it participates in
        elem_weight = [0] * (N + 1)
        
        for _ in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            # Compute current near-violation status
            near_violations = {}  # triple -> count of elements in S
            for (a, b, c) in forbidden_triples_set:
                cnt = (1 if a in S else 0) + (1 if b in S else 0) + (1 if c in S else 0)
                if cnt == 2:
                    near_violations[(a, b, c)] = (b, c) if a not in S else ((a, c) if b not in S else (a, b))
            
            # Update element weights
            elem_weight = [0] * (N + 1)
            for (a, b, c), missing in near_violations.items():
                for e in (a, b, c):
                    elem_weight[e] += 1
            
            # Try to improve: add an element or do a swap
            # Score of adding x: gain = 1 (size), cost = number of new near-violations created
            best_move = None
            best_score = -1
            
            # Try adding elements
            for x in range(1, N + 1):
                if x in S:
                    continue
                if not can_add(x, S):
                    continue
                # Count new near-violations
                new_nv = 0
                for (b, c) in constraint[x]:
                    if b in S and c in S:
                        continue  # would be a full violation, but can_add already checked
                    # This pair becomes a near-violation with x
                    # Actually if b in S and c in S, can_add would have returned False
                    # So this branch is unreachable if can_add passed
                # Since can_add passed, no full violation. Check near-violations:
                for (b, c) in constraint[x]:
                    if (b in S) != (c in S):
                        new_nv += 1
                score = 1 - 0.1 * new_nv - 0.01 * elem_weight[x]
                if score > best_score:
                    best_score = score
                    best_move = ('add', x)
            
            # Try swaps: remove r, add a
            # Only consider if it improves
            elements_list = list(S)
            if len(elements_list) > 50:
                elements_list = random.sample(elements_list, 50)
            
            for r in elements_list:
                trial = S - {r}
                # Try adding
                for a in range(1, N + 1):
                    if a in trial:
                        continue
                    if not can_add(a, trial):
                        continue
                    # Score: net change in near-violations
                    # Remove r: some near-violations go away
                    # Add a: some near-violations appear
                    score = 0.0  # size unchanged
                    # Heuristic: prefer swaps that reduce elem_weight
                    score -= 0.05 * elem_weight[a]
                    score += 0.05 * elem_weight[r]
                    if score > best_score:
                        best_score = score
                        best_move = ('swap', r, a)
            
            if best_move is None or best_score <= 0:
                # Try random perturbation to escape local minimum
                if random.random() < 0.5 and len(S) > 10:
                    r = random.choice(list(S))
                    a = random.choice([x for x in range(1, N + 1) if x not in S])
                    trial = S - {r}
                    if can_add(a, trial) and full_validate(trial | {a}):
                        S = trial | {a}
                        continue
                break
            
            if best_move[0] == 'add':
                S.add(best_move[1])
            else:
                S.discard(best_move[1])
                S.add(best_move[2])
            
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
        
        return best
    
    # Strategy: multiple restarts with different seeds
    best_overall = set()
    best_size_overall = 0
    
    # Initial strategies
    strategies = []
    
    # Strategy 1: Greedy from scratch
    S1 = greedy_build()
    strategies.append(S1)
    
    # Strategy 2: Start with large numbers, then repair
    S2 = set(range(91, N + 1))
    # Validate and fix
    S2_fixed = set()
    for x in sorted(S2, reverse=True):
        if can_add(x, S2_fixed):
            S2_fixed.add(x)
    strategies.append(S2_fixed)
    
    # Strategy 3: Start with small numbers
    S3 = set()
    for x in range(1, 91):
        if can_add(x, S3):
            S3.add(x)
    strategies.append(S3)
    
    # Strategy 4: Random seed
    random.seed(42)
    S4 = set(random.sample(range(1, N + 1), 90))
    S4_fixed = set()
    for x in sorted(S4):
        if can_add(x, S4_fixed):
            S4_fixed.add(x)
    strategies.append(S4_fixed)
    
    # Strategy 5: Start with all even numbers
    S5 = set(range(2, N + 1, 2))
    S5_fixed = set()
    for x in sorted(S5):
        if can_add(x, S5_fixed):
            S5_fixed.add(x)
    strategies.append(S5_fixed)
    
    # Run local search on each strategy
    for i, S_init in enumerate(strategies):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Phase 1: Repair local search
        S_rep = repair_local_search(S_init, max_iter=20000)
        if len(S_rep) > best_size_overall:
            best_overall = set(S_rep)
            best_size_overall = len(S_rep)
        
        # Phase 2: Dynamic weighted search
        if time.time() - start_time > DEADLINE - 3:
            break
        S_dw = dynamic_weighted_search(S_rep, max_iter=15000)
        if len(S_dw) > best_size_overall:
            best_overall = set(S_dw)
            best_size_overall = len(S_dw)
        
        # Phase 3: Another repair pass
        if time.time() - start_time > DEADLINE - 2:
            break
        S_rep2 = repair_local_search(S_dw, max_iter=10000)
        if len(S_rep2) > best_size_overall:
            best_overall = set(S_rep2)
            best_size_overall = len(S_rep2)
    
    # Final: try to add any remaining elements
    for x in range(1, N + 1):
        if time.time() - start_time > DEADLINE:
            break
        if x not in best_overall and can_add(x, best_overall):
            best_overall.add(x)
    
    # Final validation
    if not full_validate(best_overall):
        # Fallback: rebuild greedily
        best_overall = greedy_build()
    
    if len(best_overall) == 0:
        best_overall = {1}
    
    return sorted(best_overall)
