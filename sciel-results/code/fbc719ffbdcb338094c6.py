"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _is_valid_set(S):
    """Check if set S satisfies the constraint."""
    S_list = sorted(S)
    for i, a in enumerate(S_list):
        for j in range(i+1, len(S_list)):
            b = S_list[j]
            for k in range(j, len(S_list)):
                c = S_list[k]
                if (b + c) % a == 0:
                    return False
    return True

def _try_add(S, cand, n):
    """Check if cand can be added to S."""
    larger = [x for x in S if x > cand]
    for i in range(len(larger)):
        b = larger[i]
        for j in range(i, len(larger)):
            c = larger[j]
            if (b + c) % cand == 0:
                return False
    
    smaller = [x for x in S if x < cand]
    geq = [x for x in S if x >= cand]
    for a in smaller:
        for c in geq:
            if (cand + c) % a == 0:
                return False

    for a in S:
        if a >= cand:
            continue
        for b in S:
            if a < b <= cand and (b + cand) % a == 0:
                return False
    return True

def _local_search_add(S, n, deadline):
    """Try to add skipped elements to S."""
    S = set(S)
    skipped = set(range(1, n+1)) - S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for cand in sorted(skipped, reverse=True):
            if time.time() > deadline:
                break
            if _try_add(S, cand, n):
                S.add(cand)
                skipped.discard(cand)
                improved = True
    return S

def _construct_greedy(n, order, deadline):
    """Greedy construction based on a specific order of candidates."""
    S = set()
    for cand in order:
        if time.time() > deadline:
            break
        if _try_add(S, cand, n):
            S.add(cand)
    return S

def _precompute_clauses(n):
    """Precompute all forbidden triples (a,b,c) with a<b<=c and a|(b+c)."""
    clauses = []  # list of (a, b, c)
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            for c in range(b, n+1):
                if (b + c) % a == 0:
                    clauses.append((a, b, c))
    return clauses

def _walksat_search(S_init, n, clauses, deadline, max_flips=200000, restarts=5):
    """
    WalkSAT-style local search on MaxSAT formulation.
    Variables: x[i] = True means i is in the set.
    Objective: maximize sum(x[i]) - penalty for violated clauses.
    Each clause (a,b,c) is violated when x[a]=x[b]=x[c]=True.
    """
    best_score = -1
    best_assignment = None
    
    for restart in range(restarts):
        if time.time() > deadline:
            break
        
        # Initialize: start from S_init, or random perturbation on restarts > 0
        if restart == 0:
            x = [False] * (n + 1)
            for i in S_init:
                x[i] = True
        else:
            # Perturb: flip some variables
            x = [False] * (n + 1)
            for i in S_init:
                x[i] = True
            # Randomly flip 10-30% of variables
            num_flips = random.randint(int(0.1 * n), int(0.3 * n))
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
            for i in candidates[:num_flips]:
                x[i] = not x[i]
        
        # Dynamic clause weights
        clause_weights = [1.0] * len(clauses)
        
        cur_score = sum(x[1:])  # cardinality
        best_in_run = cur_score
        best_x_in_run = x[:]
        
        # Track violations per clause for efficiency
        # A clause (a,b,c) is violated iff x[a] and x[b] and x[c] are all True
        
        for step in range(max_flips):
            if time.time() > deadline:
                break
            
            # Find all currently violated clauses
            violated_clauses = []
            for idx, (a, b, c) in enumerate(clauses):
                if x[a] and x[b] and x[c]:
                    violated_clauses.append(idx)
            
            if not violated_clauses:
                # No violations; try to add more elements
                # Find variables that can be set to True without creating violations
                added = False
                for v in range(1, n+1):
                    if not x[v]:
                        # Check if setting x[v]=True creates any violation
                        creates_violation = False
                        for idx, (a, b, c) in enumerate(clauses):
                            if v in (a, b, c):
                                # Check if the other two are True
                                others = [t for t in (a, b, c) if t != v]
                                if x[others[0]] and x[others[1]]:
                                    creates_violation = True
                                    break
                        if not creates_violation:
                            x[v] = True
                            cur_score += 1
                            added = True
                            break
                if not added:
                    break  # Local optimum
                if cur_score > best_in_run:
                    best_in_run = cur_score
                    best_x_in_run = x[:]
                continue
            
            # Pick a random violated clause
            vidx = random.choice(violated_clauses)
            a, b, c = clauses[vidx]
            
            # Increase weight of this clause
            clause_weights[vidx] += 1.0
            
            # For each variable in the clause, compute the cost of flipping it
            # Flipping x[v] from True to False:
            #   - Removes this clause's violation (gain = clause_weight)
            #   - May create new violations where v is part of a clause with other two True
            #   - Loses 1 from cardinality
            # Flipping from False to True is not applicable here since all three are True
            
            best_var = None
            best_cost = float('inf')
            
            for v in (a, b, c):
                # Cost of flipping x[v] from True to False
                # Gain: remove this clause violation
                gain = clause_weights[vidx]
                # Loss: 1 from cardinality
                loss = 1.0
                # Additional cost: new violations created by this flip
                # When x[v] becomes False, clauses containing v that were not violated
                # remain not violated. Clauses that WERE violated due to v being True
                # are now fixed. But we need to check if flipping v creates NEW violations
                # for clauses where v is the False one... no, v is going from True to False.
                # Actually, flipping v from True to False can only REMOVE violations, not create them.
                # Wait, that's correct. So the only cost is -1 from cardinality.
                # But we also need to consider: by removing v, do we enable other additions?
                # For now, just use immediate cost.
                cost = loss - gain
                # Add noise for exploration
                cost += random.uniform(0, 0.5)
                
                if cost < best_cost:
                    best_cost = cost
                    best_var = v
            
            # Also consider flipping a variable NOT in the clause (to add it)
            # This helps escape local optima by adding new elements
            if random.random() < 0.3:
                # Try adding a random variable
                v = random.randint(1, n)
                if not x[v]:
                    # Check if adding v creates violations
                    new_violations = 0
                    for idx, (a2, b2, c2) in enumerate(clauses):
                        if v in (a2, b2, c2):
                            others = [t for t in (a2, b2, c2) if t != v]
                            if x[others[0]] and x[others[1]]:
                                new_violations += clause_weights[idx]
                    # Gain: +1 from cardinality
                    # Cost: new_violations
                    cost = new_violations - 1.0 + random.uniform(0, 0.5)
                    if cost < best_cost:
                        best_cost = cost
                        best_var = v
            
            if best_var is None:
                # Random flip
                best_var = random.choice((a, b, c))
            
            # Perform the flip
            x[best_var] = not x[best_var]
            cur_score = sum(x[1:])
            
            if cur_score > best_in_run:
                best_in_run = cur_score
                best_x_in_run = x[:]
        
        # End of run
        if best_in_run > best_score:
            best_score = best_in_run
            best_assignment = best_x_in_run[:]
    
    # Convert best assignment to set
    if best_assignment is None:
        return set(S_init)
    
    S_result = set()
    for i in range(1, n+1):
        if best_assignment[i]:
            S_result.add(i)
    
    # Validate and clean: remove elements that cause violations
    # Iteratively remove the element in the most violated clauses
    changed = True
    while changed:
        changed = False
        # Find all violations
        violations = []
        for (a, b, c) in clauses:
            if a in S_result and b in S_result and c in S_result:
                violations.append((a, b, c))
        
        if not violations:
            break
        
        # Count how many violations each element is in
        from collections import Counter
        count = Counter()
        for (a, b, c) in violations:
            count[a] += 1
            count[b] += 1
            count[c] += 1
        
        # Remove the most problematic element
        worst = max(count, key=count.get)
        S_result.discard(worst)
        changed = True
    
    return S_result

def construct():
    start_time = time.time()
    total_deadline = start_time + 85 # Safety margin
    n = 256
    
    best_S = set()
    
    # Strategy 1: Descending order (Current behavior, usually good)
    S1 = _construct_greedy(n, range(n, 0, -1), total_deadline)
    S1 = _local_search_add(S1, n, total_deadline)
    if len(S1) > len(best_S):
        best_S = S1
        
    # Strategy 2: Ascending order (Riskier, but might find different structures)
    if time.time() < total_deadline:
        S2 = _construct_greedy(n, range(1, n+1), total_deadline)
        S2 = _local_search_add(S2, n, total_deadline)
        if len(S2) > len(best_S):
            best_S = S2
            
    # Strategy 3: Randomized Greedy
    rng = random.Random(42)
    for _ in range(5):
        if time.time() > total_deadline - 5: # Stop if close to deadline
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        S_rand = _construct_greedy(n, order, total_deadline)
        S_rand = _local_search_add(S_rand, n, total_deadline)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    # Strategy 4: WalkSAT local search on MaxSAT formulation
    if time.time() < total_deadline - 15:
        # Precompute clauses
        clauses = _precompute_clauses(n)
        
        # Run WalkSAT starting from best_S
        S_walk = _walksat_search(best_S, n, clauses, total_deadline - 5, 
                                  max_flips=100000, restarts=3)
        if len(S_walk) > len(best_S):
            best_S = S_walk
    
    # Final Polish: Try swaps
    if time.time() < total_deadline - 10:
        S_polish = set(best_S)
        skipped = set(range(1, n+1)) - S_polish
        swapped = True
        while swapped and time.time() < total_deadline - 2:
            swapped = False
            for cand in list(skipped):
                if _try_add(S_polish, cand, n):
                    S_polish.add(cand)
                    skipped.discard(cand)
                    swapped = True
                    break
            
        if len(S_polish) > len(best_S):
            best_S = S_polish

    return sorted(best_S)
# EVOLVE-BLOCK-END
