import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 180
    elements = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples
    triples = []
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    t = (a, b, c)
                    idx = len(triples)
                    triples.append(t)
                    for x in t:
                        elem_to_triples[x].append(idx)
    
    # Precompute for each element, the set of triples it participates in
    # and for each triple, the other two elements
    triple_others = {}
    for idx, (a, b, c) in enumerate(triples):
        triple_others[idx] = (b, c) if a == a else (a, c)  # placeholder
    
    # Better: for each element x, store list of (other1, other2, triple_idx)
    elem_conflicts = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        for x in (a, b, c):
            others = [y for y in (a, b, c) if y != x]
            elem_conflicts[x].append((others[0], others[1], idx))

    def can_add(S, x):
        """Check if x can be added to S without creating a forbidden triple."""
        for o1, o2, idx in elem_conflicts[x]:
            if o1 in S and o2 in S:
                return False
        return True

    # Helper to run greedy removal with random tie-breaking
    def run_greedy(randomize_ties=False):
        active_elements = set(elements)
        active_triples = set(range(len(triples)))
        conflict_count = {i: 0 for i in range(1, N + 1)}
        
        for idx in active_triples:
            for x in triples[idx]:
                conflict_count[x] += 1
        
        while time.time() < deadline:
            # Find element with max conflict count
            max_conflict = -1
            candidates = []
            for x in active_elements:
                if conflict_count[x] > max_conflict:
                    max_conflict = conflict_count[x]
                    candidates = [x]
                elif conflict_count[x] == max_conflict:
                    candidates.append(x)
            
            if max_conflict == 0:
                break # No more conflicts
            
            # Select victim
            if randomize_ties and len(candidates) > 1:
                victim = random.choice(candidates)
            else:
                victim = candidates[0]

            active_elements.remove(victim)
            
            # Update active_triples and conflict_counts
            for idx in elem_to_triples[victim]:
                if idx in active_triples:
                    active_triples.remove(idx)
                    for x in triples[idx]:
                        if x != victim and x in active_elements:
                            conflict_count[x] -= 1
        
        return active_elements

    def local_search(S, time_budget=5.0):
        """Local search: try to add elements, and do 1-for-1 swaps if beneficial."""
        S = set(S)
        removed = set(elements) - S
        removed_list = list(removed)
        random.shuffle(removed_list)
        
        # First pass: add any element that can be added directly
        for x in removed_list:
            if time.time() > start_time + time_budget:
                break
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        
        # Second pass: try 1-for-1 swaps
        # For each removed x, check if removing some y in S allows adding x
        # and if the net effect enables further additions
        improved = True
        swap_count = 0
        while improved and time.time() < start_time + time_budget:
            improved = False
            removed = list(set(elements) - S)
            random.shuffle(removed)
            for x in removed:
                if time.time() > start_time + time_budget:
                    break
                # Find all y in S such that x and y conflict
                # i.e., there exists a triple containing both x and y
                conflicting_ys = set()
                for o1, o2, idx in elem_conflicts[x]:
                    if o1 in S:
                        conflicting_ys.add(o1)
                    if o2 in S:
                        conflicting_ys.add(o2)
                
                if not conflicting_ys:
                    continue  # x can be added directly (shouldn't happen after first pass)
                
                # Try swapping: remove y, add x
                # Check if after removing y, x can be added
                for y in list(conflicting_ys):
                    if time.time() > start_time + time_budget:
                        break
                    S.remove(y)
                    if can_add(S, x):
                        S.add(x)
                        swap_count += 1
                        improved = True
                        # Try to add back y or other elements
                        for z in list(set(elements) - S):
                            if time.time() > start_time + time_budget:
                                break
                            if z in S:
                                continue
                            if can_add(S, z):
                                S.add(z)
                    else:
                        S.add(y)  # restore
                        break  # this swap doesn't work, try next y
        
        return S

    # Strategy: Run multiple randomized greedy passes
    best_set = set()
    best_size = 0
    
    # First run: Deterministic (smallest victim on tie)
    s = run_greedy(randomize_ties=False)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # Randomized runs
    for i in range(1000):
        if time.time() > deadline - 8:  # Reserve 8s for local search
            break
        s = run_greedy(randomize_ties=True)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Final Local Search on best_set with time budget
    remaining_time = deadline - time.time() - 1  # 1s safety
    if remaining_time > 2:
        S = local_search(best_set, time_budget=min(remaining_time, 10.0))
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return list(best_set)

# EVOLVE-BLOCK-END
