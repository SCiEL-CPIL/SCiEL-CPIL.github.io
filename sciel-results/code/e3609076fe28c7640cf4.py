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

    # Helper to run greedy removal with random tie-breaking
    def run_greedy(randomize_ties=False):
        active_elements = set(elements)
        active_triples = set(range(len(triples)))
        conflict_count = {i: 0 for i in range(1, N + 1)}
        
        for idx in active_triples:
            for x in triples[idx]:
                conflict_count[x] += 1
        
        best_size = 0
        best_set = set()
        
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
                # Default: pick smallest to keep larger numbers?
                # Or just first.
                victim = candidates[0]
                if randomize_ties:
                     # Even if no ties, randomize slightly? No, just tie-breaking.
                     pass

            active_elements.remove(victim)
            
            # Update active_triples and conflict_counts
            for idx in elem_to_triples[victim]:
                if idx in active_triples:
                    active_triples.remove(idx)
                    for x in triples[idx]:
                        if x != victim and x in active_elements:
                            conflict_count[x] -= 1
        
        return active_elements

    # Strategy: Run multiple randomized greedy passes
    best_set = set()
    best_size = 0
    
    # First run: Deterministic (smallest victim on tie)
    s = run_greedy(randomize_ties=False)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # Randomized runs
    # Number of runs depends on time. Each run is very fast.
    # Let's do 500 runs.
    for i in range(500):
        if time.time() > deadline - 2:
            break
        s = run_greedy(randomize_ties=True)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Final Local Search on best_set
    # Try to add back removed elements
    removed = set(elements) - best_set
    # Shuffle removed for randomization in local search
    removed_list = list(removed)
    random.shuffle(removed_list)
    
    S = set(best_set)
    
    # Simple hill climbing: try to add x, if conflict, remove one conflicting y
    # We only accept if size increases (impossible in 1-for-1) 
    # So we accept 1-for-1 swaps only if they enable further additions?
    # This is complex. 
    # Instead, let's just try to add x if possible (0 conflicts).
    # This can only increase size.
    for x in removed_list:
        if time.time() > deadline - 1:
            break
        if x in S:
            continue
        # Check if x can be added
        can_add = True
        for idx in elem_to_triples[x]:
            a, b, c = triples[idx]
            # Check if the other two are in S
            others = [y for y in (a, b, c) if y != x]
            if others[0] in S and others[1] in S:
                can_add = False
                break
        if can_add:
            S.add(x)

    return list(S)

# EVOLVE-BLOCK-END
