import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    N = 128
    # Pre-compute all potential violation 6-sets
    violations = []
    
    # Generate all triples a < b < c
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s_ab = a + b
            if s_ab > N:
                continue
            for c in range(b + 1, N + 1):
                s_ac = a + c
                s_bc = b + c
                if s_ac > N or s_bc > N:
                    continue
                viol = (a, b, c, s_ab, s_ac, s_bc)
                violations.append(viol)
    
    # Build for each element, which violations it participates in
    elem_violations = [[] for _ in range(N + 1)]
    for i, v in enumerate(violations):
        for x in v:
            elem_violations[x].append(i)
    
    # Initial set: all 1..128
    S = [True] * (N + 1)  # 1-indexed
    
    # Helper to check if a violation is active
    def is_active(idx):
        v = violations[idx]
        return all(S[x] for x in v)
    
    # Initial check
    active_violations = set(i for i in range(len(violations)) if is_active(i))
    
    # Greedy removal
    while active_violations:
        if time.time() > deadline:
            break
            
        # Count occurrences of each element in active violations
        counts = [0] * (N + 1)
        for vi in active_violations:
            v = violations[vi]
            for x in v:
                counts[x] += 1
        
        if max(counts) == 0:
            break
            
        # Find element with max count; break ties by preferring elements
        # that appear in fewer total violations (less central, less costly to remove)
        max_count = max(counts)
        candidates = [x for x in range(1, N + 1) if counts[x] == max_count]
        
        # Among ties, prefer the one with smallest total violation count
        x_to_remove = min(candidates, key=lambda x: len(elem_violations[x]))
        
        # Remove x_to_remove from S
        S[x_to_remove] = False
        
        # Update active_violations
        to_remove = set()
        for vi in elem_violations[x_to_remove]:
            if vi in active_violations:
                to_remove.add(vi)
        active_violations -= to_remove
    
    # Local search: iteratively try to add back removed elements
    removed_elements = [i for i in range(1, N + 1) if not S[i]]
    
    # Multiple rounds of add-back (forward order)
    max_rounds = 20
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        any_added = False
        for elem in sorted(removed_elements):
            if S[elem]:
                continue
            
            # Check if adding elem would create any active violation
            can_add = True
            for vi in elem_violations[elem]:
                v = violations[vi]
                all_others_present = True
                for x in v:
                    if x != elem and not S[x]:
                        all_others_present = False
                        break
                if all_others_present:
                    can_add = False
                    break
            
            if can_add:
                S[elem] = True
                any_added = True
        
        if not any_added:
            break
    
    # Reverse order pass
    for elem in sorted(removed_elements, reverse=True):
        if time.time() > deadline - 2:
            break
        
        if S[elem]:
            continue
        
        can_add = True
        for vi in elem_violations[elem]:
            v = violations[vi]
            all_others_present = True
            for x in v:
                if x != elem and not S[x]:
                    all_others_present = False
                    break
            if all_others_present:
                can_add = False
                break
        
        if can_add:
            S[elem] = True
    
    # Convert to list of ints
    result = [i for i in range(1, N + 1) if S[i]]
    
    # Ensure types are correct
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
