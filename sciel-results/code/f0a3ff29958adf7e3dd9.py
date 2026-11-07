import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    N = 128
    # Pre-compute all potential violation sets
    # A violation is a set of 6 values (indices 1..128)
    # defined by distinct a, b, c such that a+b, a+c, b+c are in 1..128
    violations = []
    # Map from element to list of violation indices it belongs to
    # We'll use a list of sets or just tuples
    
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
                # The set of values involved
                # a, b, c, s_ab, s_ac, s_bc
                # Note: some might be equal, but a,b,c are distinct.
                # s_ab, s_ac, s_bc are distinct from each other.
                # They might equal a,b,c.
                # The condition is that ALL these values are in S.
                viol = (a, b, c, s_ab, s_ac, s_bc)
                violations.append(viol)
    
    # Initial set: all 1..128
    S = [True] * (N + 1)  # 1-indexed
    
    # Helper to check if a violation is active
    def is_active(viol):
        return all(S[x] for x in viol)
    
    # Initial check
    active_violations = [v for v in violations if is_active(v)]
    
    # Greedy removal
    # We want to remove elements to deactivate all violations.
    # Strategy: While there are active violations, pick one, and remove an element from it.
    # To be efficient, we can pick the element that appears in the most active violations.
    
    # However, recalculating counts for all elements is expensive.
    # Simple approach: Pick the first active violation, remove the first element in it.
    # This might be suboptimal but fast.
    
    # Better: Count frequency of each element in active_violations.
    # Remove the most frequent one.
    
    while active_violations:
        if time.time() > deadline:
            break
            
        # Count occurrences of each element in active_violations
        counts = {}
        for v in active_violations:
            for x in v:
                counts[x] = counts.get(x, 0) + 1
        
        # Find element with max count
        if not counts:
            break
            
        max_count = max(counts.values())
        # Pick one of the elements with max count
        # To be deterministic, pick the smallest one
        candidates = [x for x, c in counts.items() if c == max_count]
        x_to_remove = min(candidates)
        
        # Remove x_to_remove from S
        S[x_to_remove] = False
        
        # Update active_violations
        # Remove any violation that contains x_to_remove
        new_active = []
        for v in active_violations:
            if x_to_remove not in v:
                new_active.append(v)
            else:
                # This violation is deactivated because x_to_remove is gone
                pass
        
        active_violations = new_active
        
    # Convert to list of ints
    result = [i for i in range(1, N + 1) if S[i]]
    
    # Ensure types are correct
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
