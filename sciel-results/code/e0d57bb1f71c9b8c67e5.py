import time
import random
import math

def construct():
    N = 180
    start_time = time.time()
    time_limit = 45.0  # Keep safety margin
    
    # 1. Precompute constraints
    # A constraint is a set of 3 distinct values {a, b, c} in [1, N]
    # such that 1/a = 1/b + 1/c.
    # Derived: (a-b)(a-c) = a^2. Let u = b-a, v = c-a. uv = a^2.
    # u, v > 0. b = a+u, c = a+v.
    # We need b, c <= N.
    # We need a, b, c distinct. Since u,v > 0, b>a, c>a.
    # If u == v, b == c, so {a, b} has size 2. Not a triple of distinct elements.
    # So we only care if u != v.
    
    constraints = []
    # Map value (1..N) to index (0..N-1) for internal representation
    # We will store constraints as tuples of indices (0-based)
    
    # To efficiently check conflicts, we might want to store for each index
    # the list of constraints it belongs to, and the other two indices in those constraints.
    # constraints: list of (idx1, idx2, idx3)
    # adj: list of lists. adj[i] contains (j, k, constraint_id) for each constraint involving i.
    
    raw_constraints = set()
    for a in range(1, N + 1):
        a2 = a * a
        # Find divisors of a2
        # Iterate up to sqrt(a2)
        limit = int(math.isqrt(a2))
        divisors = []
        for d in range(1, limit + 1):
            if a2 % d == 0:
                divisors.append(d)
                if d != a2 // d:
                    divisors.append(a2 // d)
        
        # For each divisor u, v = a2 / u
        for u in divisors:
            v = a2 // u
            if u == v:
                continue # b == c, not 3 distinct elements
            b = a + u
            c = a + v
            if b <= N and c <= N:
                # Check distinctness of values (already handled by u!=v and u,v>0)
                # Store sorted tuple of values to deduplicate
                triple = tuple(sorted((a, b, c)))
                raw_constraints.add(triple)
    
    # Convert values to indices (0-based)
    for t in raw_constraints:
        constraints.append((t[0] - 1, t[1] - 1, t[2] - 1))
    
    # Build adjacency structure for fast delta calculation
    # adj[i] = list of (j, k) such that (i, j, k) is a constraint
    adj = [[] for _ in range(N)]
    for (i, j, k) in constraints:
        adj[i].append((j, k))
        adj[j].append((i, k))
        adj[k].append((i, j))
        
    # 2. Simulated Annealing
    # State: boolean list `present` of length N
    # Objective: Maximize Size - 1000 * Violations
    
    best_set = None
    best_score = -float('inf')
    
    # We can run multiple restarts if time permits
    # Or one long run. Let's try a few restarts with random initialization.
    
    def get_score(present, violations):
        size = sum(present)
        return size - 1000 * violations
        
    # Helper to calculate initial violations
    def calc_violations(present):
        v = 0
        for (i, j, k) in constraints:
            if present[i] and present[j] and present[k]:
                v += 1
        return v

    # Initialization strategy:
    # Start with a random subset.
    # To get a good starting point, we can start with all 1s and greedily remove elements with highest "conflict count".
    # Or just random. Random is safer for diversity.
    
    num_restarts = 50
    # We will use time-based termination instead of fixed steps
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        # Random density 0.5
        present = [random.random() < 0.5 for _ in range(N)]
        violations = calc_violations(present)
        score = get_score(present, violations)
        
        cur_set = present[:]
        cur_violations = violations
        cur_score = score
        
        T = 10.0
        cooling = 0.9995 # Slower cooling for more exploration
        
        # Run SA for this restart until time limit or convergence
        # We don't have a strict convergence check, so we run until time or fixed steps
        # Let's run for 5000 steps per restart or until time limit
        steps_this_restart = 5000
        
        for step in range(steps_this_restart):
            if time.time() - start_time > time_limit:
                break
                
            # Pick random index to flip
            i = random.randint(0, N - 1)
            
            # Calculate delta
            # n_active: number of constraints involving i where the other two are present
            n_active = 0
            for (j, k) in adj[i]:
                if cur_set[j] and cur_set[k]:
                    n_active += 1
            
            if cur_set[i]:
                # Removing i
                # Delta K = -1
                # Delta V = -n_active (violations decrease)
                delta_k = -1
                delta_v = -n_active
            else:
                # Adding i
                # Delta K = 1
                # Delta V = +n_active (violations increase)
                delta_k = 1
                delta_v = n_active
            
            delta_score = delta_k - 1000 * delta_v
            
            if delta_score >= 0:
                # Accept
                cur_set[i] = not cur_set[i]
                cur_violations += delta_v
                cur_score += delta_score
            else:
                # Accept with probability exp(-delta_score / T)
                # Note: delta_score is negative.
                prob = math.exp(delta_score / T)
                if random.random() < prob:
                    cur_set[i] = not cur_set[i]
                    cur_violations += delta_v
                    cur_score += delta_score
            
            T *= cooling
            
            # Update best
            if cur_score > best_score:
                best_score = cur_score
                best_set = cur_set[:]
                
    # Extract result
    if best_set is None:
        # Fallback: just return a valid set if we found one, or empty?
        # The problem asks for nonempty subset.
        # If SA failed, try a simple greedy or just return [1]
        # But SA should find something.
        # Let's ensure we return a valid set.
        # If best_set has violations, we need to clean it.
        # Actually, best_score tracks the max score.
        # If best_score corresponds to violations > 0, we might have an invalid set.
        # We should prioritize valid sets.
        
        # Let's re-verify the best set validity
        # If best_set has violations, we should try to fix it or find a valid one.
        # Given the structure, a valid set always exists (e.g., just {1} or {180}).
        
        # If best_score < 0, it might be invalid.
        # Let's do a quick cleanup: remove elements causing conflicts from best_set
        # until it's valid.
        
        final_set = best_set[:]
        # Simple greedy removal
        changed = True
        while changed:
            changed = False
            for (i, j, k) in constraints:
                if final_set[i] and final_set[j] and final_set[k]:
                    # Remove the one with highest index? Or random?
                    # Remove i
                    final_set[i] = False
                    changed = True
        
        # Ensure non-empty
        if not any(final_set):
            final_set[0] = True # Just pick 1
            
        result = [i + 1 for i in range(N) if final_set[i]]
        return result
    
    # Check validity of best_set
    # If best_set is valid, return it.
    # If not, clean it.
    
    # The score function penalizes violations heavily.
    # So best_set should be valid or very close.
    
    final_set = best_set[:]
    
    # Greedy cleanup to ensure validity
    # This is just a safety net
    for (i, j, k) in constraints:
        if final_set[i] and final_set[j] and final_set[k]:
            # Remove one element
            # Heuristic: remove the one that is in the most other constraints?
            # For simplicity, remove i
            final_set[i] = False
            
    # It's possible that removing i creates no new violations, but we might have missed other violations.
    # A single pass might not be enough if removing i doesn't fix all, or if order matters.
    # However, the SA likely found a valid set.
    
    # To be absolutely safe, we can iterate until no violations.
    # But the constraint graph is sparse.
    
    # Let's do a proper cleanup:
    # While there is a violation, remove an element involved in it.
    # This guarantees a valid set.
    # It might reduce size, but it's a valid fallback.
    
    # Actually, if SA worked, best_set should be valid.
    # Let's just verify.
    
    # Re-calc violations for final_set
    v = 0
    for (i, j, k) in constraints:
        if final_set[i] and final_set[j] and final_set[k]:
            v += 1
            
    if v > 0:
        # Fix it
        while v > 0:
            # Find a violation
            for (i, j, k) in constraints:
                if final_set[i] and final_set[j] and final_set[k]:
                    # Remove i
                    final_set[i] = False
                    # Recalc v? Expensive.
                    # Just break and loop again?
                    break
            # Recalculate v
            v = 0
            for (i, j, k) in constraints:
                if final_set[i] and final_set[j] and final_set[k]:
                    v += 1
    
    result = [i + 1 for i in range(N) if final_set[i]]
    
    if not result:
        result = [1]
        
    return result
