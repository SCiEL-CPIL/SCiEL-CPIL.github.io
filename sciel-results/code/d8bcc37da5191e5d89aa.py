import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s limit
    
    N = 180
    # 1. Precompute conflicts
    # conflicts_with[x] is a list of tuples (y, z) such that {x, y, z} is a forbidden triple.
    # We store pairs (y, z) with y < z to ensure uniqueness in the pair representation if needed, 
    # but since we check 'y in S and z in S', order doesn't matter for the check.
    conflicts_with = {i: [] for i in range(1, N + 1)}
    
    # Generate all forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c => a = bc / (b+c)
    # Iterate a from 1 to N
    for a in range(1, N + 1):
        a2 = a * a
        # b = a + d, c = a + a2/d
        # d must divide a2
        # b > a => d > 0
        # c > b => a + a2/d > a + d => a2/d > d => d^2 < a2 => d < a
        # Also b, c <= N
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                # We need distinct elements a, b, c
                # Since d >= 1, b > a.
                # If d == e, then b == c, which is not allowed (distinct).
                if b == c:
                    continue
                if b <= N and c <= N:
                    # We have a triple {a, b, c}
                    # Add to conflicts
                    # For a: conflicts with (b, c)
                    conflicts_with[a].append((b, c))
                    # For b: conflicts with (a, c)
                    conflicts_with[b].append((a, c))
                    # For c: conflicts with (a, b)
                    conflicts_with[c].append((a, b))
    
    # Precompute bitmask for each element's conflicts
    # For each x, conflict_mask[x] is a bitmask where bit i is set if i is in a conflict pair with x
    # Actually, we need to check if BOTH y and z are in S, so we need a different approach.
    # Let's precompute for each x, the list of (y, z) pairs, and also a set of all elements that appear in any conflict with x
    conflict_elements = {i: set() for i in range(1, N + 1)}
    for x in range(1, N + 1):
        for y, z in conflicts_with[x]:
            conflict_elements[x].add(y)
            conflict_elements[x].add(z)
    
    # Helper: Check if x can be added to set S (using set membership)
    def can_add(x, S):
        for y, z in conflicts_with[x]:
            if y in S and z in S:
                return False
        return True
    
    # Helper: Check if x can be added, but only considering conflicts with elements in a specific set
    def can_add_restricted(x, S, exclude=None):
        """Check if x can be added to S, optionally excluding certain elements from consideration."""
        for y, z in conflicts_with[x]:
            if exclude and (y in exclude or z in exclude):
                continue
            if y in S and z in S:
                return False
        return True
    
    # Helper: Count how many elements can be added to S
    def count_addable(S):
        count = 0
        for x in range(1, N + 1):
            if x not in S:
                if can_add(x, S):
                    count += 1
        return count

    # Greedy Initialization
    def greedy_init():
        S = set()
        # Order by degree (number of conflicts) ascending? 
        # Elements with fewer conflicts are "safer" to add early.
        # Or descending? If we add high conflict elements first, we block many others.
        # Usually, adding "easy" elements first allows more "hard" elements to fit later?
        # Let's try sorting by len(conflicts_with[x]) ascending.
        order = sorted(range(1, N + 1), key=lambda x: len(conflicts_with[x]))
        
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S
    
    # Improved greedy with multiple strategies
    def greedy_init_multi():
        best_S = set()
        best_size = 0
        
        # Strategy 1: Sort by conflict count ascending
        order1 = sorted(range(1, N + 1), key=lambda x: len(conflicts_with[x]))
        S1 = set()
        for x in order1:
            if can_add(x, S1):
                S1.add(x)
        if len(S1) > best_size:
            best_size = len(S1)
            best_S = set(S1)
        
        # Strategy 2: Sort by conflict count descending
        order2 = sorted(range(1, N + 1), key=lambda x: len(conflicts_with[x]), reverse=True)
        S2 = set()
        for x in order2:
            if can_add(x, S2):
                S2.add(x)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = set(S2)
        
        # Strategy 3: Random order
        for _ in range(3):
            order3 = list(range(1, N + 1))
            random.shuffle(order3)
            S3 = set()
            for x in order3:
                if can_add(x, S3):
                    S3.add(x)
            if len(S3) > best_size:
                best_size = len(S3)
                best_S = set(S3)
        
        return best_S

    # Local Search with improved swap heuristics
    def local_search(S, max_time=10):
        S = set(S)
        best_local = set(S)
        best_local_size = len(S)
        
        last_improvement_time = time.time()
        
        while time.time() - last_improvement_time < max_time and time.time() < DEADLINE:
            improved = False
            
            # 1. Try to add - prioritize elements with fewer conflicts
            candidates = [x for x in range(1, N + 1) if x not in S]
            # Sort by number of conflicts with elements in S (fewer conflicts = easier to add)
            def conflict_count_with_S(x):
                count = 0
                for y, z in conflicts_with[x]:
                    if y in S or z in S:
                        count += 1
                return count
            
            candidates.sort(key=conflict_count_with_S)
            random.shuffle(candidates)  # Add some randomness
            
            for x in candidates:
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if improved:
                last_improvement_time = time.time()
                if len(S) > best_local_size:
                    best_local_size = len(S)
                    best_local = set(S)
                continue
            
            # 2. Try to swap - use heuristic to find beneficial swaps
            S_list = list(S)
            random.shuffle(S_list)
            
            # Precompute addable count for current S
            current_addable = count_addable(S)
            
            # Try swaps: remove u, add v
            # Heuristic: v should have fewer conflicts with S\{u} than u has with S
            swaps_found = 0
            max_swaps = 20
            
            for u in S_list:
                if time.time() > DEADLINE:
                    break
                if swaps_found >= max_swaps:
                    break
                
                # Find elements that can replace u
                # v must be addable to S\{u}
                S_without_u = S - {u}
                
                # Get candidates for v: elements not in S that can be added to S_without_u
                v_candidates = []
                for v in range(1, N + 1):
                    if v in S:
                        continue
                    if can_add_restricted(v, S_without_u, exclude={u}):
                        v_candidates.append(v)
                
                if not v_candidates:
                    continue
                
                # Score v candidates by how many new elements they enable
                best_v = None
                best_score = -1
                
                for v in v_candidates[:10]:  # Limit to top 10
                    # Estimate: how many elements become addable after swap?
                    # This is expensive, so use a lighter heuristic
                    # Count conflicts of v with S_without_u
                    v_conflicts = 0
                    for y, z in conflicts_with[v]:
                        if y in S_without_u or z in S_without_u:
                            v_conflicts += 1
                    
                    # Prefer v with fewer conflicts
                    score = -v_conflicts
                    if score > best_score:
                        best_score = score
                        best_v = v
                
                if best_v is not None:
                    # Perform the swap
                    S.remove(u)
                    S.add(best_v)
                    improved = True
                    swaps_found += 1
                    last_improvement_time = time.time()
            
            if not improved:
                # No add, no beneficial swap found
                # Perform a Kick: Remove a random element to break structure
                if S:
                    # Remove element with most conflicts to free up space
                    u = max(S, key=lambda x: len([1 for y, z in conflicts_with[x] if y in S or z in S]))
                    S.remove(u)
                    last_improvement_time = time.time()
        
        return best_local

    # Main Loop
    random.seed(42)
    
    # 1. Greedy Start with multiple strategies
    S = greedy_init_multi()
    S = local_search(S, max_time=8)
    
    best_set = set(S)
    best_size = len(S)
    
    # 2. Multiple Restarts with different seeds/orders if time permits
    # We can try greedy with different orderings
    restart_count = 0
    while time.time() < DEADLINE - 10:
        restart_count += 1
        
        # Randomize order for greedy
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        
        # Give more time to local search for later restarts if we're finding improvements
        local_time = 3 if len(S) < best_size else 2
        S = local_search(S, max_time=local_time)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # 3. Final polish: try to add more elements to the best set found
    if time.time() < DEADLINE - 2:
        final_S = set(best_set)
        # Try adding elements one by one
        candidates = [x for x in range(1, N + 1) if x not in final_S]
        candidates.sort(key=lambda x: len(conflicts_with[x]))
        
        for x in candidates:
            if can_add(x, final_S):
                final_S.add(x)
        
        if len(final_S) > best_size:
            best_size = len(final_S)
            best_set = final_S
    
    return list(best_set)
