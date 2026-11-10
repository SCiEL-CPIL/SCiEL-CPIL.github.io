def construct():
    import time
    import random
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    N = 180
    
    # Precompute conflict pairs for each a
    conflict_pairs = {}
    for a in range(1, N+1):
        pairs = []
        x2 = a * a
        for u in range(1, x2 + 1):
            if x2 % u == 0:
                v = x2 // u
                b = a + u
                c = a + v
                if b <= N and c <= N and b != c:
                    pairs.append((b, c))
        conflict_pairs[a] = pairs
    
    # Count conflicts for ordering
    conflict_count = {a: len(conflict_pairs[a]) for a in range(1, N+1)}
    
    def greedy_build(order):
        S = set()
        for x in order:
            can_add = True
            for b, c in conflict_pairs[x]:
                if b in S and c in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
        return S
    
    def local_search(S, max_rounds=50):
        S = set(S)
        
        for _ in range(max_rounds):
            if time.time() - start_time > deadline:
                break
            
            improved = False
            
            # Try to add each element not in S
            for x in range(1, N+1):
                if x in S:
                    continue
                
                # Find conflicts
                conflicts = []
                for b, c in conflict_pairs[x]:
                    if b in S and c in S:
                        conflicts.append((b, c))
                
                if not conflicts:
                    S.add(x)
                    improved = True
                    break
                
                # Try removing one element from each conflict to free up space
                for b, c in conflicts:
                    for rem in [b, c]:
                        if rem not in S:
                            continue
                        
                        # Check if removing rem and adding x is valid
                        # Only need to check:
                        # 1. No triple involving x and other elements in S-{rem}
                        # 2. No triple involving rem was the only conflict (now resolved)
                        
                        S_test = set(S)
                        S_test.discard(rem)
                        S_test.add(x)
                        
                        # Check x doesn't create new conflicts
                        valid = True
                        for b2, c2 in conflict_pairs[x]:
                            if b2 in S_test and c2 in S_test:
                                valid = False
                                break
                        
                        if valid and len(S_test) >= len(S):
                            # Also need to ensure no other triples are formed
                            # Since we only added x and removed rem, check:
                            # - Triples involving x: checked above
                            # - Triples not involving x: were valid before, removing rem only helps
                            S = S_test
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return S
    
    best_S = set()
    
    # Try multiple orderings
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N+1)),
        sorted(range(1, N+1), key=lambda x: conflict_count[x]),
        sorted(range(1, N+1), key=lambda x: -conflict_count[x]),
    ]
    
    for order in orders:
        if time.time() - start_time > deadline:
            break
        S = greedy_build(order)
        if len(S) > len(best_S):
            best_S = S
    
    # Local search
    if time.time() - start_time < deadline:
        best_S = local_search(best_S)
    
    # Random restarts
    if time.time() - start_time < deadline:
        random.seed(42)
        for i in range(20):
            if time.time() - start_time > deadline:
                break
            order = list(range(1, N+1))
            random.shuffle(order)
            S = greedy_build(order)
            if len(S) > len(best_S):
                best_S = S
                if time.time() - start_time < deadline:
                    best_S = local_search(best_S)
    
    return list(best_S)
