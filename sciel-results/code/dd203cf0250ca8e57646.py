def construct():
    import time
    from math import gcd
    import random
    
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    N = 128
    # Precompute GCD matrix
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g[i][j] = g[j][i] = gcd(i, j)
    
    # Precompute "badness" contribution for adding x to a set containing a and b
    # We need to check if g[a][b] == g[a][x] == g[b][x]
    # To speed up, we can precompute for each x, a list of (a, b) pairs that form a bad triple with x.
    # But since N is small, checking on the fly is O(|S|^2) which is fine.
    
    def count_bad_triples_involving(x, S):
        """
        Counts how many pairs (a, b) in S form a bad triple (a, b, x).
        i.e., g[a][b] == g[a][x] == g[b][x]
        """
        count = 0
        # Iterate over pairs in S
        # Optimization: If S is a list, we can use indices
        for i in range(len(S)):
            a = S[i]
            g_ax = g[a][x]
            for j in range(i+1, len(S)):
                b = S[j]
                if g[a][b] == g_ax == g[b][x]:
                    count += 1
        return count

    def get_score(S, bad_count):
        # We want to maximize size, but penalize violations heavily
        # If valid, score is size. If invalid, score is size - penalty * bad_count
        # Penalty should be large enough to prefer valid sets of smaller size over invalid sets of larger size?
        # Actually, we want to find the largest VALID set.
        # So, if bad_count > 0, we should try to reduce it.
        # Let's use: Score = len(S) - 1000 * bad_count
        return len(S) - 1000 * bad_count

    # Initial Greedy Construction
    S = []
    S_set = set()
    for x in range(1, N+1):
        # Check if x can be added without creating violations
        # Since S is currently valid, we just need to ensure adding x doesn't create any bad triple
        ok = True
        for i in range(len(S)):
            a = S[i]
            g_ax = g[a][x]
            for j in range(i+1, len(S)):
                b = S[j]
                if g[a][b] == g_ax == g[b][x]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            S.append(x)
            S_set.add(x)
    
    best_S = list(S)
    best_score = len(S) # bad_count is 0
    
    # Simulated Annealing
    if time.time() < deadline - 5:
        T = 10.0 # Initial temperature
        cooling = 0.999
        min_T = 0.01
        
        # State: S (list), S_set (set), bad_count (int)
        current_bad_count = 0
        current_score = get_score(S, current_bad_count)
        
        while time.time() < deadline - 2 and T > min_T:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random element not in S
                candidates = [x for x in range(1, N+1) if x not in S_set]
                if not candidates:
                    T *= cooling
                    continue
                x = random.choice(candidates)
                
                # Calculate delta
                # New bad triples involving x
                new_bads = count_bad_triples_involving(x, S)
                delta_bad = new_bads
                delta_score = 1 - 1000 * delta_bad # +1 size, +delta_bad * penalty
                
                if delta_score > 0 or random.random() < pow(2.718, delta_score / T):
                    # Accept move
                    S.append(x)
                    S_set.add(x)
                    current_bad_count += delta_bad
                    current_score += delta_score
                    
                    if current_bad_count == 0 and len(S) > best_score:
                        best_score = len(S)
                        best_S = list(S)
                        
            elif move_type < 0.6:
                # Remove a random element from S
                if not S:
                    T *= cooling
                    continue
                idx = random.randint(0, len(S)-1)
                x = S[idx]
                
                # Calculate delta
                # Remove bad triples involving x
                old_bads = count_bad_triples_involving(x, S)
                delta_bad = -old_bads
                delta_score = -1 - 1000 * delta_bad # -1 size, -delta_bad * penalty
                
                if delta_score > 0 or random.random() < pow(2.718, delta_score / T):
                    # Accept move
                    S.pop(idx)
                    S_set.remove(x)
                    current_bad_count += delta_bad
                    current_score += delta_score
                    
            else:
                # Swap: Remove x, Add y
                if not S:
                    T *= cooling
                    continue
                idx = random.randint(0, len(S)-1)
                x = S[idx]
                
                candidates = [y for y in range(1, N+1) if y not in S_set]
                if not candidates:
                    T *= cooling
                    continue
                y = random.choice(candidates)
                
                # Calculate delta
                # 1. Remove x: reduces bad_count by old_bads_x
                old_bads_x = count_bad_triples_involving(x, S)
                # 2. Add y to S without x: increases bad_count by new_bads_y
                # We need to count bad triples (a, b, y) where a, b in S \ {x}
                # And also check if (x, a, y) or (x, b, y) were bad? No, x is removed.
                # So we just count bad triples involving y in the set S \ {x}
                
                # Efficient calculation:
                # S_without_x = S with x removed
                # We can just iterate S and skip x
                new_bads_y = 0
                for i in range(len(S)):
                    if S[i] == x:
                        continue
                    a = S[i]
                    g_ay = g[a][y]
                    for j in range(i+1, len(S)):
                        if S[j] == x:
                            continue
                        b = S[j]
                        if g[a][b] == g_ay == g[b][y]:
                            new_bads_y += 1
                
                delta_bad = -old_bads_x + new_bads_y
                delta_score = 0 - 1000 * delta_bad # Size unchanged
                
                if delta_score > 0 or random.random() < pow(2.718, delta_score / T):
                    # Accept move
                    S[idx] = y
                    S_set.remove(x)
                    S_set.add(y)
                    current_bad_count += delta_bad
                    current_score += delta_score
                    
                    if current_bad_count == 0 and len(S) > best_score:
                        best_score = len(S)
                        best_S = list(S)
            
            T *= cooling
            
            # Optional: Local search polish if we find a valid set
            # If current_bad_count == 0, try to add more elements greedily
            if current_bad_count == 0:
                improved = True
                while improved and time.time() < deadline - 2:
                    improved = False
                    for z in range(1, N+1):
                        if z in S_set:
                            continue
                        # Check if z can be added
                        ok = True
                        for i in range(len(S)):
                            a = S[i]
                            g_az = g[a][z]
                            for j in range(i+1, len(S)):
                                b = S[j]
                                if g[a][b] == g_az == g[b][z]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            S.append(z)
                            S_set.add(z)
                            improved = True
                            if len(S) > best_score:
                                best_score = len(S)
                                best_S = list(S)
                            break # Restart loop after adding one

    return best_S if best_S else [1]
