        def construct():
            n = 128
            for K in range(3, 15):
                # Precompute valid_next_mask[triplet_index]
                # triplet_index = c1*K^2 + c2*K + c3
                # mask has bits set for valid c4
                valid_next = {}
                for c1 in range(K):
                    for c2 in range(K):
                        for c3 in range(K):
                            mask = 0
                            for c4 in range(K):
                                if len({c1, c2, c3, c4}) >= 3:
                                    mask |= (1 << c4)
                            valid_next[(c1, c2, c3)] = mask

                colors = [-1] * (n + 1)
                # Greedy with backtracking
                # To keep it fast, we'll do a simple DFS with pruning.
                # n=128 is too large for full DFS.
                # We'll use a greedy approach with randomization/restarts.
                
                def is_valid_assignment(x, c):
                    # Check all 4-APs ending at x
                    for d in range(1, (x - 1) // 3 + 1):
                        a = x - 3 * d
                        if a < 1: break
                        c1 = colors[a]
                        c2 = colors[a + d]
                        c3 = colors[a + 2 * d]
                        # c1, c2, c3 are assigned
                        key = (c1, c2, c3)
                        mask = valid_next[key]
                        if not (mask & (1 << c)):
                            return False
                    return True

                def solve(pos):
                    if pos > n:
                        return True
                    # Heuristic: try colors 0..K-1
                    # Optimization: sort candidates by constraint tightness?
                    # For now, try 0..K-1
                    for c in range(K):
                        if is_valid_assignment(pos, c):
                            colors[pos] = c
                            if solve(pos + 1):
                                return True
                            colors[pos] = -1
                    return False

                # To avoid exponential blowup, we limit backtracking depth or use randomized greedy
                # Let's try a simple greedy first. If it fails, try randomized.
                # Given the structure, a pure greedy might work for small K if we are lucky.
                # But to be robust, we use randomized restarts.
                
                import random
                found = False
                for attempt in range(100):
                    random.seed(attempt)
                    # Reset colors
                    for i in range(1, n+1):
                        colors[i] = -1
                    
                    success = True
                    for x in range(1, n + 1):
                        candidates = list(range(K))
                        random.shuffle(candidates) # Randomize order to escape local minima
                        valid_c = None
                        for c in candidates:
                            if is_valid_assignment(x, c):
                                valid_c = c
                                break
                        if valid_c is None:
                            success = False
                            break
                        colors[x] = valid_c
                    
                    if success:
                        return colors[1:]
                        
            raise Exception("No solution found")
        