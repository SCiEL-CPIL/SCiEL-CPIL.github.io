    best_n = 0
    best_edges = []

    # Try to find a valid Cayley graph for n from MAX_VERTICES down to 37
    for n in range(MAX_VERTICES, 36, -1):
        found_for_n = False
        
        # Try multiple restarts for each n
        for attempt in range(20):
            random.seed(n * 7919 + attempt)
            
            # Initialize S: Random symmetric subset
            S = set()
            used = set()
            for x in range(1, n):
                if x in used:
                    continue
                neg_x = (-x) % n
                if neg_x == x:
                    if random.random() < 0.5:
                        S.add(x)
                    used.add(x)
                else:
                    if neg_x in used:
                        continue
                    if random.random() < 0.5:
                        S.add(x)
                        S.add(neg_x)
                    used.add(x)
                    used.add(neg_x)
            
            current_S = set(S)
            
            # Candidates for flipping: pairs (x, -x)
            candidates = []
            for x in range(1, n):
                neg_x = (-x) % n
                if x < neg_x or (x == neg_x and x != 0):
                    candidates.append((x, neg_x))
            
            random.shuffle(candidates)
            
            found = False
            # Run local search
            for _ in range(2000):
                c1, c2 = check_cayley(n, current_S)
                if not c1 and not c2:
                    found = True
                    break
                
                if not candidates:
                    break
                    
                idx = random.randint(0, len(candidates) - 1)
                x, nx = candidates[idx]
                
                # Flip
                if x in current_S:
                    current_S.remove(x)
                else:
                    current_S.add(x)
                if nx in current_S:
                    current_S.remove(nx)
                else:
                    current_S.add(nx)
                    
                new_c1, new_c2 = check_cayley(n, current_S)
                
                # Accept if not worse
                if (new_c1 + new_c2) <= (c1 + c2):
                    pass
                else:
                    # Revert
                    if x in current_S:
                        current_S.remove(x)
                    else:
                        current_S.add(x)
                    if nx in current_S:
                        current_S.remove(nx)
                    else:
                        current_S.add(nx)
                        
            if found:
                found_for_n = True
                best_n = n
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (j - i) % n in current_S or (i - j) % n in current_S:
                            edges.append((i, j))
                best_edges = edges
                break

        if found_for_n:
            break

    # Fallback to Paley 37 if no better found
    if best_n == 0:
        best_n, best_edges = paley_graph(37)

    return best_n, best_edges
