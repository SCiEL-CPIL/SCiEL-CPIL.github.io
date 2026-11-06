    def evaluate_greedy(init_residues):
        res = init_residues[:]
        cover_count = [0] * (L_max + 1)
        
        for i, p in enumerate(primes):
            r = res[i]
            start = r if r > 0 else p
            for y in range(start, L_max + 1, p):
                cover_count[y] += 1
        
        L = 0
        for x in range(1, L_max + 1):
            if time.time() > deadline:
                break
                
            while cover_count[x] < 2:
                found = False
                # Try to fix x
                # Iterate primes
                # Heuristic: Try primes that currently do NOT cover x first?
                # Or try large primes first?
                
                for idx in range(n_primes - 1, -1, -1):
                    p = primes[idx]
                    old_r = res[idx]
                    new_r = x % p
                    
                    if old_r == new_r:
                        continue
                        
                    c_x = cover_count[x]
                    delta_x = 0
                    if x % p == old_r: delta_x -= 1
                    if x % p == new_r: delta_x += 1
                    
                    if c_x + delta_x < 2:
                        continue
                        
                    # Check previous y
                    valid = True
                    start_y = old_r if old_r > 0 else p
                    if start_y <= x:
                        for y in range(start_y, x + 1, p):
                            if y == x: continue
                            # y is congruent to old_r
                            # delta is -1 (since old_r != new_r)
                            if cover_count[y] == 2:
                                valid = False
                                break
                            
                    if valid:
                        # Apply
                        for y in range(start_y, L_max + 1, p):
                            cover_count[y] -= 1
                        start_y_new = new_r if new_r > 0 else p
                        for y in range(start_y_new, L_max + 1, p):
                            cover_count[y] += 1
                        res[idx] = new_r
                        found = True
                        break
                if not found:
                    return L, res
            
            L = x
            
        return L, res
