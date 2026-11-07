import time


def construct():
            P = 27720
            # Precompute masks
            masks = [0] * P
            for x in range(P):
                m = 0
                for i in range(1, 13):
                    if x % i == 0:
                        m |= (1 << (i-1))
                masks[x] = m
            
            best_r = 0
            best_f = 0
            deadline = time.time() + 85
            
            # Helper for matching
            # adj[i] = list of right node indices (0..L-1) that can match divisor i+1
            # We will build adj on the fly for each L.
            
            for r in range(P):
                if time.time() > deadline:
                    break
                
                # We want min L such that matching exists.
                # Start L at 12.
                found = False
                for L in range(12, 145):
                    # Check if matching exists for window (r, r+L]
                    # Right nodes: indices 0..L-1 corresponding to r+1..r+L
                    # adj[i] for i in 0..11
                    adj = [[] for _ in range(12)]
                    
                    # Build adj
                    # Optimization: local variables
                    ms = masks
                    base = r
                    for k in range(L):
                        num_idx = (base + k + 1) % P
                        mask = ms[num_idx]
                        # For each divisor i, if mask has bit i, add k to adj[i]
                        # Unroll the loop for speed?
                        if mask & 1: adj[0].append(k)
                        if mask & 2: adj[1].append(k)
                        if mask & 4: adj[2].append(k)
                        if mask & 8: adj[3].append(k)
                        if mask & 16: adj[4].append(k)
                        if mask & 32: adj[5].append(k)
                        if mask & 64: adj[6].append(k)
                        if mask & 128: adj[7].append(k)
                        if mask & 256: adj[8].append(k)
                        if mask & 512: adj[9].append(k)
                        if mask & 1024: adj[10].append(k)
                        if mask & 2048: adj[11].append(k)
                    
                    # Run matching
                    # match_r: dict or list? List of size L, init -1
                    match_r = [-1] * L
                    
                    def dfs(u, seen):
                        for v in adj[u]:
                            if v in seen:
                                continue
                            seen.add(v)
                            if match_r[v] == -1 or dfs(match_r[v], seen):
                                match_r[v] = u
                                return True
                        return False
                        
                    count = 0
                    for u in range(12):
                        if dfs(u, set()):
                            count += 1
                            if count == 12:
                                break
                    if count == 12:
                        if L > best_f:
                            best_f = L
                            best_r = r
                        found = True
                        break
                if not found:
                    # Should not happen, but if it does, best_f remains
                    pass
            
            return int(best_r)
