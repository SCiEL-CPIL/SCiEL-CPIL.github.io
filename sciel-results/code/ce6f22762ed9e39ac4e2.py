def construct():
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 110
    
    def is_triangle_free(adj_bits, n):
        for i in range(n):
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    if common:
                        return False
        return True
        
    def alpha_exact(adj_bits, n):
        best = 0
        
        def dfs(candidates, current_size):
            nonlocal best
            if current_size + len(candidates) <= best:
                return
            if not candidates:
                best = max(best, current_size)
                return
                
            v = candidates[0]
            rest = candidates[1:]
            
            new_cands = [u for u in rest if not ((adj_bits[v] >> u) & 1)]
            dfs(new_cands, current_size + 1)
            
            dfs(rest, current_size)
            
            if best >= 10:
                return

        dfs(list(range(n)), 0)
        return best

    def find_circulant(n):
        # Try to find a circulant graph C_n(S) that is triangle-free and has alpha < 10
        # S must be sum-free: (a+b) % n not in S for a,b in S
        # Also S must be symmetric: if a in S, then n-a in S
        
        for restart in range(500):
            if time.time() > deadline:
                return None
            
            # Generate a random symmetric sum-free set S
            # Start with random subset of {1, ..., n//2}
            # Filter to ensure sum-free
            
            # Heuristic: Start with a random set, remove pairs that sum to an element in S
            # Or start with a "large" set like odd numbers, and randomize
            
            # Let's try random subsets of a base sum-free set
            # Base sum-free set: {1, ..., n//3} is not sum-free.
            # {n//3, ..., 2n//3} is sum-free?
            # (a+b) > 2n//3. If a,b in [n//3, 2n//3], a+b in [2n//3, 4n//3].
            # Mod n, this wraps around.
            
            # Simpler: Randomly select edges, ensure triangle-free.
            # But we want circulant.
            
            # Let's pick a random set of distances D from 1 to n//2
            # Symmetrize: S = D union {n-d for d in D}
            # Check if S is sum-free.
            
            D = set()
            # Randomly add distances
            # To maximize size, start with small distances? No, small distances create triangles easily (1+1=2).
            # Large distances are safer?
            
            # Let's try to construct S greedily to be large and sum-free.
            # Candidate distances: 1, ..., n//2
            # Shuffle them
            candidates = list(range(1, n//2 + 1))
            random.shuffle(candidates)
            
            S = set()
            for d in candidates:
                # Check if adding d keeps S sum-free
                # S' = S U {d, n-d}
                # Check (a+b) % n not in S' for a,b in S'
                
                new_S = set(S)
                new_S.add(d)
                new_S.add(n - d)
                
                valid = True
                # Optimization: only check sums involving new elements?
                # No, existing sums might now collide with new elements?
                # If S was sum-free, a+b not in S.
                # Now we need a+b not in {d, n-d}.
                # And d+a, d+b, d+d, n-d+a, etc. not in S'
                
                # Full check is O(|S|^2). |S| <= n/2. n=40. 1600 ops. Fast.
                for a in new_S:
                    for b in new_S:
                        if (a + b) % n in new_S:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    S = new_S
            
            # S is now a maximal (with respect to this order) sum-free symmetric set
            # Build adj_bits
            adj_bits = [0] * n
            for i in range(n):
                for d in S:
                    j = (i + d) % n
                    adj_bits[i] |= (1 << j)
            
            # Check alpha
            alpha = alpha_exact(adj_bits, n)
            if alpha < 10:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_bits[i] >> j) & 1:
                            edges.append([i, j])
                return edges
                
        return None

    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
        edges = find_circulant(n)
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback: use a known construction
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
