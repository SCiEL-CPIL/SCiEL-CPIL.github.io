import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
        n = P['n']
        deadline = time.time() + 55.0
        
        # Build adjacency bitmask
        adj = [0] * n
        for u, v in P['edges']:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        full = (1 << n) - 1
        N = 1 << n
        
        # Precompute is_clique and is_indep
        is_clique = [False] * N
        is_indep = [False] * N
        is_clique[0] = True
        is_indep[0] = True
        
        for S in range(1, N):
            lb = S & (-S)
            v = lb.bit_length() - 1
            rest = S ^ lb
            if is_clique[rest] and (adj[v] & rest) == rest:
                is_clique[S] = True
            if is_indep[rest] and (adj[v] & rest) == 0:
                is_indep[S] = True
        
        # valid[S] = is_clique[S] or is_indep[S]
        valid = [False] * N
        for S in range(1, N):
            valid[S] = is_clique[S] or is_indep[S]
        
        # DP
        INF = n + 1
        dp = [INF] * N
        parent = [0] * N
        dp[0] = 0
        
        # Iterate subsets by increasing size? No, standard subset DP works with any order as long as S^T is smaller?
        # S^T is a subset of S. If we iterate S from 1 to N-1, S^T is not necessarily < S in integer value, but it is a proper subset.
        # However, dp[S^T] must be computed before dp[S].
        # Since S^T is a subset of S, if we iterate by popcount (size), it works.
        # Or simply, since S^T < S is NOT always true (e.g., S=10, T=1, S^T=9 < 10. S=11, T=1, S^T=10 < 11. S=11, T=2, S^T=9 < 11).
        # Actually, if T is a non-empty subset of S, then S^T is a strict subset of S.
        # In integer representation, a strict subset does not necessarily have a smaller integer value?
        # Yes, it does. If A is a strict subset of B, then A < B is not guaranteed?
        # Example: B = 101 (5), A = 011 (3). 3 < 5.
        # B = 100 (4), A = 001 (1). 1 < 4.
        # B = 110 (6), A = 010 (2). 2 < 6.
        # It turns out that for any non-empty T subset of S, S^T < S is NOT always true?
        # Wait. Let S = 1010 (10). T = 0010 (2). S^T = 1000 (8). 8 < 10.
        # T = 1000 (8). S^T = 0010 (2). 2 < 10.
        # It seems S^T < S is always true if T is a non-empty subset of S?
        # Proof: Let $v$ be the most significant bit of $T$. Then $v$ is set in $S$ and $T$.
        # In $S \setminus T$, the bit $v$ is 0. In $S$, the bit $v$ is 1.
        # Bits higher than $v$ are the same in $S$ and $S \setminus T$ (since $T$ has no bits higher than $v$).
        # Thus $S \setminus T < S$.
        # So iterating S from 1 to N-1 ensures dp[S^T] is already computed.
        
        for S in range(1, N):
            if time.time() > deadline:
                # Fallback
                return [[i] for i in range(n)]
            
            best = INF
            bestT = 0
            
            # Iterate submasks of S
            T = S
            while T:
                if valid[T]:
                    prev = S ^ T
                    val = dp[prev] + 1
                    if val < best:
                        best = val
                        bestT = T
                        # Early termination if we hit the theoretical lower bound?
                        # Lower bound for dp[S] is 1 (if S is valid).
                        # If best == 1, we can break.
                        if best == 1:
                            break
                T = (T - 1) & S
            
            dp[S] = best
            parent[S] = bestT
        
        # Reconstruct
        S = full
        parts = []
        while S:
            T = parent[S]
            if T == 0:
                # This shouldn't happen if dp[full] < INF
                for i in range(n):
                    if (S >> i) & 1:
                        parts.append([i])
                S = 0
                break
            vertices = [i for i in range(n) if (T >> i) & 1]
            parts.append(vertices)
            S ^= T
        
        result = [list(map(int, p)) for p in parts]
        return result
