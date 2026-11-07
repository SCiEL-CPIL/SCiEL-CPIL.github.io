import sys

def construct():
    P = 27720
    N = 12
    
    # Precompute residues for speed
    # residues[m][i] = (-m) % i
    # We can just compute on the fly or precompute a 2D list.
    # Precomputing 27720 * 12 ints is trivial.
    residues = [[0] * (N + 1) for _ in range(P)]
    for m in range(P):
        for i in range(1, N + 1):
            residues[m][i] = (-m) % i
            
    def has_matching(m, L):
        # Build candidates for each i
        # candidates[i] is a list of x in [1, L] such that x % i == residues[m][i]
        candidates = [None] * (N + 1)
        
        for i in range(1, N + 1):
            r = residues[m][i]
            # x in [1, L] with x % i == r
            # If r == 0, x can be i, 2i, ...
            # If r > 0, x can be r, r+i, ...
            # Note: x must be >= 1.
            # If r == 0, first x is i (since x>=1 and i>=1, if i=1, x=1. if i>1, x=i).
            # Wait, if r=0, x % i == 0. Smallest x >= 1 is i? 
            # If i=1, r=0, x=1, 2, 3...
            # If i=2, r=0, x=2, 4...
            # If r > 0, smallest x is r (if r >= 1).
            
            start = r
            if start == 0:
                start = i
            # Actually, if r == 0, the sequence is i, 2i, ...
            # If r > 0, the sequence is r, r+i, ...
            # But we need x >= 1.
            # If r == 0, x=i is the first.
            # If r > 0, x=r is the first (since r < i and i>=1, r>=1 implies x>=1).
            
            cand = []
            x = start
            while x <= L:
                cand.append(x)
                x += i
                
            if not cand:
                return False
            candidates[i] = cand
            
        # Order indices by number of candidates (ascending) to prune early
        indices = list(range(1, N + 1))
        indices.sort(key=lambda i: len(candidates[i]))
        
        used = [False] * (L + 1) # used[x] is True if x is taken
        
        def dfs(k):
            if k == N:
                return True
            i = indices[k]
            for x in candidates[i]:
                if not used[x]:
                    used[x] = True
                    if dfs(k + 1):
                        return True
                    used[x] = False
            return False
            
        return dfs(0)

    # Search for maximum L from top down
    max_L = 144
    for L in range(max_L, 0, -1):
        for m in range(P):
            if has_matching(m, L):
                return m
                
    return 0
