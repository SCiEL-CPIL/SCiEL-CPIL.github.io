import sys
sys.setrecursionlimit(10000)

def construct():
    P = 27720
    best_m = 0
    best_L = 0
    
    def check(m, L):
        # Returns True if there is a matching for window (m, m+L]
        # Indices 1..12
        # Candidates for i are numbers in (m, m+L] divisible by i
        
        # Build list of candidate indices for each i
        # Index k corresponds to number m+1+k
        cands = [[] for _ in range(13)]
        
        for i in range(1, 13):
            start = m + 1
            end = m + L
            rem = start % i
            if rem == 0:
                first = start
            else:
                first = start + (i - rem)
            
            if first > end:
                return False
            
            # Add indices to cands[i]
            start_k = first - start
            for k in range(start_k, L, i):
                cands[i].append(k)
                
        # Sort indices by length of cands[i]
        indices = sorted(range(1, 13), key=lambda x: len(cands[x]))
        
        used = [False] * L
        
        def dfs(idx):
            if idx == 12:
                return True
            i = indices[idx]
            for k in cands[i]:
                if not used[k]:
                    used[k] = True
                    if dfs(idx + 1):
                        return True
                    used[k] = False
            return False
            
        return dfs(0)

    for m in range(P):
        for L in range(1, 145):
            if check(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                    if best_L == 144:
                        return best_m
                break
                
    return best_m
