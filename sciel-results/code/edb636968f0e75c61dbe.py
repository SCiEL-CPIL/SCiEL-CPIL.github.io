import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Weights
    N = 512
    weights = [0.0] * N
    for i in range(1, N):
        weights[i] = 1.0 / (N - i)
        
    # Sort candidates by weight descending
    candidates = list(range(1, N))
    candidates.sort(key=lambda x: weights[x], reverse=True)
    
    # Greedy Initialization
    chosen = []
    for a in candidates:
        if all(math.gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    # Local Search
    # Precompute coprime masks? 
    # For each a, coprime_to[a] is a set of b such that gcd(a,b)==1
    # But 511*511 is ~260k, which is small.
    coprime = [[False] * N for _ in range(N)]
    for i in range(1, N):
        for j in range(i, N):
            if math.gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True
                
    # Helper to check if a can be added to a set S
    def can_add(a, S):
        for b in S:
            if not coprime[a][b]:
                return False
        return True
        
    # Helper to get score
    def get_score(S):
        return sum(weights[a] for a in S)
        
    current_score = get_score(chosen)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        # Try adding a new element
        for a in candidates:
            if a not in chosen:
                if can_add(a, chosen):
                    chosen.append(a)
                    current_score += weights[a]
                    improved = True
                    if time.time() > deadline:
                        break
                    
        if time.time() > deadline:
            break
            
        # Try swapping: remove u, add v
        # To be efficient, we can iterate over u in chosen and v in candidates
        # But O(N^2) per iteration might be slow if many iterations.
        # However, N=511 is small.
        
        # Optimization: Only try swaps if they might improve.
        # Sort chosen by weight?
        
        for u in list(chosen):
            # Try to replace u with v
            # v must be coprime to all in chosen \ {u}
            rest = [x for x in chosen if x != u]
            for v in candidates:
                if v == u or v in chosen:
                    continue
                # Quick check: weight must be higher
                if weights[v] <= weights[u]:
                    continue
                if can_add(v, rest):
                    # Perform swap
                    chosen.remove(u)
                    chosen.append(v)
                    current_score += weights[v] - weights[u]
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
                
    return chosen
