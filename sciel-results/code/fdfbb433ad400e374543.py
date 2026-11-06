import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    # Precompute reciprocals as Fractions for exact arithmetic
    recip_frac = [Fraction(0)] * (N + 1)
    recip_float = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip_frac[i] = Fraction(1, i)
        recip_float[i] = 1.0 / i
    
    best_score_frac = Fraction(0)
    best_S = set()
    
    # Precompute for each x, the set of elements that form a 3-AP with x
    # For x as middle: a + c = 2x, a != c, a,c in [1,N]
    # For x as endpoint: x + c = 2b or a + x = 2b
    
    # For efficiency, precompute conflict pairs
    # conflict_pairs[x] = set of y such that {x,y} is part of some 3-AP
    # Actually, we need: given S and candidate x, check if adding x creates a 3-AP
    
    def is_valid_addition(S, x):
        """Check if x can be added to S without creating a 3-AP."""
        # x as middle: for each a in S, check if 2x-a in S
        for a in S:
            c = 2 * x - a
            if c != x and c in S:
                return False
        # x as endpoint: for each a in S, check if (a+x)/2 in S
        for a in S:
            s = a + x
            if s % 2 == 0:
                c = s // 2
                if c != x and c != a and c in S:
                    return False
        return True
    
    # Heuristic: Greedy + Local Search
    def heuristic_solve():
        S = set()
        # Greedy: add smallest numbers first (they have larger reciprocals)
        for x in range(1, N + 1):
            if is_valid_addition(S, x):
                S.add(x)
        
        # 1-swap hill climb
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            # Try removing each element and adding a smaller one
            for x_out in sorted(S, reverse=True):
                if time.time() > deadline - 15.0:
                    break
                S_temp = S.copy()
                S_temp.remove(x_out)
                # Try adding elements smaller than x_out
                for x_in in range(1, x_out):
                    if x_in in S_temp:
                        continue
                    if is_valid_addition(S_temp, x_in):
                        S = S_temp
                        S.add(x_in)
                        improved = True
                        break
                if improved:
                    break
        return S
    
    # Initial heuristic solution
    S_init = heuristic_solve()
    best_score_frac = sum(recip_frac[x] for x in S_init)
    best_S = S_init.copy()
    
    # B&B with exact Fraction upper bounds
    # State: S (set), C (set of candidates), current_sum (Fraction)
    
    def dfs(S, C, current_sum, depth):
        nonlocal best_score_frac, best_S
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_sum + sum of reciprocals of all elements in C
        ub = current_sum
        for y in C:
            ub += recip_frac[y]
        
        if ub <= best_score_frac:
            return
        
        if not C:
            if current_sum > best_score_frac:
                best_score_frac = current_sum
                best_S = S.copy()
            return
        
        # Pick the smallest x in C (largest reciprocal)
        x = min(C)
        
        # Branch 1: Include x
        if is_valid_addition(S, x):
            new_S = S.copy()
            new_S.add(x)
            
            # Compute new C: remove x and all elements that conflict with new_S
            new_C = set()
            for y in C:
                if y == x:
                    continue
                if is_valid_addition(new_S, y):
                    new_C.add(y)
            
            dfs(new_S, new_C, current_sum + recip_frac[x], depth + 1)
        
        # Branch 2: Exclude x
        new_C_excl = C.copy()
        new_C_excl.remove(x)
        dfs(S, new_C_excl, current_sum, depth + 1)
    
    # Start B&B from empty set
    dfs(set(), set(range(1, N + 1)), Fraction(0), 0)
    
    return list(best_S)
