import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 55  # 55s limit
    
    # Fallback: Cyclic Latin Square
    fallback = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def is_latin(sq):
        for i in range(n):
            if set(sq[i]) != set(range(n)):
                return False
        for j in range(n):
            if set(sq[i][j] for i in range(n)) != set(range(n)):
                return False
        return True

    def are_orthogonal(sq1, sq2):
        seen = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in seen:
                    return False
                seen.add(p)
        return True

    def solve():
        # Try to find a pair
        for attempt in range(1000):
            if time.time() > deadline:
                return None
            
            # Generate a random Latin Square L1
            # Simple method: Start with identity, apply random row/col permutations and symbol permutations
            L1 = [[j for j in range(n)] for i in range(n)]
            # Randomize rows
            random.shuffle(L1)
            # Randomize cols
            for i in range(n):
                L1[i] = [L1[i][random.randint(0, n-1)] for _ in range(n)] # This is wrong, needs to be a permutation of indices
                # Better:
                pass
            
            # Correct random Latin Square generation:
            # 1. Start with cyclic
            L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
            # 2. Apply random row permutation
            row_perm = list(range(n))
            random.shuffle(row_perm)
            L1 = [L1[i] for i in row_perm]
            # 3. Apply random column permutation
            col_perm = list(range(n))
            random.shuffle(col_perm)
            L1 = [[L1[i][j] for j in col_perm] for i in range(n)]
            # 4. Apply random symbol permutation
            sym_perm = list(range(n))
            random.shuffle(sym_perm)
            L1 = [[sym_perm[L1[i][j]] for j in range(n)] for i in range(n)]
            
            # Now try to find L2 orthogonal to L1
            # Use backtracking
            L2 = [[-1 for _ in range(n)] for _ in range(n)]
            row_used = [set() for _ in range(n)]
            col_used = [set() for _ in range(n)]
            pair_used = set()
            
            def backtrack(i, j):
                if time.time() > deadline:
                    return False
                if i == n:
                    return True
                next_i, next_j = (i+1, 0) if j == n-1 else (i, j+1)
                
                # Candidate values for L2[i][j]
                # Must be in 0..n-1
                # Not in row_used[i]
                # Not in col_used[j]
                # (L1[i][j], val) not in pair_used
                
                candidates = [k for k in range(n) if k not in row_used[i] and k not in col_used[j] and (L1[i][j], k) not in pair_used]
                random.shuffle(candidates)
                
                for val in candidates:
                    L2[i][j] = val
                    row_used[i].add(val)
                    col_used[j].add(val)
                    pair_used.add((L1[i][j], val))
                    
                    if backtrack(next_i, next_j):
                        return True
                        
                    L2[i][j] = -1
                    row_used[i].remove(val)
                    col_used[j].remove(val)
                    pair_used.remove((L1[i][j], val))
                
                return False
            
            if backtrack(0, 0):
                if is_latin(L1) and is_latin(L2) and are_orthogonal(L1, L2):
                    return [L1, L2]
        
        return None

    result = solve()
    if result:
        return result
    else:
        return [fallback]
