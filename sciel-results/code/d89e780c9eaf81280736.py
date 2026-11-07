import time
import random

P = {'n': 10}

def construct():
    n = P['n']
    # L1 is the cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find L2 orthogonal to L1
    # We use backtracking to fill L2
    # Constraints:
    # 1. Row i has unique values
    # 2. Col j has unique values
    # 3. Diagonal k (where (i+j)%n == k) has unique values
    
    start_time = time.time()
    timeout = 50  # Leave 10s margin
    
    def solve():
        L2 = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        diag_used = [set() for _ in range(n)]
        
        # Order of cells to fill
        # We can try to fill row by row, but dynamic ordering is better.
        # For simplicity and speed, let's try a fixed order but with randomized values.
        # Or better, use a recursive solver with MRV.
        
        def backtrack(idx):
            if time.time() - start_time > timeout:
                return False
            
            if idx == n * n:
                return True
            
            i = idx // n
            j = idx % n
            
            # Available values
            available = set(range(n))
            available -= row_used[i]
            available -= col_used[j]
            available -= diag_used[(i + j) % n]
            
            if not available:
                return False
            
            # Randomize order to avoid getting stuck in a specific bad path
            vals = list(available)
            random.shuffle(vals)
            
            for v in vals:
                # Place
                L2[i][j] = v
                row_used[i].add(v)
                col_used[j].add(v)
                diag_used[(i + j) % n].add(v)
                
                if backtrack(idx + 1):
                    return True
                
                # Unplace
                L2[i][j] = -1
                row_used[i].discard(v)
                col_used[j].discard(v)
                diag_used[(i + j) % n].discard(v)
                
            return False
        
        # Try multiple times with different random seeds/orders if needed?
        # The fixed order (row-major) might be slow.
        # Let's try a few random restarts if the first one fails or takes too long?
        # Actually, just running the backtracker once with randomization inside is usually enough.
        # But if it hits the timeout, it returns False.
        
        # To improve success rate, we can randomize the order of cells as well.
        # But for now, let's try the simple row-major with randomized values.
        
        if backtrack(0):
            return L2
        return None

    L2 = solve()
    
    if L2:
        return [L1, L2]
    else:
        return [L1]
