import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Helper: Check if a square is Latin
    def is_latin(square):
        for i in range(n):
            if sorted(square[i]) != list(range(n)):
                return False
            col = [square[j][i] for j in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    # Helper: Check orthogonality
    def are_orthogonal(s1, s2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pairs.add((s1[i][j], s2[i][j]))
        return len(pairs) == n * n

    # Generate a random Latin Square
    def generate_random_latin_square(n, seed):
        random.seed(seed)
        # Start with cyclic
        base = [[(i + j) % n for j in range(n)] for i in range(n)]
        
        # Random row perm
        rows = list(range(n))
        random.shuffle(rows)
        
        # Random col perm
        cols = list(range(n))
        random.shuffle(cols)
        
        # Random symbol perm
        syms = list(range(n))
        random.shuffle(syms)
        
        sq = [[syms[base[rows[i]][cols[j]]] for j in range(n)] for i in range(n)]
        
        # Check if it's valid (it should be, but just in case of logic errors)
        if not is_latin(sq):
            return None
            
        return sq

    # Fast Backtracker for Orthogonal Mate
    def find_orthogonal_mate(L1, time_limit):
        n = 10
        ALL_MASK = (1 << n) - 1
        L2 = [[-1] * n for _ in range(n)]
        row_mask = [0] * n
        col_mask = [0] * n
        val_mask = [0] * n # val_mask[v] tracks which L2 symbols have been used with L1 symbol v
        
        def backtrack(r, c):
            if time.time() > time_limit:
                return False
            
            if c == n:
                c = 0
                r += 1
            if r == n:
                return True
            
            v = L1[r][c]
            # Available symbols:
            # 1. Not used in row r
            # 2. Not used in col c
            # 3. Not used in val_mask[v] (i.e., this L2 symbol hasn't been paired with v yet)
            
            avail = ALL_MASK & ~row_mask[r] & ~col_mask[c] & ~val_mask[v]
            
            if avail == 0:
                return False
            
            # Iterate through set bits in avail
            # randomize order for diversity? 
            # For speed, just iterate. Or shuffle.
            # Let's shuffle to avoid getting stuck in local minima if we had choices, 
            # but here the constraint is tight.
            
            # Extract bits
            bits = []
            temp = avail
            while temp:
                lsb = temp & -temp
                bit_idx = lsb.bit_length() - 1
                bits.append(bit_idx)
                temp ^= lsb
            
            random.shuffle(bits)
            
            for s in bits:
                bit = 1 << s
                L2[r][c] = s
                row_mask[r] |= bit
                col_mask[c] |= bit
                val_mask[v] |= bit
                
                if backtrack(r, c + 1):
                    return True
                
                L2[r][c] = -1
                row_mask[r] &= ~bit
                col_mask[c] &= ~bit
                val_mask[v] &= ~bit
            
            return False
        
        # Try a few random restarts with different shuffling seeds
        for attempt in range(50):
            random.seed(attempt * 12345 + 67890)
            L2 = [[-1] * n for _ in range(n)]
            row_mask = [0] * n
            col_mask = [0] * n
            val_mask = [0] * n
            
            if backtrack(0, 0):
                return L2
            
            if time.time() > time_limit:
                break
                
        return None

    best_result = None
    
    # Try to find a pair
    # We need a non-cyclic L1. 
    # Cyclic L1 (isotopic to Z_10) has no orthogonal mate.
    # Random L1s are usually not cyclic.
    
    for seed in range(100):
        if time.time() > deadline - 5:
            break
            
        L1 = generate_random_latin_square(n, seed)
        if L1 is None:
            continue
            
        # Optional: Quick check if L1 is cyclic? 
        # It's hard to check isotopy quickly. 
        # Just try to find the mate. If L1 is cyclic, the backtracker will fail (likely quickly due to constraints, or slowly).
        # To be safe, we limit time per attempt.
        
        time_per_attempt = min(2.0, deadline - 5 - time.time())
        if time_per_attempt <= 0:
            break
            
        L2 = find_orthogonal_mate(L1, time.time() + time_per_attempt)
        
        if L2 is not None:
            # Verify
            if is_latin(L2) and are_orthogonal(L1, L2):
                best_result = [L1, L2]
                break

    if best_result is None:
        # Fallback to a single square if no pair found (should not happen for order 10 with enough time)
        # But better to return a valid single square than nothing.
        L1 = generate_random_latin_square(n, 42)
        if L1:
            best_result = [L1]
        else:
            best_result = [[(i + j) % 10 for j in range(10)] for i in range(10)]

    # Ensure types are Python int
    final_result = []
    for sq in best_result:
        final_result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        
    return final_result
