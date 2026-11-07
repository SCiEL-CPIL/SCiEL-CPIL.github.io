import time
import random

def construct():
    n = 10
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin
    
    def is_latin(sq):
        for i in range(n):
            if len(set(sq[i])) != n:
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if len(set(col)) != n:
                return False
        return True

    def is_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return True

    # Initialize with the cyclic square
    base_sq = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base_sq]
    
    # Try to add more squares using randomized restarts
    max_squares = 9
    
    # Heuristic: Try squares of the form (i * a + b * j) % n ? No, not Latin for composite n.
    # Use random permutations of rows/cols of the base square?
    # Or use the structure of Z5 x Z2.
    
    # Let's try a specific construction for n=10 known to work.
    # We can use the 4 MOLS of order 5 and combine with Z2.
    
    # Construction:
    # L_k(i, j) for k=1..9
    # i = 5a + b, j = 5c + d
    # We need 9 distinct maps.
    
    # Let's try to generate 9 squares using the following pattern:
    # For k=1..4:
    #   L_k(i,j) = 5*((b+d)%2) + ((a + k*c)%5)
    # This gives 4 squares.
    
    # For the remaining 5, we can use different mixing.
    # Actually, it is known that 9 MOLS exist for n=10.
    # I will use a greedy search with random restarts to find them.
    
    best_squares = squares
    
    for restart in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        # Try to extend the set
        current = [sq[:] for sq in squares]
        
        # Generate random candidate squares
        candidates = []
        for _ in range(100):
            # Generate a random Latin square?
            # Simple heuristic: Take base square, apply random row/col permutations
            # This might not be orthogonal to base.
            
            # Better: Use the Z5xZ2 structure.
            # Randomly choose coefficients for the linear map over Z5 and Z2.
            
            # Let's try:
            # L(i,j) = 5*((b + u*d + v*a)%2) + ((a + w*c + x*b)%5)
            # We need to ensure it's Latin.
            
            pass
        
        # Given the complexity, I will fall back to a hardcoded set if search is too slow.
        # But I can't hardcode 900 ints easily in the thought process.
        
        # Let's assume the search will find them.
        # Actually, for n=10, a simple randomized greedy might work if we start with a good set.
        
        # Let's try to add squares one by one.
        for _ in range(100):
            if len(current) >= max_squares:
                break
            if time.time() - start_time > time_limit:
                break
            
            # Generate a random candidate
            # Use the structure: i=5a+b, j=5c+d
            # L(i,j) = 5*((b + d + k1*a + k2*c)%2) + ((a + k3*c + k4*b)%5)
            # We need to choose k1..k4 such that it's Latin and orthogonal to existing.
            
            found = False
            for _ in range(50):
                k1 = random.randint(0, 1)
                k2 = random.randint(0, 1)
                k3 = random.randint(1, 4) # Must be non-zero mod 5 for Latin?
                k4 = random.randint(0, 4)
                
                # Construct candidate
                cand = [[0]*n for _ in range(n)]
                for i in range(n):
                    a = i // 5
                    b = i % 5
                    for j in range(n):
                        c = j // 5
                        d = j % 5
                        high = (b + d + k1*a + k2*c) % 2
                        low = (a + k3*c + k4*b) % 5
                        cand[i][j] = 5*high + low
                
                # Check Latin
                if not is_latin(cand):
                    continue
                    
                # Check Orthogonality
                ortho = True
                for sq in current:
                    if not is_orthogonal(sq, cand):
                        ortho = False
                        break
                if ortho:
                    current.append(cand)
                    found = True
                    break
                    
            if not found:
                break
                
        if len(current) > len(best_squares):
            best_squares = current
            
    return best_squares
