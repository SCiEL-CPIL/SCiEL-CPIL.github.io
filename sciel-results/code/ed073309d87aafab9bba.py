"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 10

    def is_latin_square(sq):
        for i in range(n):
            if len(set(sq[i])) != n:
                return False
        for j in range(n):
            if len(set(sq[i][j] for i in range(n))) != n:
                return False
        return True

    def is_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    def generate_random_latin_square():
        """Generate a random Latin square using randomized greedy approach."""
        # Start with empty grid
        grid = [[-1]*n for _ in range(n)]
        
        # Fill first row with random permutation
        perm = list(range(n))
        random.shuffle(perm)
        for j in range(n):
            grid[0][j] = perm[j]
        
        # Fill remaining rows
        for i in range(1, n):
            # Get available symbols for each column
            # Use backtracking with randomization
            def backtrack(row, col):
                if col == n:
                    return True
                if row == n:
                    return True
                # Get candidates for grid[row][col]
                used_in_row = set(grid[row][:col])
                used_in_col = set(grid[r][col] for r in range(row+1))
                candidates = [v for v in range(n) if v not in used_in_row and v not in used_in_col]
                random.shuffle(candidates)
                for v in candidates:
                    grid[row][col] = v
                    if backtrack(row, col+1):
                        return True
                    grid[row][col] = -1
                return False
            
            if not backtrack(i, 0):
                # Failed, restart
                return None
        
        return grid

    # Known first square
    s0 = [[(i+j) % n for j in range(n)] for i in range(n)]
    result = [s0]
    
    # Try to find more orthogonal squares using randomized search
    max_squares = 9
    attempts_per_square = 50000
    attempts = 0
    
    while len(result) < max_squares and time.time() < deadline:
        found = False
        for _ in range(attempts_per_square):
            if time.time() > deadline:
                break
            sq = generate_random_latin_square()
            if sq is None:
                continue
            if not is_latin_square(sq):
                continue
            # Check orthogonality with all existing squares
            all_orth = True
            for es in result:
                if not is_orthogonal(es, sq):
                    all_orth = False
                    break
            if all_orth:
                result.append(sq)
                found = True
                break
        if not found:
            # Try harder with more attempts
            attempts_per_square *= 2
            if attempts_per_square > 500000:
                break
    
    # Ensure at least 1 square
    if not result:
        result = [s0]
    
    # Convert to pure Python ints and lists
    final_result = []
    for sq in result:
        final_sq = [[int(sq[i][j]) for j in range(n)] for i in range(n)]
        final_result.append(final_sq)
    
    return final_result

# EVOLVE-BLOCK-END
