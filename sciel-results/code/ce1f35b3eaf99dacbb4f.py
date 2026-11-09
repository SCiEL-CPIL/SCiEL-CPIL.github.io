import random
import time
import sys

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # L1: Cyclic Latin Square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            if sorted([sq[r][i] for r in range(n)]) != list(range(n)):
                return False
        return True

    def count_conflicts(sq_a, sq_b):
        # Count how many pairs (sq_a[i][j], sq_b[i][j]) are duplicates
        pairs = set()
        conflicts = 0
        for i in range(n):
            for j in range(n):
                p = (sq_a[i][j], sq_b[i][j])
                if p in pairs:
                    conflicts += 1
                pairs.add(p)
        return conflicts

    def is_orthogonal(sq_a, sq_b):
        return count_conflicts(sq_a, sq_b) == 0

    def generate_random_latin():
        # Generate a random Latin square using a simple randomized algorithm
        # This might be slow or produce invalid squares if not careful.
        # Better: Start with L1 and apply random permutations of rows, columns, and symbols.
        # This preserves Latin property and orthogonality with other transformed squares?
        # No, it preserves the structure.
        
        # For diversity, we can use a randomized local search to create a new square
        # that is orthogonal to a given set.
        pass

    def find_orthogonal_to_set(existing_squares, timeout=10):
        # Use local search to find a square orthogonal to all in existing_squares
        # Start with a random permutation of symbols in L1? Or a random Latin square?
        # Let's start with a random Latin square.
        # We can generate one by:
        # 1. Create identity square.
        # 2. Apply random row swaps, col swaps, symbol swaps.
        # This keeps it Latin.
        
        def random_latin():
            sq = [[i for i in range(n)] for _ in range(n)]
            # Random row permutation
            rows = list(range(n))
            random.shuffle(rows)
            sq = [sq[r] for r in rows]
            # Random col permutation
            cols = list(range(n))
            random.shuffle(cols)
            sq = [[sq[r][c] for c in cols] for r in range(n)]
            # Random symbol permutation
            syms = list(range(n))
            random.shuffle(syms)
            sq = [[syms[sq[r][c]] for c in range(n)] for r in range(n)]
            return sq

        best_sq = random_latin()
        best_score = sum(count_conflicts(best_sq, es) for es in existing_squares)
        
        current_sq = [row[:] for row in best_sq]
        current_score = best_score
        
        steps = 0
        max_steps = 50000
        while time.time() < deadline - 5 and steps < max_steps:
            steps += 1
            # Pick a random row and column to swap symbols in
            r = random.randint(0, n-1)
            c1 = random.randint(0, n-1)
            c2 = random.randint(0, n-1)
            if c1 == c2:
                continue
            
            # Swap current_sq[r][c1] and current_sq[r][c2]
            # This might break Latin property (columns).
            # To maintain Latin property, we should swap in a way that preserves it.
            # Swapping two elements in a row breaks column uniqueness.
            # We need to swap elements in a "cycle" or use a different move.
            
            # Better move: Swap two rows? No, that's just a permutation.
            # Let's use a "2-opt" like move on the symbols in a row, but ensure columns remain valid.
            # This is tricky.
            
            # Alternative: Use a randomized restart with simulated annealing on the symbol assignment.
            # But maintaining Latin constraints is hard.
            
            # Let's try a simpler approach:
            # Since we know L1, L2, etc. exist, we can try to find them by searching over a smaller space.
            # Or, we can just return the known 4 or 5 if we can hardcode them?
            # No, we must compute.
            
            # Let's try a different local search:
            # Represent the square by the permutation of symbols in each row.
            # Row i is a permutation P_i of 0..9.
            # Constraint: For each column j, P_i[j] are all distinct.
            
            # This is a constraint satisfaction problem.
            # We can use backtracking with randomization.
            
            pass
        
        return best_sq

    # Given the complexity, let's try to hardcode a known set of 4 or 5 MOLS if possible?
    # No, the task says "compute".
    
    # Let's try a simple search for 2 squares first.
    # L1 is fixed.
    # Find L2 orthogonal to L1.
    # Then L3 orthogonal to L1, L2.
    
    # Actually, for n=10, it is known that 5 MOLS exist.
    # If we can't find them quickly, we return what we have.
    
    # Let's implement a robust random search for the next square.
    # We will use a "repair" strategy.
    
    def search_next_square(existing, time_limit=5):
        start = time.time()
        # Initial guess: random Latin square
        sq = [[(i + j) % n for j in range(n)] for i in range(n)]
        # Apply random permutations
        rows = list(range(n)); random.shuffle(rows)
        sq = [sq[r] for r in rows]
        cols = list(range(n)); random.shuffle(cols)
        sq = [[sq[r][c] for c in cols] for r in range(n)]
        syms = list(range(n)); random.shuffle(syms)
        sq = [[syms[sq[r][c]] for c in range(n)] for r in range(n)]
        
        def score(s):
            return sum(count_conflicts(s, e) for e in existing)
        
        curr_score = score(sq)
        best_sq = [row[:] for row in sq]
        best_score = curr_score
        
        while time.time() - start < time_limit:
            # Move: swap two symbols in the same row?
            # This breaks column uniqueness.
            # We need a move that preserves Latin property.
            # One such move is to swap two entire rows? No, that's a permutation.
            # Swap two symbols in a row AND adjust another row to fix columns?
            
            # Let's use a "3-cycle" move or similar.
            # Or, simpler: Randomly restart.
            # Given the time limit, random restarts might be better than complex local search.
            
            # Let's try 1000 random restarts per call.
            for _ in range(1000):
                if time.time() - start > time_limit:
                    break
                new_sq = [[(i + j) % n for j in range(n)] for i in range(n)]
                rows = list(range(n)); random.shuffle(rows)
                new_sq = [new_sq[r] for r in rows]
                cols = list(range(n)); random.shuffle(cols)
                new_sq = [[new_sq[r][c] for c in cols] for r in range(n)]
                syms = list(range(n)); random.shuffle(syms)
                new_sq = [[syms[new_sq[r][c]] for c in range(n)] for r in range(n)]
                
                s = score(new_sq)
                if s < best_score:
                    best_score = s
                    best_sq = [row[:] for row in new_sq]
                    if best_score == 0:
                        return best_sq
                        
        return best_sq

    squares = [L1]
    
    # Find L2
    if time.time() < deadline - 10:
        L2 = search_next_square(squares, time_limit=5)
        if is_orthogonal(L2, L1):
            squares.append(L2)
    
    # Find L3
    if time.time() < deadline - 10 and len(squares) > 1:
        L3 = search_next_square(squares, time_limit=5)
        if all(is_orthogonal(L3, s) for s in squares):
            squares.append(L3)
    
    # Find L4
    if time.time() < deadline - 10 and len(squares) > 2:
        L4 = search_next_square(squares, time_limit=10)
        if all(is_orthogonal(L4, s) for s in squares):
            squares.append(L4)
            
    # Find L5
    if time.time() < deadline - 5 and len(squares) > 3:
        L5 = search_next_square(squares, time_limit=20)
        if all(is_orthogonal(L5, s) for s in squares):
            squares.append(L5)

    # Ensure types are correct
    result = []
    for sq in squares:
        result.append([list(row) for row in sq])
        
    return result

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} squares")
