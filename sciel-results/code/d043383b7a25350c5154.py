"""Construct a set of mutually orthogonal Latin squares of order 10."""
import time
import random

def _validate_squares(squares):
    """Validate that all squares are Latin and pairwise orthogonal."""
    if not squares:
        return False
    n = len(squares[0])
    # Check each square is Latin
    for sq in squares:
        for r in range(n):
            if sorted(sq[r]) != list(range(n)):
                return False
        for c in range(n):
            col = [sq[r][c] for r in range(n)]
            if sorted(col) != list(range(n)):
                return False
    # Check pairwise orthogonality
    k = len(squares)
    for i in range(k):
        for j in range(i + 1, k):
            pairs = set()
            for r in range(n):
                for c in range(n):
                    pairs.add((squares[i][r][c], squares[j][r][c]))
            if len(pairs) != n * n:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    n = 10
    
    # Base square: addition table mod 10
    a0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [a0]
    
    def is_latin(square):
        for i in range(n):
            if len(set(square[i])) != n:
                return False
        for j in range(n):
            col = [square[i][j] for i in range(n)]
            if len(set(col)) != n:
                return False
        return True

    def are_orthogonal(a, b):
        seen = set()
        for i in range(n):
            for j in range(n):
                pair = (a[i][j], b[i][j])
                if pair in seen:
                    return False
                seen.add(pair)
        return True

    # Greedy backtracking search for a Latin square orthogonal to a list of existing squares
    def find_orthogonal(existing_squares, timeout):
        start_search_time = time.time()
        
        # Randomize search order
        def _search(used_pairs, current_sq, pos):
            if time.time() - start_search_time > timeout:
                return None
            if pos == n * n:
                return [row[:] for row in current_sq]
            
            i = pos // n
            j = pos % n
            
            # Try symbols in random order
            symbols = list(range(n))
            random.shuffle(symbols)
            
            for sym in symbols:
                # Check row constraint
                if sym in current_sq[i][:j]:
                    continue
                # Check column constraint
                col_has = False
                for k in range(i):
                    if current_sq[k][j] == sym:
                        col_has = True
                        break
                if col_has:
                    continue
                
                # Check orthogonality with all existing squares
                conflict = False
                for sq in existing_squares:
                    pair = (sq[i][j], sym)
                    if pair in used_pairs:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Place symbol
                current_sq[i][j] = sym
                for sq in existing_squares:
                    used_pairs.add((sq[i][j], sym))
                
                result = _search(used_pairs, current_sq, pos + 1)
                if result is not None:
                    return result
                
                # Backtrack
                for sq in existing_squares:
                    used_pairs.discard((sq[i][j], sym))
                current_sq[i][j] = -1
        
        # Initialize search
        used_pairs = set()
        current_sq = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Fix first row to identity to break symmetry
        for j in range(n):
            current_sq[0][j] = j
            for sq in existing_squares:
                used_pairs.add((sq[0][j], j))
        
        result = _search(used_pairs, current_sq, n)
        return result

    # Try to find more orthogonal squares
    attempts = 0
    while len(squares) < 9 and time.time() < deadline - 10.0:
        attempts += 1
        remaining_time = deadline - 10.0 - time.time()
        # Give each search attempt a portion of the remaining time
        attempt_timeout = min(15.0, remaining_time * 0.8)
        
        if attempt_timeout <= 1.0:
            break
        
        found = find_orthogonal(squares, attempt_timeout)
        
        if found is not None:
            # Verify
            if is_latin(found):
                ok = True
                for existing in squares:
                    if not are_orthogonal(existing, found):
                        ok = False
                        break
                if ok:
                    squares.append(found)
        
        # Shuffle for diversity
        random.seed(attempts)

    # Validate the result
    if squares and _validate_squares(squares):
        # Convert to ensure pure Python ints
        result = []
        for sq in squares:
            result.append([[int(v) for v in row] for row in sq])
        return result
    else:
        # Fallback: return just the first square
        return [[[(i + j) % 10 for j in range(10)] for i in range(10)]]
