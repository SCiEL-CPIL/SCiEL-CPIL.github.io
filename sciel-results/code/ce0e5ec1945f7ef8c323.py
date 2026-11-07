import time
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50.0
    
    squares = []
    first_square = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares.append(first_square)
    
    # Try to find more squares using backtracking
    for _ in range(8):
        if time.time() > deadline:
            break
        
        found = False
        # Use a more structured search?
        # For now, keep random but improve validity check speed
        
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            # Generate a random Latin square
            # Use a simple method: random permutation of rows, then check columns
            # This is what the current code does.
            
            square = []
            valid = True
            for i in range(n):
                row = list(range(n))
                random.shuffle(row)
                square.append(row)
            
            # Check columns
            for j in range(n):
                col = [square[i][j] for i in range(n)]
                if len(set(col)) != n:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check orthogonality
            orthogonal = True
            for sq in squares:
                pairs = set()
                for i in range(n):
                    for j in range(n):
                        pair = (sq[i][j], square[i][j])
                        if pair in pairs:
                            orthogonal = False
                            break
                        pairs.add(pair)
                    if not orthogonal:
                        break
                if not orthogonal:
                    break
            
            if orthogonal:
                squares.append(square)
                found = True
                break
        
        if not found:
            break
    
    # Convert to int
    for i in range(len(squares)):
        for j in range(n):
            for k in range(n):
                squares[i][j][k] = int(squares[i][j][k])
    
    return squares
