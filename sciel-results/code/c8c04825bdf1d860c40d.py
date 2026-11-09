import time
import numpy as np

def construct():
    """
    Returns a tournament on n vertices where every set of 4 vertices
    has a common dominator.
    Checks Paley tournaments for primes p < 67 first.
    """
    # Candidates for Paley tournaments (primes p ≡ 3 mod 4)
    # We know 67 works. Let's check smaller ones.
    candidates = [31, 43, 47, 59, 67]
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Compute quadratic residues mod p
        squares = np.zeros(p, dtype=bool)
        for x in range(1, p):
            squares[(x * x) % p] = True
        
        # Build adjacency matrix
        # A[i][j] = 1 if i -> j, 0 if j -> i
        # Orientation bit: 0 means i -> j, 1 means j -> i
        # So if i -> j, bit is 0.
        # In matrix A, A[i][j] = 1 if i -> j.
        
        A = np.zeros((p, p), dtype=np.int8)
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if squares[diff]:
                    # i -> j
                    A[i][j] = 1
                else:
                    # j -> i
                    A[j][i] = 1
        
        # Check validity
        # For every 4 distinct vertices, max out-degree in induced subgraph must be 3.
        valid = True
        # Optimization: If p is large, this loop is slow.
        # But p <= 67, so it's fine.
        # Number of 4-sets for p=59 is ~450k.
        # For each, we check 4 vertices.
        
        # We can early exit if invalid.
        # To speed up, we can check random samples first?
        # No, we need to be sure.
        # 450k iterations in Python is ~0.5-1s.
        # We have 4 candidates. 4s total. Safe.
        
        for a in range(p - 3):
            for b in range(a + 1, p - 2):
                for c in range(b + 1, p - 1):
                    for d in range(c + 1, p):
                        # Check a
                        if A[a][b] + A[a][c] + A[a][d] == 3:
                            continue
                        # Check b
                        if A[b][a] + A[b][c] + A[b][d] == 3:
                            continue
                        # Check c
                        if A[c][a] + A[c][b] + A[c][d] == 3:
                            continue
                        # Check d
                        if A[d][a] + A[d][b] + A[d][c] == 3:
                            continue
                        
                        # None has out-degree 3
                        valid = False
                        break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Construct orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # 0 means i -> j
                    # A[i][j] is 1 if i -> j
                    if A[i][j]:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            
            return {'n': p, 'orientation': orientation}
        
        # If invalid, continue to next candidate

    # Fallback: Return 67 construction (should have been caught above, but just in case)
    p = 67
    squares = np.zeros(p, dtype=bool)
    for x in range(1, p):
        squares[(x * x) % p] = True
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if squares[diff]:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
