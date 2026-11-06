import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Target: Maximize number of disjoint residue classes.
    # Strong candidate: All even moduli in [2, 128].
    # There are 64 even moduli.
    # We need to assign residues r_m such that for all m_i, m_j in M:
    # r_m_i != r_m_j mod gcd(m_i, m_j)
    
    M = list(range(2, 129, 2))
    N = len(M)
    
    # Precompute gcds
    gcds = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            g = gcd(M[i], M[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    best_solution = []
    solution_found = False
    
    # Order moduli: Ascending seems good as small moduli have fewer choices
    # and constrain larger ones.
    # Alternatively, sort by number of divisors descending?
    # Let's stick to ascending for now.
    
    def is_compatible(r, m_idx, current):
        """
        Check if residue r for modulus M[m_idx] is compatible 
        with all (r_k, m_k_idx) in current.
        """
        for r_k, m_k_idx in current:
            g = gcds[m_idx][m_k_idx]
            if r % g == r_k % g:
                return False
        return True

    def backtrack(idx, current):
        nonlocal best_solution, solution_found
        
        if time.time() > deadline:
            return False # Signal timeout
            
        if len(current) > len(best_solution):
            best_solution = list(current)
            
        if idx == N:
            solution_found = True
            return True
            
        m = M[idx]
        
        # Heuristic: Try residues in a randomized order? 
        # Or deterministic? Deterministic is more reproducible.
        # Let's try 0 to m-1.
        for r in range(m):
            if is_compatible(r, idx, current):
                current.append((r, idx))
                if backtrack(idx + 1, current):
                    return True
                current.pop()
                
        return False

    # Run backtracking
    # If it finds a full solution, great.
    # If it times out, best_solution holds the longest prefix.
    # However, the longest prefix might not be the best if we get stuck early.
    # To mitigate this, we can run multiple restarts with different residue orderings.
    
    # First run: Deterministic
    backtrack(0, [])
    
    if time.time() < deadline - 5.0:
        # Try randomized restarts to find better partial solutions or full solutions
        # If the first run failed to find a full solution, it might be due to a bad choice early on.
        # Randomizing the order of residues for the first few moduli might help.
        
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
                
            # Shuffle the residue order for the first few moduli?
            # Or just shuffle the moduli order?
            # Shuffling moduli order changes the constraint graph traversal.
            # But M is fixed.
            # Let's randomize the residue choice order.
            
            def backtrack_rand(idx, current):
                nonlocal best_solution, solution_found
                if time.time() > deadline:
                    return False
                if len(current) > len(best_solution):
                    best_solution = list(current)
                if idx == N:
                    solution_found = True
                    return True
                m = M[idx]
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if is_compatible(r, idx, current):
                        current.append((r, idx))
                        if backtrack_rand(idx + 1, current):
                            return True
                        current.pop()
                return False

            backtrack_rand(0, [])
            if solution_found:
                break

    # If we still don't have a good solution (e.g. < 40), 
    # fallback to the original greedy search on all moduli
    if len(best_solution) < 40:
         # Re-implement the fast greedy from the current program as fallback
         # But the current program's greedy was weak. 
         # Let's just return what we have.
         pass

    # Convert indices back to moduli
    result = []
    for r, idx in best_solution:
        result.append([r, M[idx]])
        
    # Ensure non-empty
    if not result:
        result = [[0, 2]]
        
    return result
