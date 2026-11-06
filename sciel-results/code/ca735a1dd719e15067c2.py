import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 512
    # adj[i] is a bitmask for vertex i (representing number i+1)
    # adj[i] has bit j set if lcm(i+1, j+1) <= N
    adj = [0] * N
    
    for i in range(N):
        num_i = i + 1
        mask = 0
        for j in range(N):
            num_j = j + 1
            # Optimization: if num_i * num_j // gcd(num_i, num_j) <= N
            # Since gcd >= 1, num_i * num_j <= N * gcd
            # Simple check is fine for N=512
            g = gcd(num_i, num_j)
            if (num_i * num_j) // g <= N:
                mask |= (1 << j)
        adj[i] = mask

    best_clique = []
    best_size = 0
    
    # Bron-Kerbosch with Pivoting
    # P: candidate set (bitmask)
    # X: excluded set (bitmask)
    # current_clique: list of vertices in current path
    
    def bron_kerbosch(P, X, current_clique):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        # If P and X are empty, we found a maximal clique
        if P == 0 and X == 0:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique[:]
            return
            
        # Pivot selection: choose u in P with max neighbors in P
        # To speed up, we can just pick the first one or a heuristic one.
        # Standard heuristic: max degree in P.
        max_deg = -1
        pivot = -1
        p_temp = P
        while p_temp:
            # Get lowest set bit
            lsb = p_temp & -p_temp
            idx = (lsb.bit_length() - 1)
            deg = (adj[idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = idx
            p_temp ^= lsb
            
        if pivot == -1:
            return
            
        # Iterate over candidates in P \ N(pivot)
        # N(pivot) is adj[pivot]
        # Candidates are in P and NOT in adj[pivot]
        candidates = P & ~adj[pivot]
        
        c_temp = candidates
        while c_temp:
            lsb = c_temp & -c_temp
            v = (lsb.bit_length() - 1)
            c_temp ^= lsb
            
            # Recurse
            # New P: P & N(v)
            # New X: X & N(v)
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bron_kerbosch(new_P, new_X, current_clique + [v])
            
            # Remove v from P, add to X
            P ^= lsb
            X |= lsb
            # Note: We update P and X in the loop. 
            # Wait, standard BK modifies P and X in place or passes copies?
            # Usually:
            # for v in P:
            #   BK(P & N(v), X & N(v), C + [v])
            #   P = P - {v}
            #   X = X + {v}
            # My loop above does this correctly if I be careful with P and X modification.
            # However, `P` and `X` are integers (bitmasks), so they are immutable.
            # The logic `P ^= lsb` and `X |= lsb` updates the local variables for the next iteration.
            # This is correct.

    # Initial call: P is all vertices, X is empty
    initial_P = (1 << N) - 1
    initial_X = 0
    bron_kerbosch(initial_P, initial_X, [])
    
    # Convert indices to numbers (index i -> number i+1)
    result = [i + 1 for i in best_clique]
    
    # Fallback if search failed or timed out without finding anything (should not happen)
    if not result:
        result = list(range(1, 23))
        
    return result
