"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline, 15s safety margin
    
    N = 512
    elements = list(range(1, N + 1))
    
    # Build adjacency: edge between i and j if lcm(elements[i], elements[j]) <= N
    # Use bitsets for fast clique search
    n = len(elements)
    
    # Precompute lcm table or check on the fly
    # For 512 nodes, we can build adjacency as a list of sets or bitmasks
    # Using Python ints as bitmasks for the adjacency
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is set if elements[i] and elements[j] can be in the same set
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = elements[i], elements[j]
            lcm_val = a * b // gcd(a, b)
            if lcm_val <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Bron-Kerbosch with pivot for maximum clique
    # We want the largest clique in this graph
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        if not P:
            return
        
        # Choose pivot u from P ∪ X that maximizes |P ∩ adj[u]|
        # For speed, just pick from P if possible
        pivot_candidates = P | X
        if not pivot_candidates:
            return
        
        # Find pivot with maximum degree in P
        max_count = -1
        pivot = None
        # Iterate through bits in pivot_candidates
        temp = pivot_candidates
        while temp:
            u_bit = temp & (-temp)
            u = u_bit.bit_length() - 1
            count = bin(P & adj[u]).count('1')
            if count > max_count:
                max_count = count
                pivot = u
            if count == len(bin(P).count('1')):
                break  # Can't do better
            temp ^= u_bit
        
        # If max_count == |P|, then P is empty after removing neighbors of pivot
        if max_count == 0:
            # No edges from pivot to P, so just recurse on all of P
            # Actually if max_count is 0, P ∩ adj[pivot] is empty, so we process all P
            pass
        
        # Iterate over P \ adj[pivot]
        candidates = P & ~adj[pivot] if pivot is not None else P
        temp = candidates
        while temp:
            if time.time() > deadline:
                return
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            temp ^= v_bit
            
            bron_kerbosch(R | v_bit, P & adj[v], X & adj[v])
            P ^= v_bit
            X |= v_bit
    
    # Start with empty R, all vertices in P, empty X
    all_bits = (1 << n) - 1
    bron_kerbosch(0, all_bits, 0)
    
    if not best_clique:
        # Fallback: return at least {1}
        return [1]
    
    # Convert bitmask to actual elements
    result = []
    for i in range(n):
        if best_clique & (1 << i):
            result.append(elements[i])
    
    return result
# EVOLVE-BLOCK-END
