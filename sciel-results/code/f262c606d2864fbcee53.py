"""Find maximum pairwise non-coprime subset of [1,255] containing 255.

Strategy:
1. The set must contain 255 = 3*5*17, so every other element must share a factor with 255.
   Thus every element must be divisible by 3, 5, or 17.
2. Build the candidate set C = {x in [1,255] : gcd(x,255) > 1}.
3. Build a graph where edges connect pairs with gcd > 1.
4. Find the maximum clique in this graph (which automatically includes 255 since 255
   is connected to all candidates).
5. Use Bron-Kerbosch with pivot for exact maximum clique search.

Since 255 must be in the set, and 255 is connected to ALL candidates (every candidate
shares a factor with 255 by construction), we can find the maximum clique among the
candidates that contains 255. This is equivalent to finding the maximum clique in the
full candidate graph, since 255 is a universal vertex in that graph.

Actually, let me think more carefully. 255 is connected to every candidate by construction.
So any clique in the candidate graph can be extended by adding 255. Therefore the maximum
clique containing 255 has size = (max clique size in candidate graph) + 1 if 255 is not
already in the max clique, or just the max clique size if it is.

Wait - 255 IS in the candidate set. So the maximum clique in the candidate graph that
contains 255 has the same size as the maximum clique in the candidate graph (since 255
is universal, any max clique can include it). So we just need the max clique size in C.

Let me implement Bron-Kerbosch with pivot.
"""

import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second safety margin
    
    n = 255
    # Candidate set: all x in [1, 255] with gcd(x, 255) > 1
    candidates = [x for x in range(1, n + 1) if math.gcd(x, n) > 1]
    
    # Build adjacency: edge between i,j if gcd(candidates[i], candidates[j]) > 1
    m = len(candidates)
    # Use bitsets for efficiency
    adj = [0] * m  # adj[i] is a bitmask of neighbors of i
    
    for i in range(m):
        for j in range(i + 1, m):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Find maximum clique using Bron-Kerbosch with pivot
    best_clique = []
    
    def bronk(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Choose pivot: vertex in P ∪ X with max degree in P
        # For efficiency, pick from X first (they don't add to R)
        pivot_candidates = P | X
        if pivot_candidates == 0:
            return
        
        # Find pivot with maximum number of neighbors in P
        best_pivot = -1
        best_count = -1
        # Check X first, then P
        temp = pivot_candidates
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            temp ^= v_bit
            count = (adj[v] & P).bit_count()
            if count > best_count:
                best_count = count
                best_pivot = v
        
        # Iterate over P \ N(pivot)
        if best_pivot >= 0:
            to_process = P & ~(adj[best_pivot])
        else:
            to_process = P
        
        while to_process:
            v_bit = to_process & (-to_process)
            v = v_bit.bit_length() - 1
            to_process ^= v_bit
            P_new = P & adj[v]
            X_new = X & adj[v]
            bronk(R | v_bit, P_new, X_new)
            P ^= v_bit
            X |= v_bit
    
    # Initial call
    all_bits = (1 << m) - 1
    bronk(0, all_bits, 0)
    
    # Ensure 255 is in the result
    if 255 in best_clique:
        result = best_clique
    else:
        # This shouldn't happen since 255 is universal, but just in case
        result = best_clique + [255]
    
    return sorted(set(result))


# Test the solution
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify pairwise non-coprime
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
