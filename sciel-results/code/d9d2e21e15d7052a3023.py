import math
import time

def construct():
    """
    Construct a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # 1. Build candidate set: all x in [1,255] with gcd(x,255) > 1
    # This includes 255 itself.
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    n = len(candidates)
    
    # Map value to index
    val_to_idx = {val: i for i, val in enumerate(candidates)}
    idx_255 = val_to_idx[255]
    
    # 2. Build adjacency bitmasks
    # adj[i] is an integer bitmask where bit j is 1 if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 3. Bron-Kerbosch with pivot and bitmasks
    # R: current clique (list of indices)
    # P: candidates (bitmask)
    # X: excluded (bitmask)
    best_clique_indices = []
    best_size = 0
    # Initialize with a known valid clique: all multiples of 3 in candidates
    # This provides a good lower bound for pruning
    init_clique = [i for i, val in enumerate(candidates) if val % 3 == 0]
    if idx_255 in init_clique:
        best_clique_indices = init_clique
        best_size = len(best_clique_indices)
    else:
        # Fallback if 255 wasn't in multiples of 3 (it is, but for safety)
        best_clique_indices = [idx_255]
        best_size = 1

    def bron_kerbosch(R, P, X):
        nonlocal best_clique_indices, best_size
        
        # Time check
        if time.time() > deadline:
            return
            
        # If P and X are empty, we have a maximal clique
        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique_indices = R.copy()
            return
        
        # Pruning: If even adding all remaining candidates doesn't beat best, stop
        if len(R) + bin(P).count('1') <= best_size:
            return
        
        # Pivot selection
        # We want a pivot u in P | X that maximizes the number of neighbors in P
        # This minimizes the number of iterations in the loop below
        U = P | X
        pivot = -1
        max_neighbors = -1
        
        # Iterate through bits in U
        temp_U = U
        while temp_U:
            # Get lowest set bit index
            lsb = temp_U & -temp_U
            u = lsb.bit_length() - 1
            temp_U ^= lsb
            
            # Count neighbors of u in P
            # adj[u] & P gives bitmask of neighbors of u that are in P
            count = bin(adj[u] & P).count('1')
            if count > max_neighbors:
                max_neighbors = count
                pivot = u
                
        # The vertices to iterate over are P \ N(pivot)
        # i.e., vertices in P that are NOT adjacent to pivot
        # If pivot is -1 (should not happen if U not empty), iterate all P
        if pivot != -1:
            candidates_mask = P & ~(adj[pivot])
        else:
            candidates_mask = P
            
        # Iterate through vertices in candidates_mask
        temp_cand = candidates_mask
        while temp_cand:
            if time.time() > deadline:
                return
            
            lsb = temp_cand & -temp_cand
            v = lsb.bit_length() - 1
            temp_cand ^= lsb
            
            new_R = R + [v]
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # After returning, v is moved from P to X
            # In our implementation, we don't explicitly update P/X in the parent frame
            # because we pass copies (bitmasks are immutable).
            # However, for correctness of BK, we conceptually remove v from P.
            # Since we iterate over a fixed mask `candidates_mask` derived before recursion,
            # and we don't modify P in the current frame's scope for the loop,
            # we are effectively iterating over the original P \ N(pivot).
            # This is correct for BK with pivot.
            
    # Start with 255 forced in the clique
    R = [idx_255]
    P = adj[idx_255]
    X = 0
    
    bron_kerbosch(R, P, X)
    
    # 4. Convert best indices back to values
    result_set = set(candidates[i] for i in best_clique_indices)
    
    # Ensure 255 is in the result
    if 255 not in result_set:
        result_set.add(255)
        
    # Final validity check
    res_list = list(result_set)
    if not res_list:
        return [255]
        
    valid = True
    for i in range(len(res_list)):
        for j in range(i + 1, len(res_list)):
            if math.gcd(res_list[i], res_list[j]) == 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid or 255 not in res_list:
        # Fallback to a known valid set: all multiples of 3
        return list(range(3, 256, 3))
        
    return res_list

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
    print(f"Valid: {valid}")
