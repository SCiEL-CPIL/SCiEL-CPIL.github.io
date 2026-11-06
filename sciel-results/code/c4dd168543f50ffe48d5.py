import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    # Vertices 0..255 represent numbers 1..256
    
    # 1. Identify squares
    max_sum = 2 * N
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Identify self-loop vertices (2*a is a square)
    # These cannot be in any independent set
    valid = [False] * N
    for i in range(N):
        val = i + 1
        if 2 * val in squares:
            valid[i] = False
        else:
            valid[i] = True
            
    # List of valid vertex indices
    valid_indices = [i for i in range(N) if valid[i]]
    n_valid = len(valid_indices)
    
    # Map valid index to compressed index 0..n_valid-1
    idx_map = {}
    for new_idx, old_idx in enumerate(valid_indices):
        idx_map[old_idx] = new_idx
        
    # 3. Build adjacency bitsets
    # adj[i] is a bitset of neighbors of vertex i in the compressed space
    adj = [0] * n_valid
    
    for i in range(n_valid):
        val_i = valid_indices[i] + 1
        for j in range(i + 1, n_valid):
            val_j = valid_indices[j] + 1
            s = val_i + val_j
            if s in squares:
                # Add edge i-j
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 4. Bron-Kerbosch with Pivoting to find Maximum Independent Set
    # We want the largest set of vertices with no edges between them.
    # This is equivalent to finding the largest clique in the complement graph.
    # However, implementing BK on the complement is tricky with bitsets if not careful.
    # Alternatively, BK finds Maximal Independent Sets. We track the size.
    
    best_set = []
    best_size = 0
    
    def bron_kerbosch(R, P, X):
        nonlocal best_set, best_size
        
        # Check time limit
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_set = R[:]
            return
            
        # Pivoting: choose pivot u from P union X with max degree in P
        # To maximize the number of branches, we pick u such that P \ N(u) is minimized
        # i.e., u has the most neighbors in P.
        
        # Efficiently finding pivot with bitsets
        # We want u in P|X that maximizes popcount(P & adj[u])
        # Note: adj[u] contains neighbors. For Independent Set, we want to iterate over P \ N(u).
        # So we want u with max popcount(P & adj[u]).
        
        pivot_candidates = P | X
        if pivot_candidates == 0:
            return
            
        best_pivot = -1
        max_deg = -1
        
        # Iterate through bits in pivot_candidates
        # Optimization: We can just pick a random one or the first one if graph is sparse,
        # but proper pivoting is better.
        # Since N is small (245), iterating through bits is fine.
        
        temp = pivot_candidates
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            temp ^= lsb
            
            # Count neighbors of u in P
            # adj[u] is bitset of neighbors
            deg = (P & adj[u]).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_pivot = u
                
        # Iterate over P \ N(best_pivot)
        # These are vertices that do NOT have an edge to best_pivot.
        # In the context of Independent Set, if we add a vertex v to R,
        # we must remove its neighbors from P.
        # The standard BK for Clique iterates over P \ N(u) where N(u) are neighbors.
        # For Independent Set, "neighbors" in the graph are conflicts.
        # So we iterate over vertices in P that are NOT connected to best_pivot.
        
        candidates = P & ~adj[best_pivot]
        
        temp_cand = candidates
        while temp_cand:
            lsb = temp_cand & -temp_cand
            v = lsb.bit_length() - 1
            temp_cand ^= lsb
            
            # Add v to R
            new_R = R + [v]
            
            # Remove v and its neighbors from P
            # Neighbors of v are adj[v]
            new_P = P & ~adj[v] & ~(1 << v)
            
            # Add v to X (to avoid duplicates in other branches)
            new_X = X | (1 << v)
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)
            
            # Time check inside loop
            if time.time() > deadline:
                return

    # Initial call
    # P is all valid vertices
    initial_P = (1 << n_valid) - 1
    bron_kerbosch([], initial_P, 0)
    
    # Convert best_set (compressed indices) back to original values
    result = [valid_indices[i] + 1 for i in best_set]
    
    # Ensure non-empty
    if not result:
        # Fallback: just return a single valid element if any, else 1
        if valid_indices:
            result = [valid_indices[0] + 1]
        else:
            result = [1]
            
    return result

# Verification helper (not used in construct, but good for sanity check in dev)
def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r*r == n

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i, len(res)):
            if is_square(res[i] + res[j]):
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
