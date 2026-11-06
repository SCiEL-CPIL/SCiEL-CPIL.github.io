import time
import sys

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    # Precompute powers of 2 for bit masks? Not strictly necessary, 1 << i is fast.
    
    best_edges = 0
    best_adj = [0] * n
    
    # Backtracking state:
    # k: current vertex index to assign neighbors for
    # current_edges: number of edges in adj[0..k-1]
    
    # Symmetry breaking:
    # We can assume WLOG that vertex 0 is connected to as many vertices as possible?
    # Or we can just search all possibilities.
    # With n=20, the search space is small enough with strong pruning.
    
    def is_valid_subset(k, S):
        # S is a bitmask of neighbors of k from {0, ..., k-1}
        # Check constraint 1: For all i in S, |S & N(i)| <= 2
        # Check constraint 2: For all distinct i, j in S, |N(i) & N(j)| <= 1
        
        # Get list of bits in S
        # Optimization: Iterate through bits
        
        # Check 1
        temp_S = S
        while temp_S:
            lsb = temp_S & -temp_S
            i = lsb.bit_length() - 1
            temp_S -= lsb
            # Count bits in S & adj[i]
            if (S & adj[i]).bit_count() > 2:
                return False
        
        # Check 2
        # We need to check pairs in S.
        # If S has > 2 bits, we check pairs.
        # Actually, if |N(i) & N(j)| is already 2, we can't add k to both.
        # So we need |N(i) & N(j)| <= 1 for all i, j in S.
        
        # Efficient check:
        # For each i in S, we can't just check pairs naively if S is large, but S is small (<=19).
        # However, we can precompute or just check.
        # Let's check pairs.
        bits_in_S = []
        temp_S = S
        while temp_S:
            lsb = temp_S & -temp_S
            i = lsb.bit_length() - 1
            temp_S -= lsb
            bits_in_S.append(i)
            
        for idx1 in range(len(bits_in_S)):
            i = bits_in_S[idx1]
            for idx2 in range(idx1 + 1, len(bits_in_S)):
                j = bits_in_S[idx2]
                if (adj[i] & adj[j]).bit_count() > 1:
                    return False
        
        return True

    def backtrack(k, current_edges):
        nonlocal best_edges, best_adj
        
        if time.time() > deadline:
            return
        
        if k == n:
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
            return
        
        # Upper bound: current_edges + sum of max possible edges for k..n-1
        # Max edges for vertex j is j (connecting to 0..j-1)
        # Sum_{j=k}^{n-1} j = (n-1)*n/2 - (k-1)*k/2
        max_remaining = ((n - 1) * n // 2) - ((k - 1) * k // 2)
        if current_edges + max_remaining <= best_edges:
            return
            
        # Try all subsets S of {0, ..., k-1}
        # To maximize edges, we should try larger subsets first?
        # Or we can iterate through all 2^k subsets.
        # For k=19, 2^19 = 524,288. This is manageable if pruning is strong.
        
        # Optimization: Generate subsets in order of decreasing popcount?
        # This helps find good solutions early, improving the bound.
        
        # Let's just iterate all masks from (1<<k)-1 down to 0.
        # This tries larger sets first (if we interpret binary weight loosely, but actually just numeric).
        # Better: Iterate by popcount descending.
        
        # For simplicity and speed, let's just iterate all masks.
        # But we can prune if the mask is invalid.
        
        for S in range((1 << k) - 1, -1, -1):
            if not is_valid_subset(k, S):
                continue
            
            # Apply S
            adj[k] = S
            for i in range(k):
                if S & (1 << i):
                    adj[i] |= (1 << k)
            
            backtrack(k + 1, current_edges + S.bit_count())
            
            # Undo S
            adj[k] = 0
            for i in range(k):
                if S & (1 << i):
                    adj[i] &= ~(1 << k)
            
            if time.time() > deadline:
                return

    # Initial call
    # Vertex 0 has no neighbors.
    backtrack(0, 0)
    
    # Extract edges
    edges = []
    for u in range(n):
        mask = best_adj[u]
        k = mask
        while k:
            lsb = k & -k
            v = lsb.bit_length() - 1
            k -= lsb
            if u < v:
                edges.append([u, v])
                
    return edges

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Edges: {len(edges)}")
    # Verify validity
    n = 20
    adj = [0]*n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = (adj[u] & adj[v]).bit_count()
            if common >= 3:
                valid = False
                print(f"Invalid: {u}, {v} have {common} common neighbors")
    print(f"Valid: {valid}")
