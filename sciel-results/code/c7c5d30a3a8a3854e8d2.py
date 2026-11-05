import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Precompute quadratic residues mod 29
    p = 29
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Generate Paley coloring for n=29
    def generate_paley(n, prime):
        res = set()
        for x in range(1, prime):
            res.add((x * x) % prime)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % prime
                # Color 0 if residue, 1 otherwise
                # Note: In Paley graph, edges are usually defined by residues.
                # We need to ensure symmetry: (j-i) % p is residue iff (i-j) % p is residue?
                # (i-j) % p = (- (j-i)) % p.
                # If p == 1 mod 4, -1 is a residue, so yes.
                # 29 = 4*7 + 1, so -1 is a residue.
                if diff in res:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Fast validity check using bitmasks
    def is_valid_fast(colors, n):
        # Build adjacency bitmasks
        # adj[0][i] is a bitmask of vertices j such that edge (i,j) is color 0
        # adj[1][i] is a bitmask of vertices j such that edge (i,j) is color 1
        adj = [[0] * n for _ in range(2)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # Check for K2,2,2 in color 0 and color 1
        for c in [0, 1]:
            A = adj[c]
            # For each pair (u1, u2)
            for u1 in range(n):
                for u2 in range(u1 + 1, n):
                    # Common neighbors of u1 and u2 in color c
                    # Must be a neighbor of both
                    S = A[u1] & A[u2]
                    # Exclude u1 and u2 themselves (though they are not neighbors of themselves in simple graph)
                    # But S might contain u1 if there was a self loop, which there isn't.
                    # However, u1 is in A[u2] if edge (u1,u2) is color c.
                    # So S contains u1 and u2?
                    # A[u2] has bit u1 set. A[u1] has bit u2 set.
                    # S = A[u1] & A[u2].
                    # Bit u1 in S: A[u1] has bit u1? No. So u1 is not in S.
                    # Bit u2 in S: A[u2] has bit u2? No. So u2 is not in S.
                    # So S is exactly the common neighbors.
                    
                    if S == 0:
                        continue
                    
                    # We need to find v1, v2 in S such that they have at least 2 common neighbors in S.
                    # Let's iterate over pairs (v1, v2) in S.
                    # To iterate over set bits in S:
                    s_bits = []
                    temp = S
                    while temp:
                        lsb = temp & -temp
                        idx_v = lsb.bit_length() - 1
                        s_bits.append(idx_v)
                        temp ^= lsb
                    
                    k = len(s_bits)
                    if k < 4:
                        continue
                        
                    # Check for K2,2 in the subgraph induced by S
                    # For each pair (v1, v2) in S:
                    for a in range(k):
                        v1 = s_bits[a]
                        for b in range(a + 1, k):
                            v2 = s_bits[b]
                            # Common neighbors of v1 and v2 in color c
                            C = A[v1] & A[v2]
                            # We need at least 2 common neighbors that are also in S
                            # C & S
                            common_in_S = C & S
                            # Count bits in common_in_S
                            if common_in_S & (common_in_S - 1):
                                # If there are at least 2 bits set
                                return False
        return True

    # Local search for n=30
    def local_search_30(colors, n, max_time=100):
        # Convert to bitmasks for speed? 
        # For now, stick to list for simplicity in flip, but use fast check?
        # Fast check is O(n^4) with small constants. 
        # n=30, n^4 = 810,000. 
        # 1000 iterations * 810,000 = 810M ops. Might be too slow in Python.
        # Let's optimize the check or reduce iterations.
        
        # Actually, the check above is only triggered if S has >= 4 bits.
        # In a near-valid coloring, S is small.
        
        current_violations = 0
        # We need a violation count, not just boolean.
        # Let's write a count function.
        
        def count_violations(colors, n):
            adj = [[0] * n for _ in range(2)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            total_viol = 0
            for c in [0, 1]:
                A = adj[c]
                for u1 in range(n):
                    for u2 in range(u1 + 1, n):
                        S = A[u1] & A[u2]
                        if S == 0:
                            continue
                        s_bits = []
                        temp = S
                        while temp:
                            lsb = temp & -temp
                            idx_v = lsb.bit_length() - 1
                            s_bits.append(idx_v)
                            temp ^= lsb
                        k = len(s_bits)
                        if k < 4:
                            continue
                        for a in range(k):
                            v1 = s_bits[a]
                            for b in range(a + 1, k):
                                v2 = s_bits[b]
                                C = A[v1] & A[v2]
                                common_in_S = C & S
                                # Count bits
                                # If >= 2, it's a violation.
                                # How many such (v1, v2) pairs?
                                if common_in_S & (common_in_S - 1):
                                    total_viol += 1
            return total_viol

        current_viol = count_violations(colors, n)
        best_colors = colors[:]
        best_viol = current_viol
        
        temp = 1.0
        cooling = 0.999
        iter_count = 0
        
        while time.time() < deadline - 5:
            iter_count += 1
            edge_idx = random.randint(0, n * (n - 1) // 2 - 1)
            old_val = colors[edge_idx]
            new_val = 1 - old_val
            colors[edge_idx] = new_val
            
            new_viol = count_violations(colors, n)
            
            delta = new_viol - current_viol
            if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_colors = colors[:]
                    if best_viol == 0:
                        return best_colors
            else:
                colors[edge_idx] = old_val
            
            temp *= cooling
            
        return best_colors

    # 1. Generate and verify Paley(29)
    n29 = 29
    colors29 = generate_paley(n29, 29)
    
    if is_valid_fast(colors29, n29):
        # We have a valid n=29 solution.
        # Try to extend to n=30
        n30 = 30
        # Initialize n=30 coloring
        # Take colors29 and add edges for vertex 29
        colors30 = colors29[:]
        # Edges (i, 29) for i in 0..28
        # Random initialization for these edges
        for i in range(n29):
            colors30.append(random.randint(0, 1))
            
        # Try local search on n=30
        # Limit time to 100s to ensure we return something
        colors30_optimized = local_search_30(colors30, n30)
        
        if is_valid_fast(colors30_optimized, n30):
            return {'n': 30, 'colors': colors30_optimized}
        else:
            # If we didn't find a valid n=30, return the valid n=29
            return {'n': 29, 'colors': colors29}
    else:
        # Fallback: random search for small n? Unlikely to happen for Paley(29)
        # Just return a small valid one
        return {'n': 17, 'colors': generate_paley(17, 17)}
