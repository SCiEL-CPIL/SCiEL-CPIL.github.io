import random
import time

def construct():
        n = 20
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        m = len(edges)
        
        # Precompute index for fast access? Not strictly needed for bitmask.
        
        def is_k33_free_add(u, v, M):
            # Check if adding (u, v) creates a K3,3
            # A = N(v) \ {u}
            # B = N(u) \ {v}
            # Check if there is K2,2 between A and B
            A = M[v] & ~(1 << u)
            B = M[u] & ~(1 << v)
            
            # Iterate over pairs in A
            # If |A| > |B|, iterate over B?
            # K2,2 check: exists a1, a2 in A such that |N(a1) & N(a2) & B| >= 2
            # Or exists b1, b2 in B such that |N(b1) & N(b2) & A| >= 2
            
            # Let's iterate over the smaller set to reduce pairs?
            # Actually, checking pairs in A is fine.
            # Optimization: If A is empty or B is empty, no K2,2.
            
            # Get list of vertices in A
            # We can iterate bits.
            a_list = []
            temp = A
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                a_list.append(idx)
                temp ^= lsb
                
            if len(a_list) < 2:
                return True
                
            # Check all pairs in a_list
            for i in range(len(a_list)):
                for j in range(i+1, len(a_list)):
                    a1 = a_list[i]
                    a2 = a_list[j]
                    common = M[a1] & M[a2] & B
                    # Count bits in common
                    if common & (common - 1): # At least 2 bits
                        return False
            return True

        def greedy_random():
            M = [0] * n
            # Random order of edges
            random.shuffle(edges)
            for u, v in edges:
                if is_k33_free_add(u, v, M):
                    M[u] |= (1 << v)
                    M[v] |= (1 << u)
            # Count edges
            edge_count = sum(bin(M[i]).count("1") for i in range(n)) // 2
            return M, edge_count

        best_M = [0] * n
        best_count = 0
        
        start_time = time.time()
        deadline = start_time + 110.0 # Safety margin
        
        while time.time() < deadline:
            M, count = greedy_random()
            if count > best_count:
                best_count = count
                best_M = M[:]
                
        # Convert best_M to list of edges
        result = []
        for i in range(n):
            temp = best_M[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if i < j:
                    result.append([i, j])
                temp ^= lsb
        return result
