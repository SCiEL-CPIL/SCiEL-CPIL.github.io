import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def get_adjacency(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_k4(adj, n):
        # Returns a K4 if found, else None
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common_ij = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                    k = common_ij
                    while k:
                        lsb = k & -k
                        k_val = lsb.bit_length() - 1
                        k &= k - 1
                        if k_val <= j: continue
                        common_ijk = common_ij & adj[k] & ~((1 << (k_val + 1)) - 1)
                        l = common_ijk
                        while l:
                            lsb_l = l & -l
                            l_val = lsb_l.bit_length() - 1
                            l &= l - 1
                            if l_val <= k_val: continue
                            if (adj[k] >> l_val) & 1:
                                return [i, j, k_val, l_val]
        return None

    def has_ind_6(adj, n):
        # Returns an independent set of 6 if found, else None
        full_mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = (full_mask ^ adj[i]) & ~(1 << i)

        for i in range(n):
            for j in range(i + 1, n):
                if (comp_adj[i] >> j) & 1:
                    common_ij = comp_adj[i] & comp_adj[j] & ~((1 << (j + 1)) - 1)
                    k = common_ij
                    while k:
                        lsb = k & -k
                        k_val = lsb.bit_length() - 1
                        k &= k - 1
                        if k_val <= j: continue
                        common_ijk = common_ij & comp_adj[k] & ~((1 << (k_val + 1)) - 1)
                        l = common_ijk
                        while l:
                            lsb_l = l & -l
                            l_val = lsb_l.bit_length() - 1
                            l &= l - 1
                            if l_val <= k_val: continue
                            common_ijkl = common_ijk & comp_adj[l] & ~((1 << (l_val + 1)) - 1)
                            m = common_ijkl
                            while m:
                                lsb_m = m & -m
                                m_val = lsb_m.bit_length() - 1
                                m &= m - 1
                                if m_val <= l_val: continue
                                common_ijklm = common_ijkl & comp_adj[m] & ~((1 << (m_val + 1)) - 1)
                                if common_ijklm:
                                    # Find the 6th vertex
                                    lsb_n = common_ijklm & -common_ijklm
                                    n_val = lsb_n.bit_length() - 1
                                    return [i, j, k_val, l_val, m_val, n_val]
        return None

    def is_valid(edges, n):
        adj = get_adjacency(edges, n)
        return has_k4(adj, n) is None and has_ind_6(adj, n) is None

    def turan_graph(n, r):
        # Construct Turan graph T(n, r)
        edges = []
        # Partition vertices into r sets
        sets = [[] for _ in range(r)]
        for i in range(n):
            sets[i % r].append(i)
        
        # Add edges between different sets
        for u in range(n):
            for v in range(u + 1, n):
                if u % r != v % r:
                    edges.append([u, v])
        return edges

    # Try to construct for n from 35 down to 17
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
            
        # Start with Turan graph T(n, 3)
        edges = turan_graph(n, 3)
        adj = get_adjacency(edges, n)
        
        # Local search to fix independent sets
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Check for independent set of 6
            ind_set = has_ind_6(adj, n)
            if ind_set is None:
                # Check for K4
                k4 = has_k4(adj, n)
                if k4 is None:
                    return {'n': n, 'edges': edges}
                else:
                    # Break K4 by removing a random edge
                    # Pick random edge in K4
                    idx = random.randint(0, 5)
                    # Map idx to pair in k4
                    # k4 is [v0, v1, v2, v3]
                    # Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
                    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
                    u, v = pairs[idx]
                    u, v = k4[u], k4[v]
                    # Remove edge
                    edges = [e for e in edges if not (e[0]==u and e[1]==v) and not (e[0]==v and e[1]==u)]
                    adj = get_adjacency(edges, n)
                    continue
            
            # Break independent set by adding a random edge within it
            u_idx = random.randint(0, 5)
            v_idx = random.randint(0, 5)
            while v_idx == u_idx:
                v_idx = random.randint(0, 5)
            u, v = ind_set[u_idx], ind_set[v_idx]
            
            # Check if edge exists
            if not ((adj[u] >> v) & 1):
                # Add edge
                edges.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                # Should not happen if it's an independent set
                pass

    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': 17, 'edges': edges}
