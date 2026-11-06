import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_edges(n):
        return [(i, j) for i in range(n) for j in range(i+1, n)]

    def precompute_c6s(n):
        edges = get_edges(n)
        edge_to_idx = {e: idx for idx, e in enumerate(edges)}
        
        # Generate all 6-cycles
        # A 6-cycle is a set of 6 vertices v0..v5 such that edges (v0,v1)..(v5,v0) exist.
        # In K_n, all edges exist.
        # We need to list all combinations of 6 vertices and all permutations that form a cycle.
        # Number of 6-cycles in K_n is n! / (12 * (n-6)!)
        # For n=17, this is ~640k.
        
        c6s = []
        c6s_per_edge = [[] for _ in range(len(edges))]
        
        # This loop might be slow in Python.
        # We can use itertools.combinations and permutations.
        # But 640k is small enough.
        
        import itertools
        for verts in itertools.combinations(range(n), 6):
            # There are 60 distinct cycles for 6 vertices (6! / 12 = 30? No, (6-1)! / 2 = 60)
            # We just need to check if the edges form a cycle.
            # Actually, in K_n, any permutation of 6 vertices forms a cycle if we connect them in order.
            # We just need to list the edge sets.
            # To avoid duplicates, we can fix v0 as the smallest vertex.
            # And fix v1 < v5?
            
            # Simpler: just generate all edge sets.
            # For each permutation of 6 vertices:
            for perm in itertools.permutations(verts):
                # Normalize: rotate so smallest is first
                min_idx = perm.index(min(perm))
                norm_perm = perm[min_idx:] + perm[:min_idx]
                # Reverse check
                rev_norm = (norm_perm[0],) + norm_perm[1:][::-1]
                if norm_perm < rev_norm:
                    continue
                    
                # Get edge indices
                e_idxs = []
                valid = True
                for k in range(6):
                    u = norm_perm[k]
                    v = norm_perm[(k+1)%6]
                    if u > v: u, v = v, u
                    e_idxs.append(edge_to_idx[(u, v)])
                
                # Add to list
                c6_idx = len(c6s)
                c6s.append(e_idxs)
                for e in e_idxs:
                    c6s_per_edge[e].append(c6_idx)
                    
        return c6s, c6s_per_edge

    # Precompute for n=17
    c6s_17, c6s_per_edge_17 = precompute_c6s(17)
    
    def solve(n, c6s, c6s_per_edge):
        edges = get_edges(n)
        m = len(edges)
        
        colors = np.random.randint(0, 4, size=m)
        
        # A_c matrices
        A = [np.zeros((n, n), dtype=int) for _ in range(4)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            A[c][i, j] = 1
            A[c][j, i] = 1
            
        # C6 mono status
        c6_mono = np.zeros(len(c6s), dtype=bool)
        c6_count = 0
        
        def update_c6_mono(e_idx, new_color):
            nonlocal c6_count
            for c6_idx in c6s_per_edge[e_idx]:
                e_idxs = c6s[c6_idx]
                # Check if all edges have new_color
                # We can just check the other 5 edges
                other_idxs = [e for e in e_idxs if e != e_idx]
                if all(colors[e] == new_color for e in other_idxs):
                    if not c6_mono[c6_idx]:
                        c6_mono[c6_idx] = True
                        c6_count += 1
                else:
                    if c6_mono[c6_idx]:
                        c6_mono[c6_idx] = False
                        c6_count -= 1
                        
        # Initial C6 count
        for c6_idx, e_idxs in enumerate(c6s):
            c = colors[e_idxs[0]]
            if all(colors[e] == c for e in e_idxs[1:]):
                c6_mono[c6_idx] = True
                c6_count += 1
                
        def count_c4():
            total = 0
            for c in range(4):
                A2 = A[c] @ A[c]
                triu_indices = np.triu_indices(n, 1)
                vals = A2[triu_indices]
                total += np.sum(vals * (vals - 1) // 2)
            return total
            
        best_c4 = count_c4()
        best_c6 = c6_count
        best_colors = colors.copy()
        
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline:
            idx = random.randint(0, m - 1)
            i, j = edges[idx]
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            while new_c == old_c:
                new_c = random.randint(0, 3)
                
            # Update A
            A[old_c][i, j] = 0
            A[old_c][j, i] = 0
            A[new_c][i, j] = 1
            A[new_c][j, i] = 1
            
            # Update C6
            # We need to revert C6 if we reject
            old_c6_count = c6_count
            old_c6_mono = c6_mono.copy()
            
            update_c6_mono(idx, new_c)
            
            new_c4 = count_c4()
            new_c6 = c6_count
            
            cost_old = best_c4 + 10 * best_c6
            cost_new = new_c4 + 10 * new_c6
            
            delta = cost_new - cost_old
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                best_c4 = new_c4
                best_c6 = new_c6
                colors[idx] = new_c
                best_colors = colors.copy()
            else:
                # Revert
                A[old_c][i, j] = 1
                A[old_c][j, i] = 1
                A[new_c][i, j] = 0
                A[new_c][j, i] = 0
                c6_mono[:] = old_c6_mono
                c6_count = old_c6_count
                
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return best_colors.tolist()

    # Try n=17
    if time.time() < deadline:
        colors17 = solve(17, c6s_17, c6s_per_edge_17)
        return {'n': 17, 'colors': colors17}
        
    # Fallback
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
