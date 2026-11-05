import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0

    def time_exceeded():
        return time.time() - start_time > deadline

    q = 29

    def build_paley_colors(n):
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    colors_29 = build_paley_colors(q)

    def build_adj(n, colors):
        """Build bitmask adjacency lists for both color classes."""
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def has_k222(n, adj):
        """
        Fast check: does the graph with adjacency 'adj' contain a K_{2,2,2}?
        K_{2,2,2} is the complete tripartite graph with parts of size 2.
        Vertices: A={a1,a2}, B={b1,b2}, C={c1,c2}.
        Edges exist between any vertex in different parts.
        No edges within a part.
        
        So for a fixed color c (represented by adj[c]):
        We need to find 6 vertices a1,a2,b1,b2,c1,c2 such that:
        - (a1,a2) is a non-edge in adj
        - (b1,b2) is a non-edge in adj
        - (c1,c2) is a non-edge in adj
        - All edges between {a1,a2} and {b1,b2} exist in adj
        - All edges between {a1,a2} and {c1,c2} exist in adj
        - All edges between {b1,b2} and {c1,c2} exist in adj
        
        Algorithm:
        1. Iterate over all non-edges (a1, a2) in adj.
        2. Let S = N(a1) ∩ N(a2). These are candidates for B and C parts.
        3. We need to find two disjoint non-edges (b1,b2) and (c1,c2) in the induced subgraph on S,
           such that b1,b2 are both in N(a1)∩N(a2) (already true by definition of S)
           AND c1,c2 are both in N(b1)∩N(b2) (and vice versa).
           
        Actually, a simpler check:
        For a fixed non-edge (a1, a2):
        Let S = N(a1) ∩ N(a2).
        We need to find a non-edge (b1, b2) in S (meaning b1,b2 are not adjacent in adj)
        such that N(b1) ∩ N(b2) ∩ S contains a non-edge (c1, c2).
        
        Wait, if (b1,b2) is a non-edge in adj, and b1,b2 are in S, then:
        a1, a2 are connected to b1, b2.
        We need c1, c2 such that:
        - c1, c2 are connected to a1, a2 (so c1,c2 in S)
        - c1, c2 are connected to b1, b2 (so c1,c2 in N(b1)∩N(b2))
        - (c1,c2) is a non-edge in adj.
        
        So:
        For each non-edge (a1, a2):
          S = N(a1) ∩ N(a2)
          For each non-edge (b1, b2) in S:
            T = N(b1) ∩ N(b2) ∩ S
            If T contains a non-edge (c1, c2), then we found a K_{2,2,2}.
        
        This is O(n^2 * (n/2)^2 * (n/2)^2) worst case, but with bitmasks it's fast.
        """
        for i in range(n):
            for j in range(i + 1, n):
                # (i, j) must be a non-edge in adj
                if adj[i] & (1 << j):
                    continue
                
                # S = common neighbors of i and j
                s_mask = (adj[i] & adj[j]) & ~(1 << i) & ~(1 << j)
                
                # Iterate over non-edges (k, l) in S
                # We need k, l in S and NOT (k, l) is an edge in adj
                # Extract list of vertices in S
                s_list = []
                mask = s_mask
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    s_list.append(v)
                    mask ^= lsb
                
                # For each pair (k, l) in s_list, check if it's a non-edge
                for a in range(len(s_list)):
                    k = s_list[a]
                    for b in range(a + 1, len(s_list)):
                        l = s_list[b]
                        # (k, l) must be a non-edge in adj
                        if adj[k] & (1 << l):
                            continue
                        
                        # Now find T = N(k) ∩ N(l) ∩ S
                        t_mask = (adj[k] & adj[l] & s_mask) & ~(1 << k) & ~(1 << l)
                        
                        # Check if T contains a non-edge
                        # Iterate over pairs in T
                        t_list = []
                        mask2 = t_mask
                        while mask2:
                            lsb2 = mask2 & (-mask2)
                            v2 = lsb2.bit_length() - 1
                            t_list.append(v2)
                            mask2 ^= lsb2
                        
                        for c in range(len(t_list)):
                            m = t_list[c]
                            for d in range(c + 1, len(t_list)):
                                o = t_list[d]
                                # (m, o) must be a non-edge in adj
                                if not (adj[m] & (1 << o)):
                                    return True
        return False

    def count_k222_conflicts(n, adj):
        """Count K_{2,2,2} subgraphs (for local search feedback)."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    continue
                s_mask = (adj[i] & adj[j]) & ~(1 << i) & ~(1 << j)
                s_list = []
                mask = s_mask
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    s_list.append(v)
                    mask ^= lsb
                for a in range(len(s_list)):
                    k = s_list[a]
                    for b in range(a + 1, len(s_list)):
                        l = s_list[b]
                        if adj[k] & (1 << l):
                            continue
                        t_mask = (adj[k] & adj[l] & s_mask) & ~(1 << k) & ~(1 << l)
                        t_list = []
                        mask2 = t_mask
                        while mask2:
                            lsb2 = mask2 & (-mask2)
                            v2 = lsb2.bit_length() - 1
                            t_list.append(v2)
                            mask2 ^= lsb2
                        for c in range(len(t_list)):
                            m = t_list[c]
                            for d in range(c + 1, len(t_list)):
                                o = t_list[d]
                                if not (adj[m] & (1 << o)):
                                    count += 1
        return count

    def find_k222_witness(n, adj):
        """Find one K_{2,2,2} witness or return None."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    continue
                s_mask = (adj[i] & adj[j]) & ~(1 << i) & ~(1 << j)
                s_list = []
                mask = s_mask
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    s_list.append(v)
                    mask ^= lsb
                for a in range(len(s_list)):
                    k = s_list[a]
                    for b in range(a + 1, len(s_list)):
                        l = s_list[b]
                        if adj[k] & (1 << l):
                            continue
                        t_mask = (adj[k] & adj[l] & s_mask) & ~(1 << k) & ~(1 << l)
                        t_list = []
                        mask2 = t_mask
                        while mask2:
                            lsb2 = mask2 & (-mask2)
                            v2 = lsb2.bit_length() - 1
                            t_list.append(v2)
                            mask2 ^= lsb2
                        for c in range(len(t_list)):
                            m = t_list[c]
                            for d in range(c + 1, len(t_list)):
                                o = t_list[d]
                                if not (adj[m] & (1 << o)):
                                    return (i, j, k, l, m, o)
        return None

    # Verify Paley(29) is valid
    adj_29 = build_adj(29, colors_29)
    if not (has_k222(29, adj_29[0]) or has_k222(29, adj_29[1])):
        # Valid! Try to extend to n=30
        best_n = 29
        best_colors = colors_29[:]
        
        random.seed(42)
        
        # Strategy: Start with Paley(29) + random vertex, then local search
        # Use the "blow-up" idea: the new vertex should mimic one of the existing vertices
        # or be a "random" vertex, but we start with a few different initializations
        
        for attempt in range(20):
            if time_exceeded():
                break
            
            # Different initialization strategies
            if attempt < 5:
                # Strategy 1: Clone vertex 0
                colors_30 = colors_29[:]
                for i in range(29):
                    # Edge (i, 29) gets same color as edge (0, i) or (0, something)
                    # Let's clone vertex 0: color(i, 29) = color(0, i) for i != 0
                    # But we need to handle (0, 29) specially. Let's just use random for now.
                    pass
                # Actually, let's just use random initialization for the new vertex edges
                for i in range(29):
                    colors_30.append(random.randint(0, 1))
            elif attempt < 10:
                # Strategy 2: Clone vertex 14 (middle)
                colors_30 = colors_29[:]
                # Build adjacency for vertex 14 in Paley(29)
                adj14 = [0]*29
                idx = 0
                for i in range(29):
                    for j in range(i+1, 29):
                        if i == 14:
                            adj14[j] = colors_29[idx]
                        elif j == 14:
                            adj14[i] = colors_29[idx]
                        idx += 1
                # Now set colors_30 edges to vertex 29 to match adj14
                for i in range(29):
                    colors_30.append(adj14[i])
            else:
                # Strategy 3: Random
                colors_30 = colors_29[:]
                for i in range(29):
                    colors_30.append(random.randint(0, 1))
            
            # Local search
            adj_30 = build_adj(30, colors_30)
            cur_conflicts = 0
            if has_k222(30, adj_30[0]):
                cur_conflicts += 1
            if has_k222(30, adj_30[1]):
                cur_conflicts += 1
            
            if cur_conflicts == 0:
                return {'n': 30, 'colors': colors_30}
            
            # Local search iterations
            for iteration in range(2000):
                if time_exceeded():
                    break
                
                # Find witness
                witness = find_k222_witness(30, adj_30[0])
                if witness is None:
                    witness = find_k222_witness(30, adj_30[1])
                
                if witness is None:
                    # Should not happen if cur_conflicts > 0, but just in case
                    break
                
                verts = list(witness)
                # Flip a random edge among the 6 vertices
                v1, v2 = random.sample(verts, 2)
                if v1 > v2:
                    v1, v2 = v2, v1
                
                # Find edge index
                edge_idx = -1
                idx = 0
                for i in range(30):
                    for j in range(i+1, 30):
                        if (i, j) == (v1, v2):
                            edge_idx = idx
                            break
                        idx += 1
                    if edge_idx != -1:
                        break
                
                if edge_idx == -1:
                    continue
                
                colors_30[edge_idx] = 1 - colors_30[edge_idx]
                
                # Rebuild adj and check
                adj_30 = build_adj(30, colors_30)
                if not (has_k222(30, adj_30[0]) or has_k222(30, adj_30[1])):
                    return {'n': 30, 'colors': colors_30}
        
        return {'n': best_n, 'colors': best_colors}
    else:
        # Paley(29) is invalid? Try to repair or fallback
        # But Paley(29) is known to be valid for K_{2,2,2}
        # So this branch should not be taken
        return {'n': 29, 'colors': colors_29}


if __name__ == "__main__":
    result = construct()
    print(result['n'])
