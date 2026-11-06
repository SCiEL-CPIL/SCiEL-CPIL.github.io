import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def adj_to_edges(adj):
        n = len(adj)
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def find_k4(adj):
        n = len(adj)
        for i in range(n):
            ni = adj[i]
            neigh = []
            mask = ni
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                neigh.append(j)
                mask -= lsb
            for idx in range(len(neigh)):
                j = neigh[idx]
                for kdx in range(idx+1, len(neigh)):
                    k = neigh[kdx]
                    if (adj[j] >> k) & 1:
                        common = ni & adj[j] & adj[k]
                        if common:
                            lsb = common & -common
                            l = lsb.bit_length() - 1
                            return (i, j, k, l)
        return None

    def find_ind6(adj):
        n = len(adj)
        all_mask = (1 << n) - 1
        for i in range(n):
            # Non-neighbors of i in G
            non_neigh = all_mask ^ adj[i] ^ (1 << i)
            # Check if non_neigh contains a K5 in G_bar
            # This is equivalent to checking if the subgraph of non_neigh in G_bar has a K5
            # G_bar adjacency for a vertex v is: non-neighbors of v in G
            # Let's list the vertices in non_neigh
            verts = []
            mask = non_neigh
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                verts.append(v)
                mask -= lsb
            
            # Check for K5 in G_bar induced by verts
            # For each v in verts, check if G_bar[verts \ {v}] contains K4
            # G_bar edges between a,b in verts exist iff NOT (adj[a] & (1<<b))
            
            def has_k5_in_bar(verts):
                m = len(verts)
                if m < 5: return False
                # Map verts to 0..m-1 for local indexing? Or just use original indices.
                # Let's use original indices.
                # For each v in verts:
                for v in verts:
                    # Neighbors of v in G_bar within verts
                    neigh_v = []
                    for u in verts:
                        if u == v: continue
                        if not (adj[v] >> u) & 1:
                            neigh_v.append(u)
                    # Check if neigh_v contains K4 in G_bar
                    if has_k4_in_bar(neigh_v):
                        return True
                return False

            def has_k4_in_bar(verts):
                m = len(verts)
                if m < 4: return False
                for v in verts:
                    neigh_v = []
                    for u in verts:
                        if u == v: continue
                        if not (adj[v] >> u) & 1:
                            neigh_v.append(u)
                    if has_k3_in_bar(neigh_v):
                        return True
                return False

            def has_k3_in_bar(verts):
                m = len(verts)
                if m < 3: return False
                for v in verts:
                    neigh_v = []
                    for u in verts:
                        if u == v: continue
                        if not (adj[v] >> u) & 1:
                            neigh_v.append(u)
                    # Check if neigh_v has an edge in G_bar
                    # i.e., exist a,b in neigh_v such that NOT (adj[a] & (1<<b))
                    for a_idx in range(len(neigh_v)):
                        a = neigh_v[a_idx]
                        for b_idx in range(a_idx+1, len(neigh_v)):
                            b = neigh_v[b_idx]
                            if not (adj[a] >> b) & 1:
                                return True
                return False

            if has_k5_in_bar(verts):
                # Found Ind6 containing i
                # Return the 6 vertices
                # We need to reconstruct the K5 in bar to get the vertices
                # For now, just return i and some 5 from verts?
                # We need the exact 6 for repair.
                # Let's modify has_k5 to return the vertices.
                pass
        return None

    # The recursive finding is complex to return the exact set.
    # Let's just use a random repair strategy:
    # 1. Check validity.
    # 2. If invalid, pick a random edge and flip it.
    # 3. If validity improves (or stays same), keep it.
    # This is slow.
    
    # Better:
    # 1. Find a K4. If found, flip a random edge in it.
    # 2. Else find an Ind6. If found, add a random edge within the 6-set.
    
    # We need find_ind6 to return the set.
    
    def find_ind6_set(adj):
        n = len(adj)
        all_mask = (1 << n) - 1
        for i in range(n):
            non_neigh = all_mask ^ adj[i] ^ (1 << i)
            verts = []
            mask = non_neigh
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                verts.append(v)
                mask -= lsb
            
            # Find K5 in G_bar induced by verts
            def find_k5_in_bar(verts):
                m = len(verts)
                if m < 5: return None
                for v in verts:
                    neigh_v = []
                    for u in verts:
                        if u == v: continue
                        if not (adj[v] >> u) & 1:
                            neigh_v.append(u)
                    res = find_k4_in_bar(neigh_v)
                    if res:
                        return [v] + res
                return None

            def find_k4_in_bar(verts):
                m = len(verts)
                if m < 4: return None
                for v in verts:
                    neigh_v = []
                    for u in verts:
                        if u == v: continue
                        if not (adj[v] >> u) & 1:
                            neigh_v.append(u)
                    res = find_k3_in_bar(neigh_v)
                    if res:
                        return [v] + res
                return None

            def find_k3_in_bar(verts):
                m = len(verts)
                if m < 3: return None
                for v in verts:
                    neigh_v = []
                    for u in verts:
                        if u == v: continue
                        if not (adj[v] >> u) & 1:
                            neigh_v.append(u)
                    for a_idx in range(len(neigh_v)):
                        a = neigh_v[a_idx]
                        for b_idx in range(a_idx+1, len(neigh_v)):
                            b = neigh_v[b_idx]
                            if not (adj[a] >> b) & 1:
                                return [v, a, b]
                return None

            k5 = find_k5_in_bar(verts)
            if k5:
                return [i] + k5
        return None

    def is_valid(adj):
        if find_k4(adj): return False
        if find_ind6_set(adj): return False
        return True

    def repair(adj, n):
        # Returns True if repaired
        for _ in range(10000):
            if time.time() > deadline: return False
            
            k4 = find_k4(adj)
            if k4:
                i, j, k, l = k4
                # Flip a random edge in the K4
                edges_in_k4 = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                u, v = random.choice(edges_in_k4)
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                continue
            
            ind6 = find_ind6_set(adj)
            if ind6:
                # Add a random edge in the Ind6 set (in G)
                # i.e., remove an edge in G_bar
                u, v = random.choice([(ind6[a], ind6[b]) for a in range(6) for b in range(a+1, 6)])
                # Only if it's currently NOT an edge in G
                if not (adj[u] >> v) & 1:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    # Should not happen if ind6 is correct
                    pass
                continue
            
            # No violations found
            return True
        return False

    best_n = 1
    best_adj = [0]
    
    # Try n=35
    n = 35
    for _ in range(10): # 10 restarts
        if time.time() > deadline: break
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if repair(adj, n):
            if is_valid(adj):
                best_n = n
                best_adj = adj
                break # Found valid 35
    
    # Try n=36
    if time.time() < deadline - 10:
        n = 36
        for _ in range(10):
            if time.time() > deadline: break
            # Start from random or from best 35 + random vertex
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if repair(adj, n):
                if is_valid(adj):
                    best_n = n
                    best_adj = adj
                    break

    edges = adj_to_edges(best_adj)
    return {'n': best_n, 'edges': edges}
