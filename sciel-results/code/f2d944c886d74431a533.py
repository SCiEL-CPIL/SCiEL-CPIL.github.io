import numpy as np
import time
import random
import math

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Construct base graph using GF(5) vector space approach
    # Map 31 vertices to 5-dimensional vectors over GF(5)
    # Use a structured assignment that creates a graph with low C4 count
    
    adj = np.zeros((n, n), dtype=np.int8)
    edge_set = set()
    
    # Create vertex labels as 5-digit base-5 representations
    # This gives us 31 distinct 5D vectors over GF(5)
    vertex_vectors = []
    for i in range(n):
        vec = [0] * 5
        val = i
        for d in range(5):
            vec[d] = val % 5
            val //= 5
        vertex_vectors.append(vec)
    
    # Define a symmetric bilinear form B over GF(5)
    # We want to choose B such that the graph where u~v iff B(u,v)=0 has low C4
    # Try multiple random symmetric matrices and pick the best
    
    best_c4_count = float('inf')
    best_adj = None
    best_edge_set = None
    
    # Generate candidate symmetric matrices over GF(5)
    # A symmetric matrix has 5*6/2 = 15 free parameters
    num_candidates = 50
    random.seed(42)
    
    for trial in range(num_candidates):
        if time.time() > deadline - 60:
            break
        
        # Generate random symmetric matrix over GF(5)
        B = np.zeros((5, 5), dtype=np.int8)
        for i in range(5):
            for j in range(i, 5):
                B[i, j] = random.randint(0, 4)
                B[j, i] = B[i, j]
        
        # Build graph: edge (u,v) iff B(vertex_u, vertex_v) == 0 mod 5
        adj_candidate = np.zeros((n, n), dtype=np.int8)
        edge_set_candidate = set()
        
        for i in range(n):
            for j in range(i + 1, n):
                dot = sum(vertex_vectors[i][k] * vertex_vectors[j][k] for k in range(5)) % 5
                if dot == 0:
                    adj_candidate[i, j] = 1
                    adj_candidate[j, i] = 1
                    edge_set_candidate.add((i, j))
        
        current_edges = len(edge_set_candidate)
        
        # If we have exactly 91 edges, this is a valid candidate
        # If we have fewer, we need to add edges; if more, we need to remove
        # For now, just evaluate C4 count for the natural graph
        
        if current_edges == m:
            # Count C4s
            A2 = adj_candidate @ adj_candidate
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            vals = A2[mask]
            c4_count = int(np.sum(vals * (vals - 1)) // 2)
            
            if c4_count < best_c4_count:
                best_c4_count = c4_count
                best_adj = adj_candidate.copy()
                best_edge_set = set(edge_set_candidate)
    
    # If no candidate had exactly 91 edges, use a different strategy
    if best_adj is None:
        # Use a greedy approach: start with empty graph, add edges that minimize C4 increase
        adj = np.zeros((n, n), dtype=np.int8)
        edge_set = set()
        
        all_non_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_non_edges)
        
        def count_c4s_for_nonedge(u, v, adj):
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            count = 0
            if len(nu) < len(nv):
                for w in nu:
                    count += np.sum(adj[w, nv])
            else:
                for x in nv:
                    count += np.sum(adj[x, nu])
            return count
        
        for _ in range(m):
            if time.time() > deadline - 30:
                break
            
            # Find the non-edge that adds the fewest C4s
            best_edge = None
            min_c4s = float('inf')
            
            for u, v in all_non_edges:
                if (u, v) in edge_set:
                    continue
                c4s = count_c4s_for_nonedge(u, v, adj)
                if c4s < min_c4s:
                    min_c4s = c4s
                    best_edge = (u, v)
            
            if best_edge:
                u, v = best_edge
                adj[u, v] = 1
                adj[v, u] = 1
                edge_set.add(best_edge)
                all_non_edges.remove(best_edge)
        
        best_adj = adj
        best_edge_set = edge_set
    
    adj = best_adj
    edge_set = best_edge_set
    current_edges = len(edge_set)
    
    # If we don't have exactly 91 edges, adjust
    if current_edges < m:
        all_non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        random.shuffle(all_non_edges)
        
        def count_c4s_for_nonedge(u, v, adj):
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            count = 0
            if len(nu) < len(nv):
                for w in nu:
                    count += np.sum(adj[w, nv])
            else:
                for x in nv:
                    count += np.sum(adj[x, nu])
            return count
        
        edges_to_add = m - current_edges
        for _ in range(edges_to_add):
            if time.time() > deadline - 20:
                break
            
            best_edge = None
            min_c4s = float('inf')
            
            for u, v in all_non_edges:
                if (u, v) in edge_set:
                    continue
                c4s = count_c4s_for_nonedge(u, v, adj)
                if c4s < min_c4s:
                    min_c4s = c4s
                    best_edge = (u, v)
            
            if best_edge:
                u, v = best_edge
                adj[u, v] = 1
                adj[v, u] = 1
                edge_set.add(best_edge)
                all_non_edges.remove(best_edge)
    
    elif current_edges > m:
        # Remove edges that contribute most to C4s
        all_edges = list(edge_set)
        random.shuffle(all_edges)
        
        def count_c4s_for_edge(u, v, adj):
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            nu_masked = [w for w in nu if w != v]
            nv_masked = [x for x in nv if x != u]
            
            count = 0
            if len(nu_masked) < len(nv_masked):
                for w in nu_masked:
                    count += np.sum(adj[w, nv_masked])
            else:
                for x in nv_masked:
                    count += np.sum(adj[x, nu_masked])
            return count
        
        edges_to_remove = current_edges - m
        for _ in range(edges_to_remove):
            if time.time() > deadline - 20:
                break
            
            best_edge = None
            max_c4s = -1
            
            for u, v in all_edges:
                if (u, v) not in edge_set:
                    continue
                c4s = count_c4s_for_edge(u, v, adj)
                if c4s > max_c4s:
                    max_c4s = c4s
                    best_edge = (u, v)
            
            if best_edge:
                u, v = best_edge
                adj[u, v] = 0
                adj[v, u] = 0
                edge_set.remove(best_edge)
                all_edges.remove(best_edge)
    
    current_edges = len(edge_set)
    
    # Helper to count C4s containing a specific non-edge (u, v)
    # This is the number of edges (w, x) such that w in N(u), x in N(v)
    def count_c4s_for_nonedge(u, v, adj):
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        count = 0
        # Iterate over smaller set
        if len(nu) < len(nv):
            for w in nu:
                # Sum adj[w, nv]
                count += np.sum(adj[w, nv])
        else:
            for x in nv:
                count += np.sum(adj[x, nu])
        return count

    # Helper to count C4s containing a specific edge (u, v)
    # Same formula, but u and v are neighbors.
    # N(u) contains v, N(v) contains u.
    # The C4 is u-w-x-v-u.
    # w must be in N(u) \ {v}
    # x must be in N(v) \ {u}
    # And (w, x) must be an edge.
    def count_c4s_for_edge(u, v, adj):
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        # Exclude v from nu and u from nv
        # Since adj is int8, we can just mask or filter
        # np.where returns indices. 
        # v is in nu because adj[u,v]=1.
        # u is in nv because adj[v,u]=1.
        
        nu_masked = [w for w in nu if w != v]
        nv_masked = [x for x in nv if x != u]
        
        count = 0
        if len(nu_masked) < len(nv_masked):
            for w in nu_masked:
                count += np.sum(adj[w, nv_masked])
        else:
            for x in nv_masked:
                count += np.sum(adj[x, nu_masked])
        return count

    # 3. Local Search: Simulated Annealing with 2-opt swaps
    # We want to minimize total C4s.
    # Move: Remove (u,v), Add (x,y).
    
    def count_c4s_for_nonedge(u, v, adj):
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        count = 0
        if len(nu) < len(nv):
            for w in nu:
                count += np.sum(adj[w, nv])
        else:
            for x in nv:
                count += np.sum(adj[x, nu])
        return count
    
    def count_c4s_for_edge(u, v, adj):
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        nu_masked = [w for w in nu if w != v]
        nv_masked = [x for x in nv if x != u]
        
        count = 0
        if len(nu_masked) < len(nv_masked):
            for w in nu_masked:
                count += np.sum(adj[w, nv_masked])
        else:
            for x in nv_masked:
                count += np.sum(adj[x, nu_masked])
        return count
    
    def get_delta(adj, u, v, x, y):
        # Remove (u,v), Add (x,y)
        # 1. Calculate C4s lost by removing (u,v) in current graph
        lost = count_c4s_for_edge(u, v, adj)
        
        # 2. Temporarily remove (u,v)
        adj[u, v] = 0
        adj[v, u] = 0
        
        # 3. Calculate C4s gained by adding (x,y) in the modified graph
        gained = count_c4s_for_nonedge(x, y, adj)
        
        # 4. Restore (u,v) for evaluation (we don't apply yet)
        adj[u, v] = 1
        adj[v, u] = 1
        
        return gained - lost
    
    # Compute initial C4 count
    A2 = adj @ adj
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A2[mask]
    current_c4 = int(np.sum(vals * (vals - 1)) // 2)
    
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.995
    
    current_edges_list = list(edge_set)
    
    steps = 0
    while T > T_min and time.time() < deadline - 5:
        steps += 1
        
        # Pick random edge to remove
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u, v = e_rm
        
        # Pick random non-edge to add
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) not in edge_set:
                break
        
        delta = get_delta(adj, u, v, x, y)
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            # Apply removal
            adj[u, v] = 0
            adj[v, u] = 0
            edge_set.remove(e_rm)
            current_edges_list[idx_rm] = (x, y)
            
            # Apply addition
            adj[x, y] = 1
            adj[y, x] = 1
            edge_set.add((x, y))
            
            current_c4 += delta
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
                
        T *= cooling_rate
    
    # Final polish: try all single-edge swaps from best solution
    if time.time() < deadline - 10:
        adj = best_adj.copy()
        edge_set = set(best_edge_set)
        
        current_c4 = best_c4
        
        # Try removing each edge and adding each non-edge
        for u, v in list(edge_set):
            if time.time() > deadline - 10:
                break
            
            lost = count_c4s_for_edge(u, v, adj)
            
            adj[u, v] = 0
            adj[v, u] = 0
            
            best_gain = -1
            best_add = None
            
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) in edge_set:
                        continue
                    gained = count_c4s_for_nonedge(i, j, adj)
                    net = gained - lost
                    if net < best_gain:
                        best_gain = net
                        best_add = (i, j)
            
            adj[u, v] = 1
            adj[v, u] = 1
            
            if best_add is not None and best_gain < 0:
                i, j = best_add
                adj[u, v] = 0
                adj[v, u] = 0
                edge_set.remove((u, v))
                
                adj[i, j] = 1
                adj[j, i] = 1
                edge_set.add((i, j))
                
                current_c4 += best_gain
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
                    best_edge_set = set(edge_set)
    
    # Return best found
    if best_edge_set and len(best_edge_set) == m:
        return [[u, v] for u, v in best_edge_set]
    elif best_edge_set:
        # If we don't have exactly m edges, try to fix
        edges = list(best_edge_set)
        if len(edges) < m:
            # Add random non-edges
            all_non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in best_edge_set]
            random.shuffle(all_non_edges)
            for e in all_non_edges[:m - len(edges)]:
                best_edge_set.add(e)
        elif len(edges) > m:
            # Remove random edges
            random.shuffle(edges)
            for e in edges[:len(edges) - m]:
                best_edge_set.remove(e)
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
