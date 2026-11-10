import time
from itertools import combinations
import collections
import heapq

def construct():
    deadline = time.time() + 80
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all subsets and their residues, sorted by size
    subsets_all = []
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            residue = sum(inv[x] for x in combo) % 101
            subsets_all.append((combo, residue))
            
    # Precompute N_k
    N_k = [0] * 11
    current_idx = 0
    for s in range(1, 11):
        for _ in combinations(range(1, 11), s):
            current_idx += 1
        N_k[s] = current_idx

    def hopcroft_karp(adj, n_left, n_right):
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = collections.deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == float('inf'):
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found
        
        def dfs(u):
            for v in adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        if matching == n_left:
            return pair_u
        return None

    def min_cost_max_flow(n_nodes, source, sink, adj_list, n_flow):
        # adj_list[u] = list of (v, cap, cost, edge_id)
        # We need reverse edges for residual graph
        # edge_id is index in adj_list[u]
        # rev_id is index in adj_list[v]
        
        # Initialize potential
        potential = [0] * n_nodes
        flow = 0
        cost = 0
        
        # Dijkstra with potentials
        while flow < n_flow:
            dist = [float('inf')] * n_nodes
            prev_v = [-1] * n_nodes
            prev_e = [-1] * n_nodes
            dist[source] = 0
            pq = [(0, source)]
            
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for i, (v, cap, w, rev) in enumerate(adj_list[u]):
                    if cap > 0:
                        nd = d + w + potential[u] - potential[v]
                        if nd < dist[v]:
                            dist[v] = nd
                            prev_v[v] = u
                            prev_e[v] = i
                            heapq.heappush(pq, (nd, v))
            
            if dist[sink] == float('inf'):
                break # No more flow possible
            
            # Update potentials
            for v in range(n_nodes):
                if dist[v] < float('inf'):
                    potential[v] += dist[v]
            
            # Augment path
            v = sink
            while v != source:
                u = prev_v[v]
                i = prev_e[v]
                adj_list[u][i][1] -= 1
                adj_list[v][adj_list[u][i][3]][1] += 1
                v = u
            
            flow += 1
            cost += potential[sink] - potential[source] # This is the actual cost of the path?
            # Actually, the cost of the path in original graph is:
            # sum(w(e)) for e in path.
            # dist[sink] is sum(w(e) + pot[u] - pot[v]).
            # sum(w(e)) = dist[sink] - sum(pot[u]-pot[v]) = dist[sink] - (pot[source] - pot[sink]).
            # So path_cost = dist[sink] + potential[source] - potential[sink].
            # Since potential[source] is usually 0 (if we start with 0 and don't change it? No, we update all).
            # Let's just sum the costs of the edges in the path to be safe.
            path_cost = 0
            v = sink
            while v != source:
                u = prev_v[v]
                i = prev_e[v]
                path_cost += adj_list[u][i][2]
                v = u
            cost += path_cost
            
        return flow, cost, prev_v, prev_e # prev_v/prev_e are from last iteration, need to trace back? 
        # Better to return the actual matching or trace it outside.
        # For simplicity, let's just return the total cost and the final flow.
        # We need to reconstruct the matching.
        # We can store the flow in the edges.

    # The above MCMF implementation is a bit messy for reconstruction.
    # Let's use a simpler approach for reconstruction:
    # After MCMF, we check which edges from Residue->Subset have flow 1.
    
    k_min = None
    n_right_min = 0
    subsets_used = []
    
    for k in range(1, 11):
        if time.time() > deadline:
            break
        
        n_right = N_k[k]
        
        # Build adjacency for HK
        adj = [[] for _ in range(101)]
        for idx in range(n_right):
            combo, residue = subsets_all[idx]
            adj[residue].append(idx)
            
        if any(len(adj[r]) == 0 for r in range(101)):
            continue
        
        matching = hopcroft_karp(adj, 101, n_right)
        
        if matching is not None:
            k_min = k
            n_right_min = n_right
            subsets_used = subsets_all[:n_right]
            break
    
    if k_min is None:
        # Fallback
        residues_by_size = [dict() for _ in range(11)]
        for s in range(1, 11):
            for combo in combinations(range(1, 11), s):
                residue = sum(inv[x] for x in combo) % 101
                if residue not in residues_by_size[s]:
                    residues_by_size[s][residue] = list(combo)
        
        result = []
        for r in range(101):
            for s in range(1, 11):
                if r in residues_by_size[s]:
                    result.append(residues_by_size[s][r])
                    break
            else:
                result.append(list(range(1, 11)))
        return result

    # Now run Min-Cost Max-Flow to minimize sum of sizes
    # Nodes:
    # 0: Source
    # 1..101: Residues
    # 102..101+n_right: Subsets
    # 102+n_right: Sink
    
    n_nodes = 102 + n_right
    source = 0
    sink = 101 + n_right
    
    adj_mcmf = [[] for _ in range(n_nodes)]
    
    def add_edge(u, v, cap, cost):
        adj_mcmf[u].append([v, cap, cost, len(adj_mcmf[v])])
        adj_mcmf[v].append([u, 0, -cost, len(adj_mcmf[u]) - 1])

    # Source to Residues
    for r in range(101):
        add_edge(source, 1 + r, 1, 0)
        
    # Residues to Subsets
    # We only use subsets of size <= k_min
    # subsets_used contains them
    for idx, (combo, residue) in enumerate(subsets_used):
        subset_node = 102 + idx
        r_node = 1 + residue
        size = len(combo)
        add_edge(r_node, subset_node, 1, size)
        
    # Subsets to Sink
    for idx in range(n_right):
        subset_node = 102 + idx
        add_edge(subset_node, sink, 1, 0)

    # Run MCMF
    # We need a robust MCMF that returns the flow paths
    
    potential = [0] * n_nodes
    flow = 0
    total_cost = 0
    # Store the actual edges used for reconstruction
    # We can just look at the final capacities
    
    while flow < 101:
        dist = [float('inf')] * n_nodes
        prev_v = [-1] * n_nodes
        prev_e = [-1] * n_nodes
        dist[source] = 0
        pq = [(0, source)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for i, edge in enumerate(adj_mcmf[u]):
                v, cap, cost, rev = edge
                if cap > 0:
                    nd = d + cost + potential[u] - potential[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        prev_v[v] = u
                        prev_e[v] = i
                        heapq.heappush(pq, (nd, v))
        
        if dist[sink] == float('inf'):
            break
            
        for v in range(n_nodes):
            if dist[v] < float('inf'):
                potential[v] += dist[v]
        
        # Augment
        v = sink
        while v != source:
            u = prev_v[v]
            i = prev_e[v]
            adj_mcmf[u][i][1] -= 1
            adj_mcmf[v][adj_mcmf[u][i][3]][1] += 1
            v = u
        
        flow += 1
        # Calculate cost of this path
        path_cost = 0
        v = sink
        while v != source:
            u = prev_v[v]
            i = prev_e[v]
            path_cost += adj_mcmf[u][i][2]
            v = u
        total_cost += path_cost

    # Reconstruct matching
    result = [None] * 101
    for r in range(101):
        r_node = 1 + r
        for edge in adj_mcmf[r_node]:
            v, cap, cost, rev = edge
            # If this is a forward edge from Residue to Subset
            # and it has flow (cap == 0, original cap was 1)
            # Note: reverse edge has cap 1 if flow was sent.
            # We look for edges where v is a subset node (102 <= v < 102+n_right)
            # and the reverse edge has cap 1 (meaning flow 1 was sent)
            if 102 <= v < 102 + n_right:
                # Check if flow was sent
                # The reverse edge is in adj_mcmf[v] at index rev
                # If flow was sent, adj_mcmf[v][rev][1] should be 1
                if adj_mcmf[v][rev][1] == 1:
                    idx = v - 102
                    combo, _ = subsets_used[idx]
                    result[r] = list(combo)
                    break
    
    # Verify
    for r in range(101):
        if result[r] is None:
            # Should not happen if flow == 101
            continue
        row = result[r]
        assert len(row) > 0
        assert len(set(row)) == len(row)
        assert all(1 <= x <= 10 for x in row)
        computed = sum(inv[x] for x in row) % 101
        assert computed == r

    return result
