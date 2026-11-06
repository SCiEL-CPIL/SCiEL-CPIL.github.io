import random
import time
import numpy as np

def _has_q3_bitset(adj_bits, n):
    """Check if graph with adjacency bitsets contains Q3 as subgraph.
    adj_bits: list of n integers, each with n bits set.
    Returns True if Q3 found, False otherwise."""
    # Q3 vertices 0-7, edges between vertices differing in exactly 1 bit
    # Q3 adjacency:
    # 0: 1,2,4
    # 1: 0,3,5
    # 2: 0,3,6
    # 3: 1,2,7
    # 4: 0,5,6
    # 5: 1,4,7
    # 6: 2,4,7
    # 7: 3,5,6
    
    q3_adj = [
        (1<<1)|(1<<2)|(1<<4),  # 0
        (1<<0)|(1<<3)|(1<<5),  # 1
        (1<<0)|(1<<3)|(1<<6),  # 2
        (1<<1)|(1<<2)|(1<<7),  # 3
        (1<<0)|(1<<5)|(1<<6),  # 4
        (1<<1)|(1<<4)|(1<<7),  # 5
        (1<<2)|(1<<4)|(1<<7),  # 6
        (1<<3)|(1<<5)|(1<<6),  # 7
    ]
    
    # Backtracking with bitsets
    # assign: dict q3_vertex -> graph_vertex
    # For speed, use arrays
    
    def backtrack(assign, remaining_mask):
        if remaining_mask == 0:
            return True
        # Pick most constrained unassigned Q3 vertex
        # Count how many neighbors are already assigned
        best_qv = -1
        best_count = -1
        for qv in range(8):
            if remaining_mask & (1 << qv):
                # Count assigned neighbors
                cnt = 0
                for qv2 in range(8):
                    if (assign[qv2] >= 0) and (q3_adj[qv] & (1 << qv2)):
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_qv = qv
        if best_qv < 0:
            return False
        
        rest_mask = remaining_mask & ~(1 << best_qv)
        
        # Try all unassigned graph vertices
        for gv in range(n):
            if assign[gv] >= 0:  # already used
                continue
            # Check edges to assigned Q3 vertices
            ok = True
            for qv2 in range(8):
                if assign[qv2] >= 0 and (q3_adj[best_qv] & (1 << qv2)):
                    if not (adj_bits[gv] & (1 << assign[qv2])):
                        ok = False
                        break
            if not ok:
                continue
            assign[best_qv] = gv
            if backtrack(assign, rest_mask):
                return True
            assign[best_qv] = -1
        return False
    
    # Try fixing Q3 vertex 0 to each graph vertex
    for start in range(n):
        assign = [-1] * n  # assign[q3_v] = graph_v
        # Actually, let's use assign as q3_v -> graph_v, and a separate used array
        assign = [-1] * 8
        used = [False] * n
        assign[0] = start
        used[start] = True
        
        # Modify backtrack to use these
        def backtrack2(remaining_mask):
            if remaining_mask == 0:
                return True
            best_qv = -1
            best_count = -1
            for qv in range(8):
                if remaining_mask & (1 << qv):
                    cnt = 0
                    for qv2 in range(8):
                        if assign[qv2] >= 0 and (q3_adj[qv] & (1 << qv2)):
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_qv = qv
            if best_qv < 0:
                return False
            
            rest_mask = remaining_mask & ~(1 << best_qv)
            
            for gv in range(n):
                if used[gv]:
                    continue
                ok = True
                for qv2 in range(8):
                    if assign[qv2] >= 0 and (q3_adj[best_qv] & (1 << qv2)):
                        if not (adj_bits[gv] & (1 << assign[qv2])):
                            ok = False
                            break
                if not ok:
                    continue
                assign[best_qv] = gv
                used[gv] = True
                if backtrack2(rest_mask):
                    return True
                assign[best_qv] = -1
                used[gv] = False
            return False
        
        if backtrack2((1<<8) - 1 - (1<<0)):
            return True
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    n = 16
    total_possible = n * (n - 1) // 2  # 120
    
    random.seed(123)
    
    best_edges = None
    best_count = 0
    
    def build_adj_bits(edge_set):
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    # Strategy: Start with dense random graphs, use local search to remove Q3, then add edges
    # Use simulated annealing style approach
    
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random initial graph with density 0.5-0.7
        density = random.uniform(0.5, 0.7)
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    edge_set.add((i, j))
        
        adj_bits = build_adj_bits(edge_set)
        
        # Phase 1: Remove edges to eliminate Q3
        iters = 0
        while time.time() < deadline - 15 and iters < 300:
            if not _has_q3_bitset(adj_bits, n):
                break
            # Find a Q3 and remove one of its edges
            # For simplicity, remove random edge
            if edge_set:
                e = random.choice(list(edge_set))
                edge_set.remove(e)
                adj_bits[e[0]] &= ~(1 << e[1])
                adj_bits[e[1]] &= ~(1 << e[0])
            iters += 1
        
        if time.time() > deadline - 10:
            break
        
        # Phase 2: Try to add edges greedily
        if not _has_q3_bitset(adj_bits, n):
            added = True
            while added and time.time() < deadline - 5:
                added = False
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
                random.shuffle(missing)
                for (u, v) in missing:
                    if time.time() > deadline - 5:
                        break
                    edge_set.add((u, v))
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    if not _has_q3_bitset(adj_bits, n):
                        added = True
                    else:
                        edge_set.remove((u, v))
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
        
        # Update best
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = list(edge_set)
    
    # Also try structured approaches
    # Complete bipartite K8,8 has 64 edges - check if it's Q3-free
    # K8,8: vertices 0-7 and 8-15, all edges between the two sets
    # Does K8,8 contain Q3? Q3 is bipartite (4+4), so it could be embedded
    # Q3 has 8 vertices, 4 in each partition. K8,8 has 8 in each partition.
    # So K8,8 likely contains Q3. Let's check.
    
    # Try K_{4,12} + K_4 (complete graph on 4 vertices)
    # This has 4*12 + 6 = 54 edges
    
    # Try Turán-like constructions
    
    # Fallback: known Q3-free graph with many edges
    # The complete graph K_5 has 10 edges, no Q3 (Q3 has 8 vertices)
    # Disjoint union of K_5's: 3*K_5 = 15 vertices, 30 edges, plus 1 isolated vertex
    # That's only 30 edges.
    
    # Better: use the complement approach
    # If we want to maximize edges in a Q3-free graph, we want to minimize edges in the complement
    # that are "forbidden" by Q3 constraints
    
    # Let's try a specific construction:
    # Take 16 vertices, partition into 4 groups of 4
    # Make each group a K4, and connect groups in a pattern that avoids Q3
    
    # Actually, let's just try more random restarts with different densities
    for trial2 in range(20):
        if time.time() > deadline - 10:
            break
        
        density = random.uniform(0.4, 0.65)
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    edge_set.add((i, j))
        
        adj_bits = build_adj_bits(edge_set)
        
        iters = 0
        while time.time() < deadline - 15 and iters < 200:
            if not _has_q3_bitset(adj_bits, n):
                break
            if edge_set:
                e = random.choice(list(edge_set))
                edge_set.remove(e)
                adj_bits[e[0]] &= ~(1 << e[1])
                adj_bits[e[1]] &= ~(1 << e[0])
            iters += 1
        
        if time.time() > deadline - 10:
            break
        
        if not _has_q3_bitset(adj_bits, n):
            added = True
            while added and time.time() < deadline - 5:
                added = False
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
                random.shuffle(missing)
                for (u, v) in missing:
                    if time.time() > deadline - 5:
                        break
                    edge_set.add((u, v))
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    if not _has_q3_bitset(adj_bits, n):
                        added = True
                    else:
                        edge_set.remove((u, v))
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
        
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = list(edge_set)
    
    # Final greedy pass on the best found
    if best_edges and time.time() < deadline - 5:
        edge_set = set(best_edges)
        adj_bits = build_adj_bits(edge_set)
        if not _has_q3_bitset(adj_bits, n):
            added = True
            while added and time.time() < deadline - 3:
                added = False
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
                random.shuffle(missing)
                for (u, v) in missing:
                    if time.time() > deadline - 3:
                        break
                    edge_set.add((u, v))
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    if not _has_q3_bitset(adj_bits, n):
                        added = True
                    else:
                        edge_set.remove((u, v))
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_edges = list(edge_set)
    
    # Fallback
    if best_edges is None or best_count <= 39:
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return [[u, v] for (u, v) in best_edges]
