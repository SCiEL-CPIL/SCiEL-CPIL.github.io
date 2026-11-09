import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    full_mask = (1 << n) - 1
    
    # Q3 structure: 8 vertices, 12 edges
    q3_adj = [0] * 8
    q3_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_adj[i] |= (1 << j)
                q3_adj[j] |= (1 << i)
                q3_edges.append((i, j))
    
    def check_q3_free_fast(adj_mask, u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph. Returns True if still Q3-free."""
        for a, b in q3_edges:
            if _embed(adj_mask, u, v, a, b):
                return False
            if _embed(adj_mask, v, u, a, b):
                return False
        return True
    
    def _embed(adj_mask, gu, gv, qa, qb):
        """Embed Q3 with gu->qa, gv->qb. Only need Q3 edges present."""
        remaining = [x for x in range(8) if x != qa and x != qb]
        
        def constraint(q):
            c = 0
            if (q3_adj[q] >> qa) & 1:
                c += 1
            if (q3_adj[q] >> qb) & 1:
                c += 1
            return c
        remaining.sort(key=constraint, reverse=True)
        
        assignment = [None] * 8
        assignment[qa] = gu
        assignment[qb] = gv
        used = (1 << gu) | (1 << gv)
        
        def backtrack(idx):
            if idx == len(remaining):
                return True
            q = remaining[idx]
            neigh = q3_adj[q]
            
            # Candidates: unused vertices adjacent to all assigned neighbors of q
            candidates = full_mask & ~used
            for qn in range(8):
                if (neigh >> qn) & 1 and assignment[qn] is not None:
                    candidates &= adj_mask[assignment[qn]]
                    if not candidates:
                        return False
            
            while candidates:
                bit = candidates & (-candidates)
                g = bit.bit_length() - 1
                candidates ^= bit
                
                assignment[q] = g
                if backtrack(idx + 1):
                    return True
                assignment[q] = None
            
            return False
        
        return backtrack(0)
    
    def greedy_construct(seed_val):
        rng = random.Random(seed_val)
        
        adj_mask = [0] * n
        edges = []
        
        def add_edge(u, v):
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.append([u, v])
        
        # K_{3,13} + triangle on A + matching on B
        for a in range(3):
            for b in range(3, 16):
                add_edge(a, b)
        add_edge(0, 1)
        add_edge(1, 2)
        add_edge(0, 2)
        for i in range(3, 15, 2):
            add_edge(i, i + 1)
        
        missing = []
        for i in range(3, 16):
            for j in range(i + 1, 16):
                if not (adj_mask[i] & (1 << j)):
                    missing.append((i, j))
        
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if check_q3_free_fast(adj_mask, u, v):
                edges.append([u, v])
            else:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        return edges
    
    best_edges = None
    best_count = 0
    
    # Deterministic first pass
    e0 = greedy_construct(42)
    best_edges = list(e0)
    best_count = len(e0)
    
    # Random restarts
    for attempt in range(25):
        if time.time() > deadline - 3:
            break
        er = greedy_construct(1000 + attempt * 7919)
        if len(er) > best_count:
            best_edges = list(er)
            best_count = len(er)
    
    return best_edges
