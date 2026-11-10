"""
Dense Q3-free graph on 16 vertices.

Strategy:
  1. Enumerate all distinct edge sets of Q3 (12 edges on 8 labeled vertices) up to
     isomorphism, then generate all concrete embeddings into vertices 0..15.
  2. Use a fast "does adding edge e create a Q3?" check via bitmasks.
  3. Initialize with a strong structured graph (e.g., based on known dense Q3-free
     constructions or random restarts with greedy insertion).
  4. Improve via simulated annealing / local search with edge swaps.

Key idea: Instead of checking all Q3 embeddings from scratch each time, we precompute
for each potential edge (u,v) which Q3s would be completed by adding it. Then the
check becomes: is there any Q3 containing (u,v) where all other 11 edges are present?

However, the total number of Q3 embeddings is large (~10M), so we use a hybrid:
  - For the "is adding (u,v) safe?" check, use a fast recursive search that only
    explores the specific structure of Q3 given (u,v) as an edge.
  - This is the approach in the seed program, which is already fairly efficient.

To improve over the seed:
  - Better initialization: use multiple structured initial graphs.
  - More aggressive local search: try many edge swaps, 2-swaps, and 3-swaps.
  - Simulated annealing to escape local optima.
  - Parallel-ish restarts with different random seeds.

Let me think about the structure of Q3 more carefully for a faster check.

Q3 = 3-dimensional hypercube. 8 vertices, 12 edges.
If we fix an edge (u,v), in Q3 the vertices can be labeled as 3-bit strings where
u and v differ in exactly one bit. WLOG, say u=000, v=001.

The other 6 vertices are:
  010, 011, 100, 101, 110, 111

Edges (12 total):
  000-001 (u-v, fixed)
  000-010, 000-100
  001-011, 001-101
  010-011, 010-110
  100-101, 100-110
  011-111, 101-111
  011-101? No, 011 and 101 differ in 2 bits.
  Wait, let me list all edges properly:
  For each vertex, connect to vertices differing in 1 bit:
  000: 001, 010, 100
  001: 000, 011, 101
  010: 000, 011, 110
  011: 001, 010, 111
  100: 000, 101, 110
  101: 001, 100, 111
  110: 010, 100, 111
  111: 011, 101, 110

So the 12 edges are:
  (000,001), (000,010), (000,100),
  (001,011), (001,101),
  (010,011), (010,110),
  (011,111),
  (100,101), (100,110),
  (101,111),
  (110,111)

Given u and v (adjacent in Q3), we need to find 6 more vertices a,b,c,d,e,f such that
the edge pattern matches. The structure is rigid: 
  - u has 2 other neighbors (call them a, b)
  - v has 2 other neighbors (call them c, d)
  - a is adjacent to one of c,d (say c) and to e
  - b is adjacent to the other (say d) and to e
  - Wait, this is getting complicated. Let me use the bitmask approach.

Actually, the seed's creates_q3_fast is already a good approach. Let me optimize it and
add better search heuristics.

Key improvements:
1. Precompute for each vertex its adjacency as a bitmask.
2. Use the creates_q3_fast check (optimized) to test edge additions.
3. Use a priority queue for edge additions: prioritize edges that are "close" to completing
   a Q3 (i.e., would create a Q3 if added, so avoid them) vs edges that are far.
4. Actually, for maximization, we want to add edges that DON'T create Q3s. So we just
   need a fast check.
5. Use simulated annealing: with some probability, accept a move that reduces edge count
   to escape local optima.

Let me also think about known results: ex(16, Q3) is not known exactly, but lower bounds
are around 50-60 edges. The seed scores 1.5128, which if the seed baseline is around 33
edges (1.5128 * 33 ≈ 50), suggests the seed finds ~50 edges.

Let me try a different approach: use integer programming or a smarter greedy.

Actually, let me try this: 
- Start with empty graph.
- Add edges in a carefully chosen order.
- Use the fast Q3 check.
- Then do local search with swaps.

For the Q3 check, I'll optimize the seed's approach by:
  - Caching common sub-results
  - Pruning early
  - Using the fact that Q3 is bipartite with parts of size 4

Let me code this up with a focus on speed and search quality.
"""
import time
import random
import math

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute bit masks
    def bit(i):
        return 1 << i
    
    def creates_q3(u, v, adj):
        """
        Fast check: does adding edge (u,v) to the graph create a Q3 subgraph?
        adj[i] is a bitmask of neighbors of vertex i (excluding self).
        We assume (u,v) is NOT yet in adj.
        """
        nu = adj[u] & ~bit(v)
        nv = adj[v] & ~bit(u)
        
        if nu.bit_count() < 2 or nv.bit_count() < 2:
            return False
        
        # In Q3 with edge (u,v), u has exactly 2 other neighbors in Q3,
        # v has exactly 2 other neighbors in Q3.
        # Let a2, a4 be u's other Q3-neighbors, a3, a5 be v's other Q3-neighbors.
        # a6 = common neighbor of a2 and a4 (not u,v,a2,a3,a4,a5)
        # a7 = common neighbor of a3 and a5 (not u,v,a2,a3,a4,a5,a6)
        # a6 and a7 must be adjacent.
        
        a2_bits = nu
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = nu & ~bit(a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                # a3 must be in nv and adjacent to a2
                a3_cand = nv & adj[a2_idx] & ~bit(a2_idx) & ~bit(a4_idx)
                if not a3_cand:
                    continue
                
                # a5 must be in nv and adjacent to a4
                a5_cand = nv & adj[a4_idx] & ~bit(a2_idx) & ~bit(a4_idx)
                if not a5_cand:
                    continue
                
                a3_bits = a3_cand
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand2 = a5_cand & ~bit(a3_idx)
                    if not a5_cand2:
                        continue
                    
                    a5_bits = a5_cand2
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used = bit(u) | bit(v) | bit(a2_idx) | bit(a3_idx) | bit(a4_idx) | bit(a5_idx)
                        
                        a6_cand = (adj[a2_idx] & adj[a4_idx]) & ~used
                        if not a6_cand:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used
                        
                        a6_bits = a6_cand
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~bit(a6_idx)
                            if not a7_cand:
                                continue
                            
                            # Check if a6 and a7 are adjacent
                            if adj[a6_idx] & a7_cand:
                                return True
        
        return False
    
    def count_edges(adj):
        s = 0
        for u in range(n):
            s += adj[u].bit_count()
        return s // 2
    
    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                b = mask & (-mask)
                v = b.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= b
        return result
    
    def greedy_add(adj, edge_order):
        """Try to add edges from edge_order in sequence, skipping those that create Q3."""
        added = 0
        for u, v in edge_order:
            if adj[u] & bit(v):
                continue
            if not creates_q3(u, v, adj):
                adj[u] |= bit(v)
                adj[v] |= bit(u)
                added += 1
        return added
    
    def get_missing_edges(adj):
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & bit(v)):
                    missing.append((u, v))
        return missing
    
    def get_existing_edges(adj):
        existing = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & bit(v):
                    existing.append((u, v))
        return existing
    
    # Simulated annealing local search
    def sa_local_search(adj, deadline, max_iters=5000):
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        best_adj = current_adj[:]
        best_count = current_count
        
        T = 5.0  # Initial temperature
        T_min = 0.01
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Generate a random move
            r = random.random()
            if r < 0.5:
                # Try to add a missing edge
                missing = get_missing_edges(current_adj)
                if not missing:
                    continue
                u, v = random.choice(missing)
                trial_adj = current_adj[:]
                trial_adj[u] |= bit(v)
                trial_adj[v] |= bit(u)
                # Check if this creates a Q3
                if creates_q3(u, v, current_adj):
                    continue
                delta = 1
            elif r < 0.8:
                # Try to remove an edge
                existing = get_existing_edges(current_adj)
                if not existing:
                    continue
                u, v = random.choice(existing)
                trial_adj = current_adj[:]
                trial_adj[u] &= ~bit(v)
                trial_adj[v] &= ~bit(u)
                delta = -1
            else:
                # Try a swap: remove one edge, add another
                existing = get_existing_edges(current_adj)
                missing = get_missing_edges(current_adj)
                if not existing or not missing:
                    continue
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                trial_adj = current_adj[:]
                trial_adj[u1] &= ~bit(v1)
                trial_adj[v1] &= ~bit(u1)
                # Check if adding (u2,v2) creates a Q3
                if creates_q3(u2, v2, trial_adj):
                    continue
                trial_adj[u2] |= bit(v2)
                trial_adj[v2] |= bit(u2)
                delta = 0  # Net change is 0, but structure changes
        
        if delta == 0:
            new_count = count_edges(trial_adj)
            delta = new_count - current_count
        
        # Acceptance criterion
        if delta > 0:
            current_adj = trial_adj
            current_count += delta
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
        else:
            # Accept with probability exp(-delta/T) if delta < 0
            if random.random() < math.exp(-delta / T):
                current_adj = trial_adj
                current_count += delta
        
        T = max(T_min, T * cooling)
        
        return best_adj, best_count
    
    # Main search loop
    random.seed(42)
    best_adj = [0] * n
    best_count = 0
    
    # Phase 1: Multiple random restarts with greedy initialization
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        adj = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        greedy_add(adj, order)
        cnt = count_edges(adj)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Phase 2: Local search
        if time.time() < deadline - 25:
            adj2, cnt2 = sa_local_search(adj, deadline - 25, max_iters=3000)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2[:]
    
    # Phase 3: Final greedy polish
    if time.time() < deadline - 3:
        missing = get_missing_edges(best_adj)
        random.shuffle(missing)
        greedy_add(best_adj, missing)
        best_count = count_edges(best_adj)
    
    return edges_from_adj(best_adj)
