"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import itertools
import sys

# EVOLVE-BLOCK-START

def _build_adj_edges(edges, n=18):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free(adj, n=18):
    # Check every 4-clique
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if not ((adj[a] >> c) & 1 and (adj[b] >> c) & 1):
                    continue
                # a,b,c form a triangle; check if any d > c completes K4
                common = adj[a] & adj[b] & adj[c]
                for d in range(c+1, n):
                    if (common >> d) & 1:
                        return False
    return True

def _max_triangle_free_size(adj, n=18):
    """Exact branch-and-bound: largest induced triangle-free subgraph."""
    all_mask = (1 << n) - 1
    
    # Precompute for each vertex, the set of vertices that form a triangle with it
    # i.e., for vertex v, tri_with[v] = { u : u and v share two common neighbors that are adjacent to each other }
    # Actually simpler: for a set S to be triangle-free, for every edge (u,v) in S, 
    # no w in S is adjacent to both u and v.
    
    # Branch and bound:
    # State: (candidate_mask, current_size)
    # We try to add vertices one by one, maintaining triangle-free property.
    
    # A vertex v can be added to current set S if:
    # For every pair (u, w) in S that are adjacent, v is not adjacent to both u and w.
    # Equivalently: for every neighbor u of v in S, the neighbors of u in S must not be adjacent to v.
    # i.e., for every u in S ∩ N(v), N(u) ∩ S ∩ N(v) must be empty.
    # i.e., N(v) ∩ N(u) ∩ S = empty for all u in S ∩ N(v).
    
    best = [0]
    
    def dfs(cand, size):
        if size + cand.bit_count() <= best[0]:
            return
        if cand == 0:
            if size > best[0]:
                best[0] = size
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with most constraints (highest degree within cand)
        v = -1
        max_deg = -1
        cm = cand
        while cm:
            lsb = cm & (-cm)
            vi = lsb.bit_length() - 1
            deg = (adj[vi] & cand).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = vi
            cm ^= lsb
        
        # Option 1: exclude v
        dfs(cand & ~(1 << v), size)
        
        # Option 2: include v (if it doesn't create a triangle)
        # Check: for every u in current selected set that is adjacent to v,
        # N(u) ∩ N(v) ∩ selected = empty
        # But we don't track 'selected' explicitly; we track cand (available) and size.
        # Actually, we need to track which vertices are in the current triangle-free set.
        # Let me restructure: pass the current set as a bitmask too.
        pass
    
    # Better approach: track current set explicitly
    best_size = [0]
    
    def dfs2(cand, current_set, size):
        if size + cand.bit_count() <= best_size[0]:
            return
        if cand == 0:
            if size > best_size[0]:
                best_size[0] = size
            return
        
        # Pick vertex with max degree in cand
        v = -1
        max_deg = -1
        cm = cand
        while cm:
            lsb = cm & (-cm)
            vi = lsb.bit_length() - 1
            deg = (adj[vi] & cand).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = vi
            cm ^= lsb
        
        # Exclude v
        dfs2(cand & ~(1 << v), current_set, size)
        
        # Include v: check no triangle formed
        # For every u in current_set adjacent to v, check N(u) & N(v) & current_set == 0
        can_add = True
        neighbors_in_set = adj[v] & current_set
        nm = neighbors_in_set
        while nm:
            lsb = nm & (-nm)
            u = lsb.bit_length() - 1
            if (adj[u] & adj[v] & current_set) & ~((1 << u) | (1 << v)):
                # There exists w in current_set adjacent to both u and v
                can_add = False
                break
            nm ^= lsb
        
        if can_add:
            dfs2(cand & ~(1 << v), current_set | (1 << v), size + 1)
    
    dfs2(all_mask, 0, 0)
    return best_size[0]

def _random_k4_free_graph(n=18, density=0.5, seed=None):
    """Generate a random K4-free graph by random edge additions with K4 check."""
    if seed is not None:
        random.seed(seed)
    
    edges = []
    adj = [0] * n
    possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(possible_edges)
    
    for u, v in possible_edges:
        if random.random() < density:
            # Check if adding edge (u,v) creates a K4
            # A K4 would be formed if there exist two vertices a,b such that:
            # a-b edge exists, a-u edge exists, a-v edge exists, b-u edge exists, b-v edge exists
            # i.e., a and b are both common neighbors of u and v, and a-b is an edge
            common = adj[u] & adj[v]
            # For each pair (a,b) in common, check if a-b is an edge
            cm = common
            has_k4 = False
            while cm and not has_k4:
                lsb_a = cm & (-cm)
                a = lsb_a.bit_length() - 1
                # Check if any b in common \ {a} is adjacent to a
                rest = common & ~(1 << a)
                if (adj[a] & rest) != 0:
                    has_k4 = True
                cm ^= lsb_a
            
            if not has_k4:
                edges.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    return edges, adj

def _try_improve(edges, adj, n=18, time_limit=30, rng=None):
    """Try to improve the graph by flipping edges to reduce max triangle-free size."""
    if rng is None:
        rng = random.Random()
    
    current_score = _max_triangle_free_size(adj, n)
    best_edges = edges[:]
    best_adj = adj[:]
    best_score = current_score
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Randomly try flipping an edge
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        u, v = rng.choice(possible_edges)
        
        # Check current state of edge (u,v)
        has_edge = (adj[u] >> v) & 1
        
        # Try flipping
        new_adj = adj[:]
        if has_edge:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            # Check K4-free before adding
            common = new_adj[u] & new_adj[v]
            cm = common
            has_k4 = False
            while cm and not has_k4:
                lsb_a = cm & (-cm)
                a = lsb_a.bit_length() - 1
                rest = common & ~(1 << a)
                if (new_adj[a] & rest) != 0:
                    has_k4 = True
                cm ^= lsb_a
            if has_k4:
                continue
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
        
        new_score = _max_triangle_free_size(new_adj, n)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = new_adj
            best_edges = [[a, b] for a in range(n) for b in range(a+1, n) if (new_adj[a] >> b) & 1]
            adj = new_adj
            current_score = new_score
        elif new_score == current_score and rng.random() < 0.1:
            # Accept equal with some probability (annealing-like)
            adj = new_adj
            current_score = new_score
            edges = [[a, b] for a in range(n) for b in range(a+1, n) if (new_adj[a] >> b) & 1]
    
    return best_edges, best_score

def construct():
    n = 18
    deadline = time.time() + 140  # Safety margin
    
    # Start with the known K6,6,6 construction (score 12)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    adj = _build_adj_edges(edges, n)
    best_score = _max_triangle_free_size(adj, n)
    best_edges = edges[:]
    
    # Try random K4-free graphs and local search
    rng = random.Random(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Generate random K4-free graph
        density = 0.4 + 0.2 * (trial % 5) / 5.0
        trial_edges, trial_adj = _random_k4_free_graph(n, density, seed=trial)
        trial_score = _max_triangle_free_size(trial_adj, n)
        
        if trial_score < best_score:
            best_score = trial_score
            best_edges = trial_edges[:]
            adj = trial_adj[:]
        
        # Local search improvement
        if time.time() > deadline - 5:
            break
        
        remaining_time = min(30, deadline - time.time() - 2)
        if remaining_time > 5:
            improved_edges, improved_score = _try_improve(
                trial_edges, trial_adj, n, time_limit=remaining_time, rng=rng
            )
            if improved_score < best_score:
                best_score = improved_score
                best_edges = improved_edges[:]
    
    # Final check
    if time.time() > deadline:
        return best_edges
    
    return best_edges

# EVOLVE-BLOCK-END
