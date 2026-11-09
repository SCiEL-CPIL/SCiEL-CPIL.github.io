"""
Hexagon-free graph on 40 vertices – structural K5-block design with greedy inter-block search.

Design:
  - Start with 8 disjoint K5 blocks (80 edges).
  - Greedily add inter-block edges (each K5 vertex is 'clique-like', so adding an
    inter-block edge creates C6 only if there is a path of length 4 between the two
    endpoints in the current graph).
  - Refine via 1-swap local search: try removing one inter-block edge and adding
    a different one, accepting if it keeps the graph C6-free and the total edge
    count does not drop (or occasionally drops to escape local optima).
  - Use bitset adjacency (Python ints) for fast C6 path-of-length-4 checks.

Key insight: in a graph where each vertex has high degree, the number of length-4
paths between two vertices is roughly O(deg^4). For 40 vertices with ~105 edges,
avg degree ~5.25, so deg^4 ~ 760. The bitset approach lets us check this in
O(n) word operations rather than O(deg^4) loop iterations.
"""

import time
import random
import numpy as np

N = 40
G = 8       # number of K5 blocks
S = 5       # size of each block

# Precompute block structure
def _block_of(v):
    return v // S

def _block_members(g):
    base = g * S
    return list(range(base, base + S))

def _build_bitset_adj(edges):
    """Build bitset adjacency: adj[v] is an int with bit u set iff edge (u,v)."""
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _creates_c6_bitset(adj, u, v):
    """
    Check if adding edge (u,v) to graph with bitset adj creates a C6.
    A C6 through (u,v): u-v-w-x-y-z-u, i.e., path of length 4 from v to u
    avoiding the direct edge (which isn't there yet).
    
    Equivalent: is there a walk u - a - b - c - d - v where all intermediate
    vertices are distinct and not u or v? Actually we need a simple path of
    length 4 from v to u in the current graph (without the edge u-v).
    
    More precisely: a C6 is u-v-w-x-y-z-u. So we need a path of length 4
    from v to u in the current graph where the internal vertices are all
    distinct and not u or v.
    
    Using bitsets: 
    - Let A = adj[u] (neighbors of u)
    - For each neighbor a of u (a != v), compute neighbors of a that are not u
    - For each such neighbor b, compute neighbors of b not in {u, a}
    - For each such neighbor c, compute neighbors of c not in {u, a, b}
    - Check if v is among those neighbors
    
    This is O(deg(u) * deg(a) * deg(b) * deg(c)) in worst case, but with bitsets
    we can do it more efficiently by tracking the set of possible 'c' vertices.
    """
    # We need: exists w, x, y such that v-w-x-y-u is a path of length 4
    # i.e., w in N(v), x in N(w)\{v}, y in N(x)\{v,w}, and u in N(y)\{v,w,x}
    
    # Bitset approach:
    # N_v = adj[v] (neighbors of v)
    # For each w in N_v (w != u):
    #   For each x in N(w) \ {v, w}:
    #     For each y in N(x) \ {v, w, x}:
    #       if u in N(y) \ {v, w, x, y}: return True
    #
    # Optimize: for each w, let X_w = adj[w] & ~(1<<v) & ~(1<<w)  (candidates for x)
    # For each x in X_w, let Y_x = adj[x] & ~(1<<v) & ~(1<<w) & ~(1<<x)
    # We need: exists y in Y_x such that adj[y] has bit u set and y not in {u,v,w,x}
    
    # Even better: for fixed w, define:
    #   Cands_x = adj[w] & ~(1<<v) & ~(1<<w)
    #   For each x in Cands_x:
    #     Cands_y = adj[x] & ~(1<<v) & ~(1<<w) & ~(1<<x) & ~(1<<u)  # y != u
    #     If Cands_y is non-empty and any y in Cands_y has u in adj[y]:
    #       return True
    #
    # Let's precompute for each w:
    #   union_y = OR over x in Cands_x of (adj[x] & ~(1<<v) & ~(1<<w) & ~(1<<x) & ~(1<<u))
    #   Then check if union_y & adj[u] is non-empty... wait, we need u in adj[y],
    #   so we need y in adj[u]. So check union_y & adj[u] != 0.
    
    u_bit = 1 << u
    v_bit = 1 << v
    
    N_v = adj[v] & ~u_bit  # neighbors of v, excluding u (u-v edge not present yet)
    
    for w in range(N):
        if not (N_v >> w) & 1:
            continue
        # w is a neighbor of v, w != u
        w_bit = 1 << w
        # x candidates: neighbors of w, not v, not w
        Cands_x = adj[w] & ~v_bit & ~w_bit
        # For each x, y candidates: neighbors of x, not v, not w, not x, not u
        # We want: exists y such that y in N(x) \ {v,w,x,u} AND u in N(y)
        # i.e., y in (N(x) & ~v_bit & ~w_bit & ~x_bit & ~u_bit) AND y in adj[u]
        
        # Precompute adj_u = adj[u]
        # For this w, compute union of (adj[x] & mask_x) for x in Cands_x
        # where mask_x = ~v_bit & ~w_bit & ~x_bit & ~u_bit
        
        # Instead of iterating x, let's compute:
        # For each x in Cands_x:
        #   valid_y = adj[x] & ~v_bit & ~w_bit & ~(1<<x) & ~u_bit
        #   if valid_y & adj[u]: return True
        
        adj_u = adj[u]
        for x in range(N):
            if not (Cands_x >> x) & 1:
                continue
            x_bit = 1 << x
            valid_y = adj[x] & ~v_bit & ~w_bit & ~x_bit & ~u_bit
            if valid_y & adj_u:
                return True
    return False

def _has_c6_bitset(adj):
    """Full C6 check using bitset adjacency."""
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj[u] >> v) & 1):
                continue
            # Check if removing edge (u,v) and then checking if (u,v) would create C6
            # is equivalent to checking if there's a path of length 4 from v to u
            # in the graph without edge (u,v).
            # Temporarily remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            result = _creates_c6_bitset(adj, u, v)
            # Restore
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if result:
                return True
    return False

def _seed_k5_blocks():
    """Build 8 disjoint K5 blocks."""
    edges = []
    for g in range(G):
        members = _block_members(g)
        for i in range(S):
            for j in range(i + 1, S):
                edges.append([members[i], members[j]])
    return edges

def _all_inter_block_pairs():
    """All possible inter-block edges."""
    pairs = []
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            m1 = _block_members(g1)
            m2 = _block_members(g2)
            for a in m1:
                for b in m2:
                    pairs.append((a, b))
    return pairs

def _greedy_inter_edges(seed_edges, rng, deadline):
    """
    Greedily add inter-block edges to seed graph.
    Returns list of inter-block edges added.
    """
    adj = _build_bitset_adj(seed_edges)
    added = []
    all_pairs = _all_inter_block_pairs()
    rng.shuffle(all_pairs)
    
    for (u, v) in all_pairs:
        if time.time() > deadline - 5:
            break
        if _creates_c6_bitset(adj, u, v):
            continue
        # Add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        added.append([u, v])
    
    return added

def _local_search_1swap(edges, rng, deadline):
    """
    1-swap local search: try removing one edge and adding another.
    Accept if C6-free and count >= current count (or with small probability if count drops by 1).
    """
    adj = _build_bitset_adj(edges)
    current_count = len(edges)
    best_count = current_count
    best_edges = list(edges)
    
    all_pairs = _all_inter_block_pairs()
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        # Try a random swap: remove edge i, add random pair j
        if not edges:
            break
        
        i = rng.randint(0, len(edges) - 1)
        ru, rv = edges[i]
        
        # Remove edge (ru, rv)
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)
        edge_set.discard((ru, rv))
        edge_set.discard((rv, ru))
        edges.pop(i)
        current_count -= 1
        
        # Try to add a random inter-block edge
        added = False
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            u, v = rng.choice(all_pairs)
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            if _creates_c6_bitset(adj, u, v):
                continue
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            edge_set.add((v, u))
            edges.append([u, v])
            current_count += 1
            added = True
            break
        
        if not added:
            # Restore removed edge
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
            edge_set.add((ru, rv))
            edge_set.add((rv, ru))
            edges.append([ru, rv])
            current_count += 1
            no_improve += 1
            continue
        
        # Check if we improved
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edges)
            no_improve = 0
        elif current_count == best_count:
            no_improve += 1
        else:
            # Count dropped; revert
            # Remove the added edge
            # Find it in edges
            for j in range(len(edges) - 1, -1, -1):
                if edges[j][0] == u and edges[j][1] == v:
                    edges.pop(j)
                    break
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edge_set.discard((u, v))
            edge_set.discard((v, u))
            # Restore removed edge
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
            edge_set.add((ru, rv))
            edge_set.add((rv, ru))
            edges.append([ru, rv])
            current_count += 1
            no_improve += 1
    
    # Return best found
    return best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    rng = random.Random(42)
    
    # Seed: 8 disjoint K5 blocks (80 edges)
    seed_edges = _seed_k5_blocks()
    
    # Greedy inter-block edge addition
    inter_edges = _greedy_inter_edges(seed_edges, rng, deadline)
    all_edges = seed_edges + inter_edges
    
    best_edges = list(all_edges)
    best_count = len(all_edges)
    
    # Local search refinement
    refined = _local_search_1swap(all_edges, rng, deadline)
    if len(refined) > best_count:
        best_edges = refined
        best_count = len(best_edges)
    
    # Final verification
    adj = _build_bitset_adj(best_edges)
    if _has_c6_bitset(adj):
        # Fallback to seed
        return seed_edges
    
    return best_edges
