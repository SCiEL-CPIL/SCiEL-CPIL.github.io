"""
K2,3-free graph construction on 20 vertices.
Uses branch-and-bound maximum independent set on the conflict graph
where nodes represent potential edges and conflicts encode K2,3 constraints.
"""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)
    pair_index = {(i, j): k for k, (i, j) in enumerate(all_pairs)}

    # Build conflict graph: two edges conflict if adding both creates a K2,3
    # K2,3: vertices a,b share 3 common neighbors.
    # Edge (a,b) conflicts with edge (c,d) if:
    #   - They share a vertex (say a), and there exist 2 other vertices
    #     that are neighbors of both a and the other vertex in the pair
    # Actually, let me think more carefully.
    #
    # A K2,3 subgraph consists of 2 vertices {u,v} and 3 vertices {a,b,c}
    # such that u and v are both connected to a, b, c.
    # 
    # Two edges e1=(x,y) and e2=(z,w) conflict if there exists a K2,3
    # that uses both edges. The K2,3 has 5 vertices: the two "hub" vertices
    # and the three "leaf" vertices.
    #
    # For edges (u,v) and (u,w) sharing vertex u:
    #   They conflict if there exists a vertex x such that x is adjacent to
    #   both v and w (in the final graph). But we're building the conflict
    #   graph statically... 
    #
    # Actually, the conflict is: if we include edge (u,v) and edge (u,w),
    # then for any vertex x that is adjacent to both v and w, we'd have
    # K2,3 with hubs {v,w} and leaves {u, x, ...}. Wait, that's not right either.
    #
    # Let me reconsider. K2,3 has bipartition {a,b} and {c,d,e} where
    # a-c, a-d, a-e, b-c, b-d, b-e are all edges.
    #
    # So two edges conflict if they are part of some K2,3. The K2,3 uses
    # 6 edges. Two edges are in conflict if they can both appear in some
    # K2,3 subgraph of the complete graph K_20.
    #
    # In K_20, any two edges that share a vertex: e1=(u,v), e2=(u,w).
    # Can they be in a K2,3? The K2,3 would need hubs {v,w} and leaves
    # including u. We need 2 more leaves x,y such that v-x, v-y, w-x, w-y
    # are edges. In K_20, these are all present, so YES. Any two edges
    # sharing a vertex conflict!
    #
    # What about two disjoint edges e1=(u,v), e2=(w,x)?
    # K2,3 with hubs {u,w} and leaves {v,x,...}: need one more leaf y with
    # u-y, w-y edges. In K_20, yes. So they conflict too.
    #
    # Wait, that means ALL pairs of edges conflict in K_20? That can't be right.
    # Let me re-examine.
    #
    # K2,3: 2 hubs, 3 leaves, all 6 cross-edges present.
    # For edges (u,v) and (u,w) to be in a K2,3:
    #   Case 1: u is a hub. Then v and w must be leaves. Need a second hub h
    #     such that h-v, h-w are edges, and a third leaf l such that u-l, h-l
    #     are edges. In K_20, pick any h != u,v,w and l != u,v,w,h. Works.
    #   So yes, they conflict.
    #
    # For disjoint edges (u,v) and (w,x):
    #   Case 1: u and w are hubs, v and x are leaves. Need third leaf l with
    #     u-l, w-l edges. Pick l != u,v,w,x. Works in K_20.
    #   So they conflict too.
    #
    # This means in the complete graph, every pair of edges is in some K2,3.
    # That's expected - the problem is to find a large subset with NO K2,3.
    #
    # So the conflict graph on 190 nodes has an edge between every pair?
    # No wait - I need to be more careful. Two edges conflict if their
    # simultaneous presence FORCES a K2,3, i.e., if there's a K2,3 that
    # contains both edges as part of its 6 edges.
    #
    # In K_20, for any two edges, there IS a K2,3 containing both. So the
    # conflict graph is K_190, and the max independent set has size 1?
    # That's clearly wrong since we can have many edges.
    #
    # I think I'm confusing myself. The issue is: we're not choosing from K_20.
    # We're choosing a subset of edges. Two edges e1, e2 "conflict" if there
    # exists a set of other edges such that {e1, e2} ∪ S contains a K2,3.
    # But that's always true in K_20.
    #
    # The correct formulation: we want to select edges so that NO K2,3 exists.
    # A K2,3 is determined by its 5 vertices: 2 hubs + 3 leaves. The 6 edges
    # between them must ALL be present. So a K2,3 is "formed" when all 6 of
    # its edges are selected.
    #
    # So the constraint is: for every choice of 2 hubs and 3 leaves, NOT all
    # 6 cross-edges are selected. This is a 6-clique constraint, not a pairwise
    # conflict.
    #
    # This means the simple "conflict graph" approach doesn't directly work
    # because the constraint is on 6 edges simultaneously, not 2.
    #
    # However, we can still use branch-and-bound with the constraint that
    # for any pair of vertices (u,v), at most 2 of their common neighbors
    # can be connected to BOTH u and v... no wait, that's the same thing.
    #
    # Let me use a different approach: branch-and-bound where we build the
    # graph edge by edge, and prune when adding an edge would make some pair
    # (u,v) have 3 common neighbors.
    
    # Actually, let me use a smarter formulation. The key constraint is:
    # For every pair (u,v): |N(u) ∩ N(v)| <= 2
    #
    # This means: if u and v already have 2 common neighbors, we cannot add
    # any edge from u or v to a vertex that's adjacent to the other.
    #
    # Let me implement a branch-and-bound that:
    # 1. Orders edges heuristically
    # 2. At each step, decides include/exclude
    # 3. Prunes when a constraint is violated or upper bound is exceeded

    # Precompute: for each pair (u,v), which edges would violate the constraint
    # if added when (u,v) already has 2 common neighbors.
    
    # Let me use a bitset-based approach for speed.
    # adj[u] is a bitmask of neighbors of u.
    
    def common_count(adj, u, v):
        return bin(adj[u] & adj[v]).count('1')

    def is_valid_add(adj, u, v):
        """Check if adding edge (u,v) maintains the K2,3-free property."""
        # After adding (u,v), for every neighbor w of u (w != v):
        #   common_count(adj_with_edge, v, w) must be <= 2
        #   Currently common_count(adj, v, w) is some value.
        #   After adding (u,v), u becomes a common neighbor of v and w
        #   (if u was already adjacent to w, which it is since w is neighbor of u).
        #   So common_count increases by 1 for pair (v,w) if u is already
        #   adjacent to w. Wait, u IS adjacent to w by assumption.
        #   So for pair (v,w): new common count = old common count + 1
        #   (because u is now adjacent to v, and u was already adjacent to w).
        #   We need old common count + 1 <= 2, i.e., old common count <= 1.
        
        # Similarly for every neighbor w of v (w != u):
        #   For pair (u,w): new common count = old common count + 1
        #   Need old common count <= 1.
        
        # Also, for pair (u,v) itself: common count doesn't change
        # (neither u nor v is a neighbor of itself).
        
        nv = adj[v]
        while nv:
            low = nv & (-nv)
            w = low.bit_length() - 1
            if w != u:
                if common_count(adj, u, w) > 1:
                    return False
            nv ^= low
        
        nu = adj[u]
        while nu:
            low = nu & (-nu)
            w = low.bit_length() - 1
            if w != v:
                if common_count(adj, v, w) > 1:
                    return False
            nu ^= low
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2

    # Greedy initialization
    def greedy_init():
        adj = [0] * n
        # Try edges in order of minimum potential conflicts
        candidates = all_pairs[:]
        random.shuffle(candidates)
        for u, v in candidates:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
        return adj

    # Local search: edge swap
    def local_search(adj, max_iters=2000):
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        no_improve = 0

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            cur_edges = []
            missing_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (current_adj[i] >> j) & 1:
                        cur_edges.append((i, j))
                    else:
                        missing_edges.append((i, j))

            if not cur_edges or not missing_edges:
                break

            # Try a random swap
            u1, v1 = cur_edges[random.randint(0, len(cur_edges) - 1)]
            u2, v2 = missing_edges[random.randint(0, len(missing_edges) - 1)]

            backup = current_adj[:]
            remove_edge(current_adj, u1, v1)
            if is_valid_add(current_adj, u2, v2):
                add_edge(current_adj, u2, v2)
                new_count = count_edges(current_adj)
                if new_count > best_count:
                    best_adj = current_adj[:]
                    best_count = new_count
                    no_improve = 0
                else:
                    current_adj = backup[:]
                    no_improve += 1
            else:
                current_adj = backup[:]
                no_improve += 1

            if no_improve > 100:
                break

        return best_adj, best_count

    # Branch and bound
    def branch_and_bound(adj, remaining_edges, best_count, depth=0):
        """
        Branch and bound: try adding edges from remaining_edges.
        adj: current adjacency (list of ints)
        remaining_edges: list of (u,v) edges to consider
        best_count: current best edge count
        """
        if time.time() > deadline - 3:
            return best_count

        current_count = count_edges(adj)
        if current_count + len(remaining_edges) <= best_count:
            return best_count  # Upper bound

        if not remaining_edges:
            return max(best_count, current_count)

        # Choose next edge to branch on
        # Heuristic: pick edge with minimum "cost" (most constrained)
        u, v = remaining_edges[0]
        rest = remaining_edges[1:]

        # Option 1: Try adding edge (u,v)
        if not ((adj[u] >> v) & 1):
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                # Filter remaining edges: remove any that are now invalid
                filtered = []
                for e in rest:
                    eu, ev = e
                    if (adj[eu] >> ev) & 1:
                        continue
                    if is_valid_add(adj, eu, ev):
                        filtered.append(e)
                best_count = branch_and_bound(adj, filtered, best_count, depth + 1)
                remove_edge(adj, u, v)

        # Option 2: Skip edge (u,v)
        best_count = branch_and_bound(adj, rest, best_count, depth + 1)

        return best_count

    # Main strategy
    best_adj_overall = None
    best_count_overall = 0

    # Strategy 1: Multiple greedy initializations + local search
    for r in range(50):
        if time.time() > deadline - 20:
            break
        adj = greedy_init()
        adj, cnt = local_search(adj, max_iters=500)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]

    # Strategy 2: Branch and bound from best solution found
    if best_adj_overall is not None and time.time() < deadline - 10:
        # Build remaining edges from best solution
        current_count = count_edges(best_adj_overall)
        remaining = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((best_adj_overall[i] >> j) & 1):
                    remaining.append((i, j))
        
        # Sort remaining edges by some heuristic
        # Edges that are "easier" to add first
        def edge_heuristic(e):
            u, v = e
            # Count how many constraints this edge would affect
            cnt = 0
            for w in range(n):
                if w != u and w != v:
                    cnt += common_count(best_adj_overall, u, w) + common_count(best_adj_overall, v, w)
            return cnt
        
        remaining.sort(key=edge_heuristic)
        
        # B&B with time limit
        best_count_overall = branch_and_bound(best_adj_overall[:], remaining, best_count_overall)

    # Final local search polish
    if best_adj_overall is not None and time.time() < deadline - 5:
        best_adj_overall, best_count_overall = local_search(best_adj_overall, max_iters=1000)

    # Convert to edge list
    if best_adj_overall is None:
        # Fallback: simple greedy
        adj = [0] * n
        for u, v in all_pairs:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        best_adj_overall = adj

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj_overall[i] >> j) & 1:
                edges.append([i, j])

    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
