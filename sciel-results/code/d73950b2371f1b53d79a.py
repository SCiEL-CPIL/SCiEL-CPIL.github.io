import time
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Map edge (u, v) to an index 0..119
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    # Precompute edge indices for all pairs
    edge_idx_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = edge_index(u, v)

    # Inverse map: index -> (u, v)
    inv_edge_map = [None] * num_edges_total
    for (u, v), idx in edge_idx_map.items():
        inv_edge_map[idx] = (u, v)

    # Q3 abstract edges
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Local edge index mapping for K8
    def local_edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (7 - i)
        idx += (v - u - 1)
        return idx

    # Precompute all unique local Q3 edge sets (bitmasks of 28 bits)
    local_q3_edge_sets = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_index(u_val, v_val)
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(frozenset(edges_set))
    
    # Convert to list of edge lists for faster global mask generation
    local_q3_edge_lists = []
    for mask_set in local_q3_edge_sets:
        local_q3_edge_lists.append(list(mask_set))

    # Precompute global Q3 masks
    # For each subset of 8 vertices, map local edges to global edges
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    # Precompute local edge (u,v) -> local_idx for K8
    local_edge_to_idx = {}
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = local_edge_index(u, v)
            
    # Optimization: Precompute the list of global edge indices for each local edge index
    # for each subset? No, that's 12870 * 28 = 360k entries.
    # Better: for each subset, build a l2g array of size 28.
    
    for subset in all_subsets:
        # Map local edge idx to global edge idx
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
                
        # For each local Q3, generate the global mask
        for edge_list in local_q3_edge_lists:
            global_mask = 0
            for le in edge_list:
                ge = l2g[le]
                global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    if time.time() > deadline - 20:
        # Fallback: Return a known valid sparse graph if precomputation is too slow
        # K_{4,12} is Q3-free? Q3 has 8 vertices. 
        # A bipartite graph with parts of size 4 and 12.
        # Q3 is bipartite 4-4.
        # If one part is size 4, we can't fit Q3 if the other part is too small?
        # Q3 needs 4 vertices in each part.
        # If we have K_{4, 12}, we have 4 in left, 12 in right.
        # We can pick 4 from left and 4 from right. That forms K_{4,4}.
        # Does K_{4,4} contain Q3?
        # Q3 is a subgraph of K_{4,4}?
        # Q3 has 12 edges. K_{4,4} has 16 edges.
        # Q3 is 3-regular. K_{4,4} is 4-regular.
        # Yes, Q3 is a subgraph of K_{4,4}.
        # So K_{4,12} contains Q3.
        # We need a graph that is Q3-free.
        # A star graph K_{1,15} is Q3-free. 15 edges.
        # A matching is Q3-free.
        # Let's just return a random sparse graph or a star.
        return [[0, i] for i in range(1, 16)]

    num_q3s = len(global_q3_masks)
    
    # Greedy Set Cover on the complement
    # We want to remove edges to hit all Q3s.
    removed_edges = [False] * num_edges_total
    
    # q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # counts[e] = number of active Q3s containing e
    counts = [0] * num_edges_total
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    # active_q3s: boolean list, True if Q3 is not yet hit
    active_q3s = [True] * num_q3s
    
    # Greedy Loop
    while True:
        if time.time() > deadline - 10:
            break
            
        # Find edge with max count among not-removed edges
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if removed_edges[e]:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Remove best_edge
        removed_edges[best_edge] = True
        
        # Update active Q3s and counts
        for q3_idx in q3s_containing_edge[best_edge]:
            if active_q3s[q3_idx]:
                active_q3s[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb

    # Local Search Polish: try to re-add removed edges
    # A Q3 M is "present" in G = K \ R if M \cap R is empty.
    # We maintain removed_count[M] = |M \cap R|.
    # Initially, compute removed_count.
    removed_count = [0] * num_q3s
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            if removed_edges[e_idx]:
                removed_count[i] += 1
            m ^= lsb
            
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for e in range(num_edges_total):
            if not removed_edges[e]:
                continue
            
            # Check if we can re-add e
            # We can re-add e if for all M containing e, removed_count[M] > 1.
            # If we re-add e, removed_count[M] decreases by 1 for all M containing e.
            # The condition for M to NOT be formed is removed_count[M] > 0.
            # After re-adding e, new_count = removed_count[M] - 1.
            # We need new_count > 0 => removed_count[M] > 1.
            
            min_count = 999
            for q3_idx in q3s_containing_edge[e]:
                if removed_count[q3_idx] < min_count:
                    min_count = removed_count[q3_idx]
                if min_count <= 1:
                    break
                    
            if min_count > 1:
                # Safe to re-add
                removed_edges[e] = False
                improved = True
                # Update removed_count for all Q3s containing e
                for q3_idx in q3s_containing_edge[e]:
                    removed_count[q3_idx] -= 1

    # Construct final edge list
    final_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            final_edges.append([u, v])
            
    return final_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
