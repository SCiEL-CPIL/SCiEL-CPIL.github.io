import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n = 16
    all_edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(all_edges)
            all_edges.append((i, j))
    
    num_edges = len(all_edges)  # 120

    # Precompute Q3 edge masks within an 8-vertex subset
    # Q3 edges: pairs (i,j) where i^j has exactly one bit set
    q3_edges_local = set()
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_local.add((i, j))
    
    # Map local edge (i,j) to a bit position in the 28-edge space of K8
    local_edge_to_bit = {}
    k8_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            local_edge_to_bit[(i, j)] = len(k8_edges)
            k8_edges.append((i, j))
    
    # Generate all distinct Q3 masks (subsets of 12 edges in K8 that form a Q3)
    # There are 8! = 40320 labelings, but many produce the same edge set.
    # A Q3 is determined by which 12 edges are present.
    q3_masks = set()
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b: a, b = b, a
            mask |= (1 << local_edge_to_bit[(a, b)])
        q3_masks.add(mask)
    q3_masks_list = list(q3_masks)
    # There are 112 distinct Q3 edge sets in K8 (8!/8/4!... actually 112 is a known number? Let's just use the set)
    # Actually the number of distinct 12-edge subsets of K8 that form a cube is 112.

    # Precompute all 8-subsets of vertices and their edge mappings
    subsets = list(combinations(range(n), 8))
    num_subsets = len(subsets)
    
    subset_mappings = []  # For each subset, list of (global_edge_idx, local_bit)
    edge_to_subsets = {i: [] for i in range(num_edges)}
    
    for idx, subset in enumerate(subsets):
        mapping = []
        for i in range(8):
            for j in range(i + 1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u, v)]
                le_bit = local_edge_to_bit[(i, j)]
                mapping.append((ge_idx, le_bit))
                edge_to_subsets[ge_idx].append(idx)
        subset_mappings.append(mapping)

    # Cache for checking if a local 28-bit mask contains a Q3
    contains_q3_cache = {}
    
    def contains_q3(local_mask):
        if local_mask in contains_q3_cache:
            return contains_q3_cache[local_mask]
        for qm in q3_masks_list:
            if (local_mask & qm) == qm:
                contains_q3_cache[local_mask] = True
                return True
        contains_q3_cache[local_mask] = False
        return False

    def find_violating_q3_in_subset(s_idx, mask):
        """Return list of global edge indices forming a Q3, or None."""
        mapping = subset_mappings[s_idx]
        local_mask = 0
        for ge_idx, le_bit in mapping:
            if (mask >> ge_idx) & 1:
                local_mask |= (1 << le_bit)
        for qm in q3_masks_list:
            if (local_mask & qm) == qm:
                viol_edges = []
                for ge_idx, le_bit in mapping:
                    if (qm >> le_bit) & 1:
                        viol_edges.append(ge_idx)
                return viol_edges
        return None

    def is_safe_to_add(edge_idx, mask):
        """Check if adding edge_idx to mask creates a Q3."""
        for s_idx in edge_to_subsets[edge_idx]:
            mapping = subset_mappings[s_idx]
            local_mask = 0
            new_edge_bit = None
            for ge_idx, le_bit in mapping:
                if ge_idx == edge_idx:
                    new_edge_bit = le_bit
                elif (mask >> ge_idx) & 1:
                    local_mask |= (1 << le_bit)
            test_local_mask = local_mask | (1 << new_edge_bit)
            if contains_q3(test_local_mask):
                return False
        return True

    def count_violations(mask):
        """Count total number of Q3 subgraphs in the graph defined by mask."""
        total = 0
        for s_idx in range(num_subsets):
            mapping = subset_mappings[s_idx]
            local_mask = 0
            for ge_idx, le_bit in mapping:
                if (mask >> ge_idx) & 1:
                    local_mask |= (1 << le_bit)
            # Count how many Q3 masks are contained
            for qm in q3_masks_list:
                if (local_mask & qm) == qm:
                    total += 1
        return total

    def find_any_violation(mask):
        """Find one Q3 violation, return list of edge indices or None."""
        # Try random subsets first for speed
        indices = list(range(num_subsets))
        random.shuffle(indices)
        for s_idx in indices:
            viol = find_violating_q3_in_subset(s_idx, mask)
            if viol:
                return viol
        return None

    subsets_indices_cache = list(range(num_subsets))

    # --- Initial Construction ---
    # Start with a known Q3-free graph: the complete bipartite graph K_{4,12} has 48 edges?
    # K_{4,12} is Q3-free? Q3 is bipartite 4+4. K_{4,12} contains K_{4,4} as a subgraph.
    # K_{4,4} is NOT Q3. Q3 has 12 edges, K_{4,4} has 16.
    # Does K_{4,4} contain Q3? Yes, Q3 is a subgraph of K_{4,4}.
    # So K_{4,12} is NOT Q3-free.
    
    # Let's try the complement of the 4x4 grid?
    # Or start with K_{3,13} (39 edges) which is known to be Q3-free.
    # K_{3,13}: any Q3 would need 4 vertices on each side. We only have 3 on one side.
    # So K_{3,13} is Q3-free.
    
    # Let's try a better start: Turan graph T_4(16)?
    # T_4(16) is complete 4-partite with parts of size 4.
    # Edges = 16*15/2 - 4*(4*3/2) = 120 - 24 = 96.
    # Is T_4(16) Q3-free?
    # Q3 is bipartite. A bipartite subgraph of T_4(16) can only use vertices from 2 parts.
    # The largest bipartite subgraph is K_{4,4} (using two parts of size 4).
    # As established, K_{4,4} contains Q3.
    # So T_4(16) is NOT Q3-free.

    # What about T_5(16)? Parts: 4,3,3,3,3.
    # Bipartite subgraph uses 2 parts. Max is K_{4,3} or K_{3,3}.
    # K_{4,3} has 12 edges. Q3 has 12 edges.
    # Is K_{4,3} isomorphic to Q3? No, Q3 has 8 vertices, K_{4,3} has 7.
    # So T_5(16) is Q3-free!
    # Number of edges in T_5(16):
    # Total 120 - (6+3+3+3+3) = 120 - 18 = 102 edges.
    # Let's verify: parts sizes 4,3,3,3,3.
    # Edges within parts: 4C2 + 3*3C2 = 6 + 9 = 15? No.
    # 4C2 = 6. 3C2 = 3. Four parts of size 3? No, 16 = 4+3+3+3+3.
    # Edges to remove: 6 + 3 + 3 + 3 + 3 = 18.
    # 120 - 18 = 102.
    
    # Let's construct T_5(16) as the starting point.
    # Vertices 0-3: Part 1 (size 4)
    # Vertices 4-6: Part 2 (size 3)
    # Vertices 7-9: Part 3 (size 3)
    # Vertices 10-12: Part 4 (size 3)
    # Vertices 13-15: Part 5 (size 3)
    
    best_mask = 0
    parts = [[0,1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]]
    for i in range(len(parts)):
        for j in range(i+1, len(parts)):
            for u in parts[i]:
                for v in parts[j]:
                    idx = edge_index[(min(u,v), max(u,v))]
                    best_mask |= (1 << idx)
    
    best_count = best_mask.bit_count()
    current_mask = best_mask

    # Verify initial is Q3-free (should be, but just in case)
    viol = find_any_violation(current_mask)
    while viol:
        rem_idx = random.choice(viol)
        current_mask &= ~(1 << rem_idx)
        best_mask = current_mask
        best_count = best_mask.bit_count()
        viol = find_any_violation(current_mask)

    # --- Local Search with Simulated Annealing ---
    # We want to maximize edges.
    # Move: add an edge, remove an edge, or swap an edge (remove one, add another).
    # Cost function: We want to minimize violations and maximize edges.
    # Let's use a penalty method: Score = -violations * 1000 - edges.
    # Or simply: try to add edges if safe. If not safe, try to remove a "blocking" edge.
    
    # Given the complexity of checking safety (120 edges, ~12870 subsets),
    # is_safe_to_add is expensive.
    # Let's estimate cost:
    # is_safe_to_add checks ~1000 subsets on average?
    # Each check involves building a 28-bit mask (28 ops) and checking 112 Q3 masks (112 ops).
    # So ~1000 * 140 = 140,000 ops per edge addition check.
    # 120 edges * 140,000 = 16.8M ops per full pass.
    # Python is slow. 16.8M simple ops might take ~1-2 seconds.
    # We have 145s. So we can do ~50-100 full passes.
    
    # Let's do a greedy improvement loop first.
    # Then a perturbation loop.

    for iteration in range(1000):
        if time.time() > deadline - 5:
            break

        # 1. Greedy Improvement: try to add any missing edge
        missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
        random.shuffle(missing_edges)
        
        improved = True
        while improved:
            improved = False
            for idx in missing_edges:
                if time.time() > deadline - 5:
                    break
                if is_safe_to_add(idx, current_mask):
                    current_mask |= (1 << idx)
                    missing_edges.remove(idx)
                    improved = True
                    if current_mask.bit_count() > best_count:
                        best_count = current_mask.bit_count()
                        best_mask = current_mask

        # 2. Perturbation: remove a random set of edges to escape local optima
        if time.time() > deadline - 10:
            break
            
        current_edges = [i for i in range(num_edges) if (current_mask >> i) & 1]
        if len(current_edges) > 10:
            # Remove 5-15 edges
            num_remove = random.randint(5, 15)
            to_remove = random.sample(current_edges, num_remove)
            for idx in to_remove:
                current_mask &= ~(1 << idx)
        
        # 3. Repair: if we created violations, remove edges until Q3-free
        # Since we removed random edges, we might have broken a Q3, but we might have also
        # created a situation where adding edges is harder.
        # Actually, removing edges cannot CREATE a Q3. It can only destroy them.
        # So if the graph was Q3-free, it remains Q3-free.
        # So no repair needed after removal.
        
        # However, we need to ensure we don't get stuck in a low-edge state.
        # The greedy step will try to add edges back.

    # Final Polish
    if time.time() < deadline - 2:
        missing_edges = [i for i in range(num_edges) if not (best_mask >> i) & 1]
        random.shuffle(missing_edges)
        for idx in missing_edges:
            if time.time() > deadline - 1:
                break
            if is_safe_to_add(idx, best_mask):
                best_mask |= (1 << idx)
                best_count = best_mask.bit_count()

    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result
