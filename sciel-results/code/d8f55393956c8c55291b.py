import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(all_pairs)

    def count_k33_violations(adj):
        """
        Count the number of K3,3 subgraphs in the graph.
        A K3,3 is formed by two disjoint 3-sets {a,b,c} and {d,e,f}
        such that every vertex in the first set is connected to every
        vertex in the second set.
        
        We count each K3,3 subgraph once by requiring the first set 
        has smaller min vertex than the second set's min vertex.
        """
        violations = 0
        # Precompute adjacency sets for fast intersection
        adj_sets = [set() for _ in range(n)]
        for i in range(n):
            mask = adj[i]
            while mask:
                bit = mask & (-mask)
                j = bit.bit_length() - 1
                adj_sets[i].add(j)
                mask ^= bit
        
        # For each triple {a,b,c}, find all vertices connected to all three
        # Then for each such vertex set, check if any 3-subset forms a K3,3
        # More efficiently: for each triple, get common neighbors, then
        # count 3-subsets of common neighbors that are mutually "compatible"
        
        # Actually, a cleaner approach:
        # A K3,3 exists iff there are 6 vertices partitioned into two sets of 3
        # where all 9 cross-edges exist.
        # Count by: for each 3-set S, let CN(S) = common neighbors of S.
        # For each 3-subset T of CN(S) where min(T) > max(S) (to avoid double count),
        # check if all edges between S and T exist (they do by definition of CN).
        # But we also need S and T to be disjoint (guaranteed since T ⊆ CN(S) and 
        # CN(S) contains only vertices not in S... wait, not necessarily).
        
        # Let me be more careful. CN(S) = {v : v adjacent to all of a,b,c}.
        # v could be in S? No, because graph is simple, no self-loops.
        # So v ∉ S automatically.
        # But we need T ∩ S = ∅, which is automatic.
        # We need to count each K3,3 exactly once.
        # A K3,3 has two sides. We'll assign the side containing the smallest 
        # vertex as S.
        
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    # Common neighbors of {a, b, c}
                    common = adj_sets[a] & adj_sets[b] & adj_sets[c]
                    # Remove a, b, c themselves (shouldn't be there, but safe)
                    common.discard(a)
                    common.discard(b)
                    common.discard(c)
                    
                    if len(common) < 3:
                        continue
                    
                    # We want 3-subsets T of common such that min(T) > c
                    # (since c is the largest in S, this ensures S contains the 
                    #  smallest vertex of the 6)
                    # Actually, to avoid double counting: require that the smallest
                    # vertex of S ∪ T is in S. Since a < b < c, the smallest is a.
                    # So we just need T ⊆ common, |T| = 3, and we count each once.
                    # But the same K3,3 could be found from another triple S'.
                    # To fix: only count if min(S) < min(T).
                    
                    # Simpler: just count all 3-subsets T of common with min(T) > c
                    # This ensures S = {a,b,c} contains the global minimum of S∪T
                    # only if a < min(T), which is true since a < b < c < min(T).
                    
                    common_list = sorted(v for v in common if v > c)
                    m = len(common_list)
                    if m < 3:
                        continue
                    # Number of 3-subsets
                    violations += m * (m - 1) * (m - 2) // 6
        
        return violations

    def can_add_edge(adj, u, v):
        """Quick check: can we add edge (u,v) without creating a NEW K3,3?
        This is a necessary condition for speed. If it returns False, adding
        the edge definitely creates a K3,3. If True, it might or might not."""
        if adj[u] & (1 << v):
            return False
        
        # Adding edge u-v could create K3,3 where u-v is one of the 9 cross-edges.
        # For this to happen, we need 2 other vertices on u's side and 2 on v's side
        # such that all cross-edges exist.
        
        # u's side: {u, x, y} where x,y are common neighbors of u (not v)
        # v's side: {v, p, q} where p,q are common neighbors of v (not u)
        # All 9 cross-edges must exist: u-p, u-q, x-p, x-q, y-p, y-q, v-x, v-y, u-v
        # u-v is being added. u-p and u-q: p,q must be neighbors of u.
        # v-x and v-y: x,y must be neighbors of v.
        # x-p, x-q, y-p, y-q: x,y must be neighbors of p,q.
        
        # So: x ∈ N(u) ∩ N(v), y ∈ N(u) ∩ N(v), x ≠ y
        #     p ∈ N(u) ∩ N(x) ∩ N(y), q ∈ N(u) ∩ N(x) ∩ N(y), p ≠ q
        #     p,q ∉ {u,v,x,y}
        
        common_uv = (adj[u] & adj[v]) & ~(1 << u) & ~(1 << v)
        
        # Get list of common neighbors of u and v
        cu_list = []
        mask = common_uv
        while mask:
            bit = mask & (-mask)
            x = bit.bit_length() - 1
            cu_list.append(x)
            mask ^= bit
        
        if len(cu_list) < 2:
            return True  # Can't form K3,3 with u-v as cross-edge
        
        # For each pair (x,y) from cu_list, check if there exist p,q
        for i in range(len(cu_list)):
            for j in range(i + 1, len(cu_list)):
                x = cu_list[i]
                y = cu_list[j]
                # p must be in N(u) ∩ N(x) ∩ N(y) and not in {u,v,x,y}
                common_uxy = (adj[u] & adj[x] & adj[y]) & ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << y)
                # Need at least 2 such vertices
                count = common_uxy.bit_count()
                if count >= 2:
                    return False  # Adding u-v creates a K3,3
        
        return True

    def edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    # Simulated annealing
    def simulated_annealing(seed):
        random.seed(seed)
        
        # Start with a random sparse graph
        adj = [0] * n
        for u, v in all_pairs:
            if random.random() < 0.3:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        current_edges = edge_count(adj)
        current_violations = count_k33_violations(adj)
        
        best_adj = adj[:]
        best_edges = current_edges
        best_violations = current_violations
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        
        # Precompute for speed
        pair_index = {}
        for idx, (u, v) in enumerate(all_pairs):
            pair_index[(u, v)] = idx
            pair_index[(v, u)] = idx
        
        step = 0
        while T > min_T:
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            u, v = random.choice(all_pairs)
            has_edge = bool(adj[u] & (1 << v))
            
            # Try flipping
            if has_edge:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                delta_edges = -1
            else:
                # Quick reject if it would definitely create K3,3
                if not can_add_edge(adj, u, v):
                    # Still might accept at high T, but skip for efficiency
                    # Actually, let's allow it at high T
                    pass
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                delta_edges = 1
            
            # Compute new violations
            new_violations = count_k33_violations(adj)
            delta_violations = new_violations - current_violations
            new_edges = current_edges + delta_edges
            
            # Objective: maximize edges, minimize violations
            # Score = edges - lambda * violations
            lam = 2.0
            delta_score = delta_edges - lam * delta_violations
            
            # Accept if improvement, or with probability exp(-delta/T) if worsening
            if delta_score >= 0:
                accept = True
            else:
                # delta_score < 0
                if T > 0:
                    accept = random.random() < pow(2.718281828, delta_score / T)
                else:
                    accept = False
            
            if accept:
                current_violations = new_violations
                current_edges = new_edges
                if current_violations == 0 and current_edges > best_edges:
                    best_adj = adj[:]
                    best_edges = current_edges
                    best_violations = 0
            else:
                # Revert
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            T *= cooling
            step += 1
        
        # Final: try to fix violations greedily
        # Remove edges that are part of K3,3 violations until none remain
        violations = count_k33_violations(adj)
        while violations > 0 and time.time() < deadline - 10:
            # Find an edge that's in a violation and remove it
            # For simplicity, just randomly remove edges until no violations
            for u, v in random.sample(all_pairs, min(50, n_pairs)):
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if count_k33_violations(adj) == 0:
                        break
            violations = count_k33_violations(adj)
        
        final_edges = edge_count(adj)
        if final_edges > best_edges and count_k33_violations(adj) == 0:
            best_adj = adj[:]
            best_edges = final_edges
        
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Run multiple SA restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        adj, edges = simulated_annealing(i)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
        
        # Occasionally print progress
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Restart {i}, best edges: {best_edges}, elapsed: {elapsed:.1f}s")
    
    # Final polish: greedy add on best solution
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u, v in all_pairs:
                if time.time() > deadline - 2:
                    break
                if not (best_adj[u] & (1 << v)):
                    if can_add_edge(best_adj, u, v) and count_k33_violations(best_adj) == 0:
                        # Check if adding keeps K3,3-free
                        best_adj[u] |= (1 << v)
                        best_adj[v] |= (1 << u)
                        if count_k33_violations(best_adj) == 0:
                            best_edges = edge_count(best_adj)
                            improved = True
                        else:
                            best_adj[u] &= ~(1 << v)
                            best_adj[v] &= ~(1 << u)
    
    # Convert to edge list
    result_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
    
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
