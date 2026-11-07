"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        if time.time() > deadline:
            # Fallback: return trivial solution (edges) if time runs out
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]
            
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # 1. Generate candidate cliques
        # In a split graph (C clique, I independent), a clique is a subset of C plus at most one vertex from I.
        # Specifically, for each v in I, let N(v) be its neighbors in C.
        # The maximal cliques involving v are of the form S ∪ {v} where S ⊆ N(v) is maximal?
        # Actually, any S ⊆ N(v) forms a clique S ∪ {v}.
        # To keep the ILP size manageable, we should restrict candidates.
        # A good set of candidates:
        # a) For each v in I: The clique N(v) ∪ {v}. (Maximal for v)
        # b) For each v in I: All cliques of size 2 {u, v} for u in N(v). (Base cases)
        # c) For C: The clique C itself.
        # d) For C: All cliques of size 2 {u, w} for u, w in C. (Base cases)
        
        # However, using ONLY N(v) ∪ {v} might not be optimal if we need to split edges to satisfy "exactly once" constraints with other cliques.
        # But note: Edges in E_CI are disjoint per v. So for a fixed v, edges (u,v) can only be covered by cliques containing v.
        # If we use a clique K containing v and S ⊆ N(v), it covers edges (u,v) for u in S AND edges within S.
        # If we use multiple cliques containing v, they partition N(v) into disjoint sets S1, S2, ...
        # The edges within S1 are covered by K1, within S2 by K2, etc.
        # The edges between S1 and S2 (which are in C) are NOT covered by K1 or K2. They must be covered by C-only cliques.
        
        # So the problem decomposes somewhat:
        # 1. Partition N(v) into sets S_{v,1}, ..., S_{v,m_v}.
        # 2. For each set, create clique S_{v,j} ∪ {v}.
        # 3. The edges within each S_{v,j} are "consumed".
        # 4. The remaining edges in C are those not within any S_{v,j}.
        # 5. Cover the remaining edges in C with C-only cliques.
        
        # To minimize total cliques:
        # Cost = sum_v m_v + (cliques to cover remaining C edges).
        
        # Let's formulate an ILP.
        # Variables:
        # x_{v, S} = 1 if we select clique S ∪ {v} for S ⊆ N(v).
        # y_{T} = 1 if we select clique T ⊆ C for T being a clique in the remaining graph.
        
        # This is too large if we enumerate all subsets S of N(v).
        # For k=30, N(v) can be size ~21. 2^21 is 2 million. Too many variables.
        
        # Restriction:
        # We only need to consider "maximal" subsets of N(v) that are cliques in the "available" C-graph?
        # No, the availability depends on other selections.
        
        # Alternative Strategy:
        # Since E_CI edges are independent per v, we can decide the partition of N(v) for each v independently IF we ignore the interaction on C edges.
        # But the interaction is: choosing a large S for v consumes C-edges, which might force more cliques for the rest of C.
        # Is it better to consume C-edges or not?
        # If we choose S={u}, we consume 0 C-edges. Cost 1 clique for 1 CI edge.
        # If we choose S={u, w}, we consume 1 C-edge (u,w). Cost 1 clique for 2 CI edges.
        # Saving: 1 clique. Cost: 1 C-edge consumed.
        # Consuming a C-edge saves 1 clique (the one that would have covered (u,w)).
        # So, net gain is 0?
        # Wait.
        # Seed:
        # - Cover (u,v) with {u,v}. Cost 1.
        # - Cover (u,w) with {u,w}. Cost 1.
        # Total 2 cliques for 3 edges.
        # New:
        # - Cover (u,v), (w,v), (u,w) with {u,w,v}. Cost 1.
        # Total 1 clique for 3 edges.
        # Gain: 1 clique.
        
        # So we WANT to group neighbors in C if the C-edges between them exist and are not "needed" elsewhere?
        # Actually, every C-edge MUST be covered.
        # If we cover (u,w) inside {u,w,v}, we don't need a separate clique for (u,w).
        # So yes, grouping is beneficial.
        
        # How to solve?
        # Let's try a greedy heuristic that is smarter than the parent.
        # Parent: Greedy maximal clique in N(v) based on remaining C-edges.
        # Problem with Parent: It processes v's sequentially. Early v's might consume C-edges that would have been better used by later v's?
        # Or maybe not, since C-edges are symmetric.
        
        # Let's try a different structural approach:
        # 1. Identify all "triangles" in the graph? No, split graphs have triangles only if they involve 2 C and 1 I.
        # 2. We can view this as a matching/packing problem.
        
        # Let's stick to the ILP but with a restricted set of candidates.
        # Candidates for v in I:
        # - All pairs {u, w} ∪ {v} for u, w in N(v) such that (u,w) is an edge.
        # - All singletons {u} ∪ {v} for u in N(v).
        # - The full set N(v) ∪ {v} (if N(v) is small enough? No, N(v) is up to 30).
        
        # Actually, if we allow ANY subset S ⊆ N(v), the number of variables is exponential.
        # But we can restrict to "maximal" cliques in the induced subgraph of N(v) on the C-edges?
        # No, N(v) is a subset of C. The edges between them are all present in the original graph (since C is a clique).
        # So ANY subset of N(v) is a clique in C.
        # The constraint is that the C-edges inside S must NOT be covered by any other clique.
        
        # Let's try a different decomposition.
        # Total Cliques = (Cliques for CI edges) + (Cliques for remaining CC edges).
        
        # For each v, we partition N(v) into blocks B_1, ..., B_m.
        # Each block B_j gives a clique B_j ∪ {v}.
        # The C-edges covered are all pairs within each B_j.
        # Let U_v be the set of C-edges covered by cliques involving v.
        # U_v = union over j of { (u, w) : u, w in B_j }.
        # Note: If u, w are in different blocks B_j, B_k, the edge (u,w) is NOT covered by v's cliques.
        
        # We need to choose partitions for all v such that the union of U_v over all v does not overlap (disjointness of covered C-edges).
        # And then cover the rest of C-edges with C-only cliques.
        
        # This is a complex global constraint.
        
        # Let's try a simpler heuristic:
        # 1. For each v, sort neighbors by some metric.
        # 2. Greedily form cliques.
        # 3. But allow "swap" moves?
        
        # Given the stagnation, let's try the ILP with a small candidate set.
        # Candidate set for v:
        # - All {u, v} (size 2)
        # - All {u, w, v} where u, w in N(v) and (u,w) is an edge. (Size 3)
        # - N(v) ∪ {v} (Size |N(v)|+1)
        
        # This limits the size of the "block" to 3 or full.
        # Is size 3 enough?
        # If we have 4 neighbors u1, u2, u3, u4.
        # We can form {u1, u2, v} and {u3, u4, v}.
        # This covers (u1,u2) and (u3,u4).
        # It leaves (u1,u3), (u1,u4), etc. for C-only cliques.
        # If we formed {u1, u2, u3, u4, v}, we cover all 6 C-edges among them.
        # Which is better?
        # Option A (Size 3s): 2 cliques for v. Covers 2 C-edges.
        # Option B (Full): 1 clique for v. Covers 6 C-edges.
        # Option B is clearly better (1 clique vs 2, and covers more C-edges).
        # So we should include larger cliques.
        
        # How many subsets of N(v) are there?
        # If |N(v)| = 20, 2^20 = 1M.
        # Total variables for all v: 60 * 1M = 60M. Too many for MILP.
        
        # We need to prune candidates.
        # Only keep "maximal" cliques in the sense of inclusion?
        # If S ⊂ T, and both are candidates, S is dominated by T?
        # Not necessarily. Using T might "waste" C-edges that could be covered by other v's?
        # No, covering a C-edge is always good (saves a clique).
        # So if T ⊃ S, using T covers all edges S covers, plus more.
        # And it uses 1 clique in both cases (if we replace S with T).
        # Wait, if we use S, we use 1 clique. If we use T, we use 1 clique.
        # T covers more C-edges. So T is strictly better?
        # Yes, provided T is a valid clique (which it is, since C is a clique and T ⊆ N(v)).
        # So for a fixed v, we should ALWAYS use the largest possible clique?
        # i.e., S = N(v)?
        
        # If we use S = N(v) for all v, we have a conflict:
        # Edge (u, w) might be in N(v) and N(w).
        # Then (u, w) is covered by clique K_v and K_w.
        # This is the conflict we identified before.
        
        # So we cannot use N(v) for all v.
        # We must choose subsets S_v ⊆ N(v) such that for any edge (u, w) in C, it is contained in S_v for at most one v.
        # This is a packing problem.
        
        # Let's formulate this as an ILP with variables x_{v, S} for "Maximal" S?
        # Or just x_{v, S} for all S?
        
        # Let's try a heuristic that resolves the conflict:
        # 1. Start with S_v = N(v) for all v.
        # 2. Find conflicts: edges (u, w) in multiple S_v.
        # 3. Resolve by removing one endpoint from one of the S_v's?
        #    If we remove u from S_v, the edge (u, v) must be covered by another clique for v.
        #    So we split S_v into S_v \ {u} and {u}.
        #    This increases the number of cliques for v by 1.
        #    But it resolves the conflict for all edges incident to u in S_v.
        
        # Algorithm:
        # 1. For each v, S_v = N(v).
        # 2. While there is a conflict (edge (u,w) in S_v and S_w):
        #    - Pick a conflict edge (u, w).
        #    - Pick one of the vertices, say u, and one of the sets, say S_v.
        #    - Remove u from S_v.
        #    - This creates a new "block" {u} for v.
        #    - Actually, we just need to ensure the partition of N(v) is valid.
        #    - If we remove u from the main block, we add a block {u}.
        #    - The edge (u, w) is no longer in the main block of v.
        #    - Is it covered by {u}? No, {u} ∪ {v} does not contain w.
        #    - So the edge (u, w) is no longer covered by v's cliques.
        #    - It is still covered by w's clique (if w is in S_w).
        #    - So the conflict is resolved.
        
        # This heuristic seems promising and fast.
        
        edges_cc = set((u, v) for u in range(k) for v in range(u+1, k))
        adj_ci = {v: set() for v in range(k, n)}
        
        for u, v in edges:
            if u < k and v >= k:
                adj_ci[v].add(u)
            elif v < k and u >= k:
                adj_ci[u].add(v)
        
        # Initialize blocks
        # blocks[v] is a list of sets (subsets of N(v))
        blocks = {v: [set(adj_ci[v])] for v in range(k, n)}
        
        # Find conflicts
        # For each edge (u, w) in C, check which v's have both u and w in their current "main" block?
        # Actually, we need to check if (u, w) is covered by any clique.
        # A clique K_v = S ∪ {v} covers (u, w) if u, w in S.
        
        # Let's maintain a count of how many times each C-edge is covered.
        # covered_count[(u, w)] = number of cliques containing u and w.
        
        covered_count = {}
        
        def update_counts(v, old_set, new_sets):
            # Remove counts for old_set
            if old_set:
                for u, w in combinations(sorted(old_set), 2):
                    if u < w:
                        e = (u, w)
                        covered_count[e] = covered_count.get(e, 0) - 1
            # Add counts for new_sets
            for S in new_sets:
                for u, w in combinations(sorted(S), 2):
                    if u < w:
                        e = (u, w)
                        covered_count[e] = covered_count.get(e, 0) + 1
        
        # Initial update
        for v in range(k, n):
            update_counts(v, set(), blocks[v])
            
        # Resolve conflicts
        # Find all edges with count > 1
        conflict_edges = [e for e, c in covered_count.items() if c > 1]
        
        while conflict_edges and time.time() < deadline - 5:
            # Pick a random conflict edge
            e = random.choice(conflict_edges)
            u, w = e
            
            # Find all v such that u, w are in the same block in blocks[v]
            candidates = []
            for v in range(k, n):
                for S in blocks[v]:
                    if u in S and w in S:
                        candidates.append((v, S))
            
            if not candidates:
                # Should not happen if count > 1
                continue
                
            # Pick a random candidate (v, S)
            v, S = random.choice(candidates)
            
            # Remove u from S
            S.remove(u)
            
            # If S is empty, remove it from the list
            new_blocks = []
            for S_prime in blocks[v]:
                if S_prime == S:
                    if S_prime:
                        new_blocks.append(S_prime)
                    # If S_prime was empty, skip
                else:
                    new_blocks.append(S_prime)
            
            # Add {u} as a new block
            new_blocks.append({u})
            
            blocks[v] = new_blocks
            
            # Update counts
            # We need to know the old set. It was S ∪ {u}.
            old_set = S | {u}
            update_counts(v, old_set, new_blocks)
            
            # Remove e from conflict_edges if resolved
            if covered_count.get(e, 0) <= 1:
                conflict_edges.remove(e)
            else:
                # Still conflicting, maybe add other edges that might be affected?
                # Just re-scan? Too slow.
                # Let's just keep the list and check at the end.
                pass
                
            # Periodically re-scan for new conflicts?
            # Or just loop until no conflicts found in the list?
            # The list might contain resolved edges.
            # Let's filter the list occasionally.
            if len(conflict_edges) > 1000:
                conflict_edges = [ee for ee in conflict_edges if covered_count.get(ee, 0) > 1]
        
        # Final check for conflicts
        # If any remain, we must break them by splitting further.
        # For simplicity, if conflicts remain, we fall back to size-2 for those specific edges?
        # No, that's hard.
        # Let's just accept the heuristic result and hope it's valid.
        # Wait, if a conflict remains, the solution is INVALID.
        # We MUST resolve all conflicts.
        
        # Robust resolution:
        # If conflict (u,w) remains, it means u,w are in the same block for some v1 and some v2.
        # We must break one of them.
        # We can do this in a loop until no conflicts.
        
        # Let's implement a safer loop.
        max_iter = 10000
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Find any conflict
            conflict_edge = None
            for e, c in covered_count.items():
                if c > 1:
                    conflict_edge = e
                    break
            
            if conflict_edge is None:
                break
                
            u, w = conflict_edge
            # Find v, S containing u, w
            found = False
            for v in range(k, n):
                for S in blocks[v]:
                    if u in S and w in S:
                        # Remove u from S
                        S.remove(u)
                        new_blocks = [S_prime for S_prime in blocks[v] if S_prime]
                        new_blocks.append({u})
                        blocks[v] = new_blocks
                        old_set = S | {u}
                        update_counts(v, old_set, new_blocks)
                        found = True
                        break
                if found:
                    break
            
            if not found:
                # This should not happen
                break
                
        # Construct the final list of cliques
        cliques = []
        
        # 1. Cliques involving I vertices
        for v in range(k, n):
            for S in blocks[v]:
                if S:
                    clique = list(S) + [v]
                    cliques.append(clique)
        
        # 2. Cover remaining C-edges
        # Find C-edges with count 0
        remaining_cc = []
        for u in range(k):
            for w in range(u+1, k):
                e = (u, w)
                if covered_count.get(e, 0) == 0:
                    remaining_cc.append(e)
        
        # Cover remaining_cc with size-2 cliques
        for u, w in remaining_cc:
            cliques.append([u, w])
            
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed-like solution
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            results.append(res)
            
    return results
# EVOLVE-BLOCK-END
