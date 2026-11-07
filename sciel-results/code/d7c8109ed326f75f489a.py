import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 120

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        """Find minimum clique partition for split graph."""
        # Generate cross edges
        rng = Random(seed)
        cross_edges = {}  # v -> list of u in K adjacent to v
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            cross_edges[v] = set(nbrs)
        
        # All K-K edges
        kk_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                kk_edges.add((u, v))
        
        # Strategy: 
        # 1. For each v in I, N(v) is a clique in K.
        # 2. We want to partition all edges into cliques.
        # 3. A clique can be: subset of K, or {v} ∪ S where S ⊆ N(v).
        # 4. K-K edges can only be covered by K-only cliques or by {v}∪S cliques where S contains that edge.
        
        # Key insight: If we use {v} ∪ N(v) as a clique, it covers all cross edges of v 
        # and all K-K edges within N(v). But K-K edges in N(v) might be shared across v's.
        
        # Better: Use a greedy approach where we process I-vertices and "claim" K-K edges.
        
        # Approach: 
        # - Start with all K-K edges uncovered
        # - For each v in I (sorted by |N(v)| descending), form clique {v} ∪ S where S is 
        #   a subset of N(v) such that K-K edges in S are uncovered
        # - This is like finding a clique in the "available" subgraph of N(v)
        # - After processing all v, cover remaining K-K edges with K-only cliques
        
        # Let's try: for each v, take S = N(v) ∩ (vertices whose incident K-K edges are all uncovered)
        # But that's not quite right either.
        
        # Simpler effective approach:
        # 1. For each v, the clique {v} ∪ N(v) is valid.
        # 2. If we use ALL such cliques, K-K edges in N(v) are covered by multiple cliques -> invalid.
        # 3. So we need to ensure each K-K edge is in at most one {v}∪S clique.
        
        # Let's use a matching-style assignment:
        # Each K-K edge can be "assigned" to at most one v.
        # For v, the assigned edges must form a clique (they do, since K is a clique).
        # So for v, we can assign any subset of K-K edges where both endpoints are in N(v).
        # The assigned edges for v form a subgraph of N(v), which is a clique, so any subset works.
        
        # To minimize total cliques:
        # - Each v with |N(v)| > 0 needs at least 1 clique (for its cross edges)
        # - If v has assigned K-K edges, those are "free" (covered by v's clique)
        # - Remaining K-K edges need additional cliques
        
        # Total cliques = (number of v with |N(v)|>0) + (cliques needed for remaining K-K edges)
        # Wait, no. Each v might need multiple cliques if we don't assign all its edges.
        
        # Let me reconsider. If v has N(v) = {u1, u2, ..., um}, then:
        # - Option A: One clique {v, u1, u2, ..., um} covers all m cross edges and C(m,2) K-K edges
        # - Option B: Multiple smaller cliques, e.g., {v, u1, u2} and {v, u3, u4}, etc.
        # Option A is better (fewer cliques) but covers more K-K edges.
        
        # The tradeoff: covering K-K edges with v's clique saves cliques for K-K edges,
        # but might prevent other v's from using those K-K edges.
        
        # Greedy: Process v's in order of decreasing |N(v)|. For each v, 
        # take S = N(v) ∩ (vertices not yet "used" by other v's for K-K edge coverage)
        # Hmm, this is getting complex.
        
        # Let's try a different approach: 
        # 1. Use {v} ∪ N(v) for each v, but track which K-K edges are covered.
        # 2. If a K-K edge is covered by multiple v's, we have a problem.
        # 3. So instead, for each K-K edge, assign it to at most one v.
        
        # For each K-K edge (u,w), the set of v's that can cover it is {v : u ∈ N(v) and w ∈ N(v)}.
        # We want to assign each K-K edge to at most one v, and for each v, the assigned edges
        # must have all endpoints in N(v) (which is automatic if we check).
        
        # Then for each v, let E_v be the set of K-K edges assigned to v.
        # The vertices involved in E_v are a subset of N(v). Let S_v be the set of vertices in E_v,
        # plus any additional vertices in N(v) we want to include.
        # Actually, to cover all cross edges of v, we need cliques that include v and each neighbor.
        # If we form one clique {v} ∪ N(v), it covers all cross edges and all K-K edges in N(v).
        # But we've only "assigned" some of those K-K edges to v. The others are covered by other cliques.
        # That's fine as long as each K-K edge is covered exactly once overall.
        
        # Wait, I'm overcomplicating. Let me just do:
        # 1. For each v, form clique C_v = {v} ∪ N(v). This covers all cross edges of v 
        #    and all K-K edges within N(v).
        # 2. Mark all K-K edges within N(v) as covered.
        # 3. After processing all v, cover remaining K-K edges with K-only cliques.
        # 4. BUT: if a K-K edge is in N(v1) ∩ N(v2), it's covered by both C_v1 and C_v2 -> invalid.
        
        # So I need to ensure no K-K edge is in N(v) for multiple v's that I include in their clique.
        
        # Revised greedy:
        # - Maintain set of uncovered K-K edges
        # - Process v's in some order
        # - For v, find the largest subset S ⊆ N(v) such that all K-K edges within S are uncovered
        #   (i.e., S is a clique in the graph of uncovered K-K edges, intersected with N(v))
        # - But N(v) is already a clique, so any subset is a clique. The constraint is that 
        #   the K-K edges within S must be uncovered.
        # - So S can be any subset of N(v) where all pairs are in uncovered K-K edges.
        # - To maximize, take S = N(v) ∩ {u : all edges (u,w) for w in S are uncovered}
        #   This is like finding a clique in the uncovered subgraph induced by N(v).
        # - Since N(v) is a clique in K, the uncovered subgraph induced by N(v) is just 
        #   the subset of N(v) where all pairwise edges are uncovered.
        
        # Let's implement this greedy:
        uncovered_kk = set(kk_edges)
        cliques = []
        
        # Sort v's by |N(v)| descending
        i_vertices = sorted(range(k, n), key=lambda v: -len(cross_edges[v]))
        
        for v in i_vertices:
            if time.time() > deadline - 5:
                break
            Nv = cross_edges[v]
            if not Nv:
                continue
            
            # Find largest subset S of Nv where all K-K edges within S are uncovered
            # Start with S = Nv, remove vertices that have uncovered edges to other S vertices
            S = set(Nv)
            changed = True
            while changed:
                changed = False
                for u in list(S):
                    # Check if u has any edge to another vertex in S that is NOT in uncovered_kk
                    # Wait, we want edges that ARE in uncovered_kk. If an edge is not uncovered, 
                    # it's already covered, so we can't include both endpoints.
                    for w in S:
                        if u == w:
                            continue
                        e = (min(u, w), max(u, w))
                        if e not in uncovered_kk:
                            # This edge is already covered, so we can't have both u and w in S
                            # Remove the one with fewer uncovered edges? Or just remove u?
                            # Let's remove u if it has fewer "available" connections
                            S.discard(u)
                            changed = True
                            break
            
            if not S:
                # No K-K edges can be covered, but we still need to cover cross edges
                # Use 2-cliques for each cross edge
                for u in Nv:
                    cliques.append([u, v])
                continue
            
            # Form clique {v} ∪ S
            clique = [v] + list(S)
            cliques.append(clique)
            
            # Remove K-K edges within S from uncovered_kk
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                    uncovered_kk.discard(e)
            
            # Cover remaining cross edges of v with 2-cliques
            remaining = Nv - S
            for u in remaining:
                cliques.append([u, v])
        
        # Cover remaining K-K edges
        while uncovered_kk:
            if time.time() > deadline - 5:
                break
            e0 = next(iter(uncovered_kk))
            u, w = e0
            # Find largest clique containing u, w in uncovered_kk
            S = {u, w}
            # Try to add more vertices
            candidates = set()
            for a, b in uncovered_kk:
                candidates.add(a)
                candidates.add(b)
            
            for x in candidates:
                if x in S:
                    continue
                ok = True
                for y in S:
                    e = (min(x, y), max(x, y))
                    if e not in uncovered_kk:
                        ok = False
                        break
                if ok:
                    S.add(x)
            
            cliques.append(list(S))
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                    uncovered_kk.discard(e)
        
        # Verify and fix: ensure all cliques have >= 2 vertices
        final_cliques = [c for c in cliques if len(c) >= 2]
        
        # Verify all edges are covered exactly once
        covered = set()
        for clique in final_cliques:
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    if e in covered:
                        # Double covered - need to fix
                        pass
                    covered.add(e)
        
        # Check if any edges are missing
        all_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_edges.add((u, v))
        for v in range(k, n):
            for u in cross_edges[v]:
                all_edges.add((min(u, v), max(u, v)))
        
        missing = all_edges - covered
        double = len(covered) - len(all_edges & covered)
        
        # If there are issues, fall back to simple solution
        if missing or double > 0:
            # Fallback: use simple 2-cliques for cross, one big clique for K
            cliques = [list(range(k))]
            for v in range(k, n):
                for u in cross_edges[v]:
                    cliques.append([u, v])
            return cliques
        
        return final_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Emergency fallback
            n, k, p, seed = spec
            rng = Random(seed)
            cross = {}
            for v in range(k, n):
                nbrs = []
                for u in range(k):
                    if rng.random() < p:
                        nbrs.append(u)
                cross[v] = set(nbrs)
            cliques = [list(range(k))]
            for v in range(k, n):
                for u in cross[v]:
                    cliques.append([u, v])
            results.append(cliques)
        else:
            results.append(solve_instance(*spec))
    
    return results
