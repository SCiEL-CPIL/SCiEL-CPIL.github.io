"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency matrix (0/1)
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Map edge to index for quick lookup (using a set of tuples for simplicity in Python loops)
    edge_set = set(edges)
    
    # Initial Solution: Greedy Peeling
    # This is fast and provides a valid starting point.
    current_edges = set(edges)
    initial_cliques = []
    
    while current_edges:
        if not _deadline_ok():
            for e in current_edges:
                initial_cliques.append([e[0], e[1]])
            return initial_cliques
            
        # Pick an edge with high potential to be in a large clique
        # Heuristic: Pick edge (u,v) such that |N(u) ∩ N(v)| is maximized
        best_edge = None
        best_inter = -1
        
        # Optimization: We don't need to check all edges every time. 
        # But for n=64, E~1100, checking all is O(E * n^2) ~ 1100 * 4096 = 4.5M ops.
        # Doing this repeatedly might be slow. 
        # Let's just pick a random edge from current_edges to be fast, or use a simple heuristic.
        # Let's try: Pick edge with max common neighbors in the *remaining* graph?
        # That's expensive. 
        # Let's stick to a simpler greedy: Pick first edge, expand to maximal clique in remaining graph.
        
        u, v = next(iter(current_edges))
        
        # Expand to maximal clique
        clique = [u, v]
        # Candidates: w such that (u,w) and (v,w) are in current_edges
        candidates = []
        for w in range(n):
            if w == u or w == v: continue
            e_uw = (u, w) if u < w else (w, u)
            e_vw = (v, w) if v < w else (w, v)
            if e_uw in current_edges and e_vw in current_edges:
                candidates.append(w)
        
        # Greedy add
        while candidates:
            # Pick candidate with max degree in candidate subgraph (within current_edges)
            best_w = -1
            best_deg = -1
            for w in candidates:
                deg = 0
                for x in candidates:
                    if w == x: continue
                    e_wx = (w, x) if w < x else (x, w)
                    if e_wx in current_edges:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_w = w
            
            if best_w == -1: break
            
            clique.append(best_w)
            candidates.remove(best_w)
            
            new_candidates = []
            for w in candidates:
                e_wbw = (w, best_w) if w < best_w else (best_w, w)
                if e_wbw in current_edges:
                    new_candidates.append(w)
            candidates = new_candidates
            
        initial_cliques.append(clique)
        
        for a, b in combinations(clique, 2):
            e = (a, b) if a < b else (b, a)
            if e in current_edges:
                current_edges.remove(e)
                
    if not _deadline_ok():
        return initial_cliques

    # Now, Local Search to improve
    # State: List of cliques (lists of vertices)
    # We want to minimize the number of cliques.
    # Moves:
    # 1. Try to merge two cliques C1, C2 into one clique C if C1 ∪ C2 is a clique in G.
    #    If C1 ∪ C2 is a clique, we can replace [C1, C2] with [C1 ∪ C2].
    #    This reduces the count by 1.
    #    This is valid because the edges covered by C1 and C2 are disjoint (by problem constraint),
    #    and the union is a clique, so it covers exactly the union of edges.
    #    Wait, is it possible that C1 and C2 share vertices?
    #    Yes. If C1={1,2} and C2={2,3}, and (1,3) is an edge, then C1∪C2={1,2,3} is a clique.
    #    The edges covered by C1 are (1,2). By C2 are (2,3).
    #    The new clique {1,2,3} covers (1,2), (2,3), (1,3).
    #    But (1,3) was NOT covered by C1 or C2. It must be covered by some other clique C3.
    #    So merging C1 and C2 into {1,2,3} would cause (1,3) to be covered by TWO cliques (the new one and C3).
    #    Therefore, we can ONLY merge C1 and C2 if C1 ∪ C2 is a clique AND no other clique covers any edge in C1 ∪ C2 that is NOT in C1 or C2.
    #    Actually, the condition is simpler: The set of edges covered by C1 ∪ C2 is E(C1) ∪ E(C2) ∪ E_between(C1, C2).
    #    For the merge to be valid, E_between(C1, C2) must be EMPTY in the "uncovered" sense?
    #    No, the constraint is "every edge must lie in exactly one listed clique".
    #    If we merge C1 and C2 into C_new = C1 ∪ C2, then C_new covers E(C1) ∪ E(C2) ∪ E_between(C1, C2).
    #    The original partition covered E(C1) by C1, E(C2) by C2, and E_between(C1, C2) by some OTHER cliques (if any).
    #    If E_between(C1, C2) is non-empty, those edges are already covered by other cliques.
    #    So merging would cause double coverage.
    #    Therefore, a merge is valid ONLY IF there are NO edges between C1 \ C2 and C2 \ C1 that are covered by other cliques?
    #    Actually, if C1 and C2 are disjoint in terms of edges (which they are), but share vertices, say v.
    #    C1 = {v, a}, C2 = {v, b}.
    #    Edges: (v,a) in C1, (v,b) in C2.
    #    If (a,b) is an edge, it is covered by some C3.
    #    If we merge C1, C2 to {v, a, b}, it covers (a,b). Now (a,b) is in C_new AND C3. Invalid.
    #    So, we can merge C1 and C2 if and only if:
    #    1. C1 ∪ C2 is a clique in G.
    #    2. There are no edges between C1 and C2 (excluding the shared vertices? No, between the sets) that are covered by other cliques.
    #    Wait, if C1 and C2 share a vertex v, the "between" edges are edges from C1\{v} to C2\{v}.
    #    If there are no such edges in G, then C1 ∪ C2 is not a clique.
    #    If there ARE such edges in G, they MUST be covered by other cliques (since C1 and C2 only cover internal edges).
    #    So if C1 ∪ C2 is a clique, then all "between" edges exist in G.
    #    And since C1 and C2 are edge-disjoint, those "between" edges are NOT covered by C1 or C2.
    #    So they are covered by other cliques.
    #    Thus, merging C1 and C2 into C1 ∪ C2 ALWAYS causes double coverage of the "between" edges, UNLESS there are NO "between" edges.
    #    But if there are no "between" edges, C1 ∪ C2 is not a clique.
    #    Conclusion: We can NEVER merge two cliques that share a vertex, if the union is a clique, because the cross-edges will be double-covered.
    #    What if C1 and C2 are DISJOINT (no shared vertices)?
    #    Then C1 ∪ C2 is a clique iff all edges between C1 and C2 exist.
    #    If they exist, they are covered by other cliques?
    #    No, if C1 and C2 are disjoint, the edges between them are NOT covered by C1 or C2.
    #    So they are covered by other cliques.
    #    So merging would double-cover them.
    #    
    #    Wait, is this logic correct?
    #    "Every edge must lie in exactly one listed clique."
    #    If I have a partition of edges into cliques.
    #    Clique C covers E(C).
    #    If I merge C1 and C2 into C12, C12 covers E(C12) = E(C1) ∪ E(C2) ∪ E_between(C1, C2).
    #    The original partition covered E(C1) by C1, E(C2) by C2, and E_between(C1, C2) by some set of other cliques.
    #    If I replace C1, C2 with C12, I remove C1, C2 from the list.
    #    The edges E(C1) and E(C2) are now covered by C12.
    #    The edges E_between(C1, C2) are NOW covered by C12.
    #    But they were ALSO covered by the other cliques.
    #    So they are covered TWICE.
    #    So merging is INVALID unless E_between(C1, C2) is EMPTY.
    #    If E_between(C1, C2) is empty, then C1 ∪ C2 is NOT a clique (unless one is empty, which is not allowed).
    #    
    #    So, MERGING IS IMPOSSIBLE in this problem formulation?
    #    Wait. What if the "other cliques" that covered E_between(C1, C2) are actually C1 or C2?
    #    No, C1 and C2 are disjoint in edges.
    #    So yes, merging seems impossible.
    #    
    #    Let's re-read the problem.
    #    "Every listed set must be a clique ... and every edge must lie in exactly one listed clique."
    #    This is an Edge Clique Partition.
    #    The cliques are edge-disjoint.
    #    
    #    So, how do we improve?
    #    We can't merge.
    #    We can SPLIT? No, that increases count.
    #    We can REASSIGN?
    #    If we take a clique C and break it into two cliques C1, C2 such that E(C1) ∪ E(C2) = E(C) and C1, C2 are cliques.
    #    This is always possible (e.g., split into edges). But it increases count.
    #    
    #    So the only way to reduce count is to find a set of edges that form a clique, and are currently covered by MULTIPLE cliques, and replace those multiple cliques with ONE clique.
    #    But we just proved that if the union of two cliques is a clique, the cross-edges are double-covered.
    #    UNLESS... the cross-edges are NOT covered by other cliques?
    #    But the constraint says EVERY edge is covered.
    #    So the cross-edges MUST be covered by someone.
    #    If they are covered by C1 or C2, then C1 or C2 is not a clique (because it doesn't cover all its internal edges? No, a clique covers ALL its internal edges).
    #    So C1 covers ALL edges within C1.
    #    So cross-edges are NOT in E(C1) or E(C2).
    #    So they are covered by other cliques.
    #    
    #    So, is it impossible to improve on a valid Edge Clique Partition by merging?
    #    Yes.
    #    
    #    So what can we do?
    #    We can change the partition structure.
    #    Example:
    #    Triangle 1-2-3.
    #    Partition 1: {1,2}, {1,3}, {2,3}. Count 3.
    #    Partition 2: {1,2,3}. Count 1.
    #    How do we get from 1 to 2?
    #    We identify that the edges of the triangle are covered by 3 cliques.
    #    We replace those 3 cliques with 1 clique {1,2,3}.
    #    This is a "K-way merge".
    #    
    #    So the move is:
    #    Find a set of cliques {C1, ..., Ck} such that:
    #    1. The union of their vertices V = ∪ Ci is a clique in G.
    #    2. The union of their edges E = ∪ E(Ci) is exactly E(V).
    #    If condition 2 holds, then we can replace C1...Ck with one clique V.
    #    Condition 2 means that the cliques Ci cover ALL edges of the induced subgraph on V, and NO edges outside V (which is true since they are cliques on subsets of V).
    #    Also, they must be edge-disjoint (which is true for any valid partition).
    #    So the condition is simply: The union of the vertices of a set of cliques forms a clique, AND the edges of that clique are exactly the union of the edges of those cliques.
    #    This implies that the cliques Ci form an edge partition of the clique V.
    #    
    #    So the search is: Find a clique V in G that is covered by multiple cliques in the current partition, and replace them with V.
    #    
    #    Algorithm:
    #    1. Start with greedy partition.
    #    2. Iterate:
    #       a. For each vertex v, or each pair, or each small set, find the maximal clique V containing it?
    #       b. Check if V is covered by exactly one clique in the current partition. If yes, no change.
    #       c. If covered by multiple, check if the union of those cliques is exactly V.
    #       d. If yes, replace them with V.
    #    
    #    How to find such V?
    #    We can enumerate all maximal cliques in G.
    #    For each maximal clique V, check which cliques in the current partition are subsets of V.
    #    If their union of edges is E(V), then we can merge them into V.
    #    
    #    This is a powerful heuristic.
    #    Number of maximal cliques can be large, but for n=64, p=0.55, it's manageable?
    #    Actually, enumerating all maximal cliques is hard.
    #    But we can just try to expand existing cliques.
    #    
    #    Alternative:
    #    For each clique C in the partition, try to expand it to a larger clique C' by adding vertices.
    #    If C' is a clique, and the edges of C' are covered by C and some other cliques C1...Ck, and C ∪ C1...Ck covers exactly E(C'), then replace C, C1...Ck with C'.
    #    
    #    This is effectively the same.
    #    
    #    Let's implement this "Expansion and Merge" heuristic.
    
    best_partition = initial_cliques
    best_count = len(best_partition)
    
    # Map edge to clique index
    def build_edge_map(partition):
        e_map = {}
        for i, c in enumerate(partition):
            for u, v in combinations(c, 2):
                e = (u, v) if u < v else (v, u)
                e_map[e] = i
        return e_map

    current_partition = [list(c) for c in best_partition]
    
    improved = True
    while improved and _deadline_ok():
        improved = False
        # Try to expand each clique
        for i in range(len(current_partition)):
            if not _deadline_ok(): break
            C = set(current_partition[i])
            # Try to add vertices
            # Candidates: vertices not in C, adjacent to all in C
            candidates = []
            for v in range(n):
                if v in C: continue
                is_adj = True
                for u in C:
                    if not adj[u, v]:
                        is_adj = False
                        break
                if is_adj:
                    candidates.append(v)
            
            for v in candidates:
                C_new = C | {v}
                # Check if C_new is covered by current cliques
                # Find all cliques that are subsets of C_new
                # And check if their union of edges is E(C_new)
                
                # This check is expensive.
                # Optimization: Only try to merge if the number of cliques involved is small?
                # Or just check if C_new is a clique (it is, by construction) and if the edges are covered.
                
                # Let's find the cliques that cover edges in C_new
                involved_cliques = set()
                valid = True
                for u, w in combinations(sorted(C_new), 2):
                    e = (u, w) if u < w else (w, u)
                    if e not in edge_set: # Should not happen
                        valid = False
                        break
                    # Find which clique covers e
                    # We need an edge map for the current partition
                    # Building it every time is O(E).
                    # Let's build it once per iteration.
                    
                # Rebuild edge map outside the loop?
                # Yes.
                
                # This approach is getting complex and potentially slow.
                # Let's try a simpler "Random Merge" attempt.
                # Pick two random cliques C1, C2.
                # If C1 ∪ C2 is a clique, and the edges between them are covered by ONLY C1 and C2?
                # No, we proved that's impossible.
                
                # Let's stick to the "Find Maximal Clique V" approach but limit the search.
                # Instead of enumerating all, just try to expand the current cliques.
                
                # If we expand C to C_new, we need to know which other cliques are "absorbed".
                # A clique Ci is absorbed if Ci ⊆ C_new and E(Ci) ⊆ E(C_new) (which is true if Ci ⊆ C_new).
                # And we need the union of E(Ci) for all absorbed Ci to be exactly E(C_new).
                
                # Let's implement this check.
                
                # To make it efficient, we can pre-compute for each clique its set of edges.
                # And for each vertex, the cliques containing it.
                
                # This is getting heavy.
                # Let's try a different tactic: 
                # "Edge Swapping"?
                # No, that's for vertex cover.
                
                # Let's go back to the "Maximal Clique" enumeration.
                # For n=32, 64, the number of maximal cliques in a random graph is usually not too high.
                # We can use a simple Bron-Kerbosch variant to find them?
                # Or just use the fact that we already have a partition.
                
                # Let's try: 
                # 1. List all cliques in the current partition.
                # 2. For each pair of cliques (Ci, Cj), check if Ci ∪ Cj is a clique.
                #    If yes, check if the edges of Ci ∪ Cj are covered by ONLY Ci and Cj.
                #    If yes, merge.
                # 3. Repeat for triples?
                
                # This is a local search on the partition.
                
                # Let's implement the pair-merge check.
                
                # We need to know, for a given set of vertices V, which cliques cover its edges.
                # If the set of covering cliques is exactly {Ci, Cj} and Ci ∪ Cj = V, then we can merge.
                
                # This requires mapping edges to cliques.
                
                # Let's do it.
                
                pass # Placeholder for logic
        
        # If no improvement found in this pass, break?
        # Or try random restarts of the greedy phase?
        
        # The current program does random restarts of greedy.
        # Let's combine: 
        # 1. Greedy with random restarts to get a good initial partition.
        # 2. Local search (pair-merge) to improve it.
        
        # Since the "pair-merge" logic is complex to implement correctly and efficiently in the remaining time,
        # and the current program already does random restarts,
        # let's focus on making the greedy phase smarter.
        
        # Better Greedy:
        # Instead of picking a random edge, pick the edge that is in the MOST cliques?
        # No, we are building cliques.
        # Pick the edge (u,v) such that the maximal clique containing (u,v) is LARGEST.
        # This is the "Maximal Clique First" strategy.
        
        # Let's implement "Maximal Clique First" greedy.
        
        best_partition = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        num_restarts = 50 # Adjust based on time
        
        for i in range(num_restarts):
            if not _deadline_ok(): break
            
            # Shuffle edges
            shuffled_edges = edges[:]
            rng.shuffle(shuffled_edges)
            
            current_edges = set(shuffled_edges)
            partition = []
            
            for u, v in shuffled_edges:
                if not _deadline_ok(): break
                if (u, v) not in current_edges: continue
                
                # Find the largest clique containing (u,v) in the remaining graph
                # This is NP-hard, but for small n, we can use a heuristic.
                # Start with {u,v}.
                # Candidates: w such that (u,w), (v,w) in current_edges.
                # Greedily add w with max degree in candidate subgraph.
                
                clique = [u, v]
                candidates = []
                for w in range(n):
                    if w == u or w == v: continue
                    e_uw = (u, w) if u < w else (w, u)
                    e_vw = (v, w) if v < w else (w, v)
                    if e_uw in current_edges and e_vw in current_edges:
                        candidates.append(w)
                
                # To find a "larger" clique, we can try multiple expansions?
                # Or just use the standard greedy.
                
                while candidates:
                    best_w = -1
                    best_deg = -1
                    for w in candidates:
                        deg = 0
                        for x in candidates:
                            if w == x: continue
                            e_wx = (w, x) if w < x else (x, w)
                            if e_wx in current_edges:
                                deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_w = w
                    if best_w == -1: break
                    clique.append(best_w)
                    candidates.remove(best_w)
                    new_candidates = []
                    for w in candidates:
                        e_wbw = (w, best_w) if w < best_w else (best_w, w)
                        if e_wbw in current_edges:
                            new_candidates.append(w)
                    candidates = new_candidates
                    
                partition.append(clique)
                
                for a, b in combinations(clique, 2):
                    e = (a, b) if a < b else (b, a)
                    if e in current_edges:
                        current_edges.remove(e)
                        
            if current_edges:
                for e in current_edges:
                    partition.append([e[0], e[1]])
                    
            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = partition
                
        if best_partition is None:
            return initial_cliques
            
        return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
