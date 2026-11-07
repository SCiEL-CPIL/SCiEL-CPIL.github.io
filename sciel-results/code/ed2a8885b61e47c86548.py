import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Preprocess edges
    edge_set = set()
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v: continue
        if u > v: u, v = v, u
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    def get_edge(u, v):
        return (min(u, v), max(u, v))

    def try_greedy_partition(random_seed):
        """
        Attempts a single greedy partition.
        Returns list of bicliques [[L, R], ...] or None if failed.
        """
        random.seed(random_seed)
        remaining_edges = set(edge_set)
        bicliques = []
        
        # Remaining adjacency for fast lookup
        # We'll maintain a list of active edges and update it
        # For speed, we might just iterate over remaining_edges each time if it's small enough
        # Or maintain a dynamic adjacency structure
        
        while remaining_edges:
            if time.time() > deadline:
                return None
            
            # 1. Select a seed vertex
            # Calculate degrees in the remaining graph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Find max degree vertex
            max_deg = max(deg)
            if max_deg == 0:
                break # Should not happen if remaining_edges is not empty
            
            candidates = [i for i in range(n) if deg[i] == max_deg]
            # Randomly pick one to introduce variation
            seed_v = random.choice(candidates)
            
            # 2. Initialize biclique
            # L contains seed_v, R contains its neighbors in remaining graph
            L = {seed_v}
            R = set()
            for u, v in remaining_edges:
                if u == seed_v:
                    R.add(v)
                elif v == seed_v:
                    R.add(u)
            
            if not R:
                # Isolated vertex in remaining graph? Impossible if deg > 0.
                # But if seed_v had deg 0, we wouldn't pick it.
                # If R is empty, something is wrong.
                # Fallback: pick a random edge
                e = next(iter(remaining_edges))
                u, v = e
                L = {u}
                R = {v}
            
            # 3. Expand biclique greedily
            # We want to add vertices to L or R that are connected to ALL vertices on the other side.
            # And these connections must be in remaining_edges.
            
            # Build a temporary adjacency for remaining edges for fast subset checks?
            # Or just check against remaining_edges set directly.
            # Since n=30, checking all pairs is O(n^2) which is 900.
            # Number of iterations is likely small.
            
            changed = True
            while changed:
                changed = False
                
                # Try to add to L
                # A vertex x can be added to L if for all r in R, (x,r) is in remaining_edges
                # And x is not in L or R
                best_L_candidate = None
                best_L_score = -1
                for x in range(n):
                    if x in L or x in R:
                        continue
                    # Check if x connects to all in R
                    valid = True
                    for r in R:
                        if get_edge(x, r) not in remaining_edges:
                            valid = False
                            break
                    if valid:
                        # Score: prefer vertices with high degree in remaining graph
                        # to encourage larger bicliques
                        score = deg[x]
                        # Add small random noise to break ties
                        score += random.uniform(0, 0.1)
                        if score > best_L_score:
                            best_L_score = score
                            best_L_candidate = x
                
                if best_L_candidate is not None:
                    L.add(best_L_candidate)
                    changed = True
                    # Update deg? No, deg is static for this expansion step roughly.
                    # Actually, adding to L doesn't change remaining edges yet.
                    
                # Try to add to R
                best_R_candidate = None
                best_R_score = -1
                for y in range(n):
                    if y in L or y in R:
                        continue
                    valid = True
                    for l in L:
                        if get_edge(l, y) not in remaining_edges:
                            valid = False
                            break
                    if valid:
                        score = deg[y]
                        score += random.uniform(0, 0.1)
                        if score > best_R_score:
                            best_R_score = score
                            best_R_candidate = y
                
                if best_R_candidate is not None:
                    R.add(best_R_candidate)
                    changed = True
            
            # Ensure L and R are disjoint (they should be by construction)
            if L & R:
                # This should not happen, but handle gracefully
                # Remove intersection from R
                R -= L
            
            if not L or not R:
                # Fallback
                e = next(iter(remaining_edges))
                u, v = e
                L = {u}
                R = {v}
            
            # 4. Remove covered edges
            for l in L:
                for r in R:
                    e = get_edge(l, r)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
            
            bicliques.append((L, R))
            
        return bicliques

    def merge_bicliques(bicliques):
        """
        Attempts to merge bicliques.
        Returns new list of bicliques.
        """
        changed = True
        while changed:
            changed = False
            new_bicliques = []
            i = 0
            while i < len(bicliques):
                L1, R1 = bicliques[i]
                j = i + 1
                merged = False
                while j < len(bicliques):
                    L2, R2 = bicliques[j]
                    
                    # Check if (L1 U L2) x (R1 U R2) is a valid biclique
                    # Conditions:
                    # 1. L1 U L2 and R1 U R2 are disjoint
                    # 2. All cross pairs are in edge_set
                    # 3. No "new" edges are formed that weren't in B1 or B2?
                    #    Actually, if it's a valid biclique in G, it covers ALL cross edges.
                    #    The edges covered by B1 are L1 x R1.
                    #    The edges covered by B2 are L2 x R2.
                    #    The edges covered by Merged are (L1 U L2) x (R1 U R2).
                    #    This set includes L1 x R2 and L2 x R1.
                    #    For the partition to be valid, L1 x R2 and L2 x R1 must be EMPTY in G.
                    #    Because if there was an edge (u,v) with u in L1, v in R2, it would be covered by the merged biclique,
                    #    but it was NOT covered by B1 or B2 (since v not in R1, u not in L2).
                    #    So it must have been covered by some OTHER biclique in the partition.
                    #    But we are merging B1 and B2, so that other biclique is still there.
                    #    Wait, if we merge B1 and B2, we remove them and add Merged.
                    #    If edge (u,v) exists in G, it MUST be covered exactly once.
                    #    It is covered by Merged.
                    #    Was it covered before?
                    #    If u in L1, v in R2: Not in B1 (v not in R1). Not in B2 (u not in L2).
                    #    So it was covered by some other biclique Bk.
                    #    If we merge B1 and B2, Bk is still there.
                    #    So edge (u,v) is covered by Merged AND Bk. Double coverage! Invalid.
                    #    Therefore, for a merge to be valid, there must be NO edges in G between L1 and R2, and NO edges between L2 and R1.
                    
                    if (L1 | L2) & (R1 | R2):
                        j += 1
                        continue
                    
                    # Check for cross edges
                    valid = True
                    for u in L1:
                        for v in R2:
                            if get_edge(u, v) in edge_set:
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        for u in L2:
                            for v in R1:
                                if get_edge(u, v) in edge_set:
                                    valid = False
                                    break
                            if not valid: break
                    
                    if valid:
                        # Also need to ensure that the merged biclique itself is valid in G
                        # i.e., all edges in (L1 U L2) x (R1 U R2) are in edge_set.
                        # We already know L1 x R1 and L2 x R2 are in edge_set (from previous steps).
                        # We just checked that L1 x R2 and L2 x R1 are NOT in edge_set.
                        # Wait, if they are NOT in edge_set, then the merged biclique is NOT a biclique in G?
                        # A biclique requires ALL cross pairs to be edges.
                        # If (u,v) is not an edge, then (L1 U L2) x (R1 U R2) is not a biclique.
                        # So the condition "No edges between L1 and R2" means the merged set is NOT a biclique.
                        
                        # Let me re-read the definition.
                        # "Every cross pair must be a host edge."
                        # So if I merge B1 and B2, the new biclique B_new has L_new = L1 U L2 and R_new = R1 U R2.
                        # For B_new to be valid, EVERY pair (u,v) with u in L_new, v in R_new must be an edge in G.
                        # This includes pairs (u,v) with u in L1, v in R2.
                        # So I MUST have edges between L1 and R2.
                        
                        # But if I have edges between L1 and R2, they are covered by B_new.
                        # Were they covered before?
                        # B1 covers L1 x R1. B2 covers L2 x R2.
                        # Edge (u,v) with u in L1, v in R2 is NOT in B1 or B2.
                        # So it must be in some other biclique Bk.
                        # If I merge B1 and B2, B_new covers (u,v). Bk still covers (u,v).
                        # Double coverage.
                        
                        # Conclusion: You can ONLY merge two bicliques if they are "compatible" in a way that doesn't create double coverage.
                        # This implies that the union of their edge sets must be exactly the edge set of the merged biclique.
                        # E(B1) U E(B2) = E(B1 U B2).
                        # E(B1 U B2) = E(B1) U E(B2) U (L1 x R2) U (L2 x R1).
                        # So we need (L1 x R2) U (L2 x R1) to be a subset of E(B1) U E(B2).
                        # But E(B1) = L1 x R1 and E(B2) = L2 x R2.
                        # L1 x R2 is disjoint from L1 x R1 and L2 x R2 (assuming L1, L2, R1, R2 are disjoint sets of vertices? No, vertices can be reused in biclique partition? No, "disjoint sides" usually means L and R are disjoint from each other within a biclique. But can a vertex be in L of B1 and L of B2? Yes. Can a vertex be in L of B1 and R of B2? No, because that would mean the vertex is on both sides of the partition? No, the partition is of EDGES. Vertices can appear in multiple bicliques.
                        # However, within a single biclique, L and R are disjoint.
                        
                        # If u is in L1 and v is in R2.
                        # If (u,v) is an edge, it is covered by B_new.
                        # Was it covered by B1? No, v not in R1.
                        # Was it covered by B2? No, u not in L2.
                        # So it was covered by some other biclique.
                        # Thus, merging B1 and B2 causes double coverage for any edge between L1 and R2 (or L2 and R1).
                        
                        # Therefore, a merge is valid ONLY IF there are NO edges in G between L1 and R2 AND NO edges in G between L2 and R1.
                        # BUT, if there are no edges, then B_new is NOT a biclique (since it requires ALL cross pairs to be edges).
                        
                        # This suggests that simple union merging is almost never possible unless the bicliques are "separated" in the graph structure, but then they wouldn't form a biclique together.
                        
                        # Wait, did I misunderstand "merge"?
                        # Maybe "merge" means combining the bicliques into a larger structure that is NOT a single biclique, but the problem asks for a list of bicliques.
                        # So the result must be a list of bicliques.
                        # If I merge B1 and B2 into B3, B3 must be a biclique.
                        # So the logic holds: B3 is a biclique only if all cross edges exist.
                        # And partition validity requires no double coverage.
                        # These two conditions are mutually exclusive for non-trivial bicliques.
                        
                        # UNLESS... the edges between L1 and R2 were NOT covered by any other biclique?
                        # That's impossible, all edges must be covered.
                        
                        # So, is it possible to reduce the number of bicliques?
                        # Yes, if we can replace a set of bicliques with a smaller set of bicliques.
                        # But merging two into one is the most obvious reduction.
                        # If merging is impossible, how did the score improve?
                        
                        # Perhaps the "merge" in the inspiration code was different?
                        # Or maybe my assumption about "double coverage" is wrong?
                        # "The cross-edge sets must partition all pinned graph edges exactly once."
                        # If I have B1 and B2. E1 and E2.
                        # If I replace them with B3. E3.
                        # We need E3 = E1 U E2.
                        # And B3 is a biclique.
                        # E3 = L3 x R3.
                        # E1 = L1 x R1. E2 = L2 x R2.
                        # If L3 = L1 U L2 and R3 = R1 U R2.
                        # Then E3 = (L1 U L2) x (R1 U R2) = (L1 x R1) U (L1 x R2) U (L2 x R1) U (L2 x R2).
                        # For E3 to equal E1 U E2, we need (L1 x R2) U (L2 x R1) to be empty.
                        # i.e., No edges between L1 and R2, and no edges between L2 and R1.
                        # But if there are no edges, B3 is not a biclique.
                        
                        # So, simple union merging is IMPOSSIBLE for valid biclique partitions.
                        
                        # So what did the inspiration code do?
                        # Maybe it didn't merge? Maybe it just found better initial partitions?
                        # The score improvement 1.038 -> 1.174 suggests a drop in raw value (more bicliques? No, lower raw value is better. Score = seed/value. Higher score = lower value. So 1.174 is better than 1.038. Raw value decreased).
                        # So the number of bicliques decreased.
                        
                        # How can you decrease the number of bicliques if you can't merge them?
                        # You can't.
                        # Unless... my understanding of "biclique" or "partition" is flawed.
                        
                        # Let's re-read carefully.
                        # "construct() -> list of bicliques, each [left_vertices,right_vertices] with nonempty disjoint sides. Every cross pair must be a host edge. The cross-edge sets must partition all pinned graph edges exactly once."
                        
                        # Okay, my analysis stands. You cannot merge two bicliques into one if they share vertices in a way that creates cross edges.
                        # If they are vertex-disjoint, you can't merge them into one biclique unless they form a complete bipartite graph together, which would imply edges between them, which would have to be covered by one of them originally, contradiction.
                        
                        # So, the only way to reduce the count is to find a BETTER partition from the start.
                        # The "merge" phase in the current code is likely a no-op or very rare.
                        # The improvement must come from the greedy selection.
                        
                        # So I should focus entirely on the greedy phase.
                        pass
                    j += 1
                if not merged:
                    new_bicliques.append(bicliques[i])
                i += 1
            if len(new_bicliques) < len(bicliques):
                bicliques = new_bicliques
                changed = True
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple randomized trials
    # Budget: 85s.
    # Each trial takes ~0.1s?
    # We can run ~500 trials.
    
    num_trials = 500
    for i in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        bicliques = try_greedy_partition(i)
        if bicliques is None:
            continue
            
        # Optional: Local search to improve?
        # Since merging is impossible, we can try to "swap" vertices to make bicliques larger?
        # But that's complex.
        
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_bicliques = bicliques
            
            # If we hit a very low count, we might stop early?
            # Theoretical minimum?
            # For n=30, maybe 10-15?
            if best_count <= 12:
                break

    if best_bicliques is None:
        # Fallback: single edges
        return [[ [u], [v] ] for u, v in edges]
        
    # Convert to required format
    result = []
    for L, R in best_bicliques:
        result.append([list(int(x) for x in sorted(L)), list(int(x) for x in sorted(R))])
        
    return result

# EVOLVE-BLOCK-END
