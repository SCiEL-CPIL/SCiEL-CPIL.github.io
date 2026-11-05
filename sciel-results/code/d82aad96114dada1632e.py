import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges

    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            P_X = P.union(X)
            if not P_X:
                return
            u = max(P_X, key=lambda v: len(adj[v].intersection(P)))
            
            for v in list(P - adj[u]):
                R_new = R.union({v})
                P_new = P.intersection(adj[v])
                X_new = X.intersection(adj[v])
                bk(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj, n, deadline):
        cliques = bron_kerbosch_maximal_cliques(adj)
        if not cliques:
            return []
        
        num_cliques = len(cliques)
        # Map cliques to indices
        # vertex -> bitmask of cliques
        vertex_masks = [0] * n
        for i, c in enumerate(cliques):
            mask_bit = 1 << i
            for v in c:
                vertex_masks[v] |= mask_bit
        
        all_mask = (1 << num_cliques) - 1
        
        def greedy_init():
            S = set()
            covered = 0
            while covered != all_mask:
                best_v = -1
                max_new = -1
                for v in range(n):
                    if v in S:
                        continue
                    new_cover = vertex_masks[v] & ~covered
                    cnt = bin(new_cover).count('1')
                    if cnt > max_new:
                        max_new = cnt
                        best_v = v
                if best_v == -1:
                    break
                S.add(best_v)
                covered |= vertex_masks[best_v]
            return S

        def local_search(S):
            S = set(S)
            covered = 0
            for v in S:
                covered |= vertex_masks[v]
            
            no_improve = 0
            max_no_improve = 100
            
            while time.time() < deadline and no_improve < max_no_improve:
                improved = False
                
                # 1. Try Removals
                for v in list(S):
                    new_covered = 0
                    for u in S:
                        if u != v:
                            new_covered |= vertex_masks[u]
                    if new_covered == all_mask:
                        S.remove(v)
                        covered = new_covered
                        improved = True
                        no_improve = 0
                        break # Restart loop with new S
                
                if improved:
                    continue
                
                # 2. Try Add-Remove-Remove
                # Identify critical vertices: vertices that are the ONLY cover for some clique
                # If we remove a critical vertex, we must add a vertex that covers its unique cliques.
                
                # Calculate coverage counts
                cover_count = [0] * num_cliques
                for v in S:
                    mask = vertex_masks[v]
                    # Count bits set in mask
                    # Efficient way: iterate set bits? Or just use a helper?
                    # For small num_cliques, loop is fine.
                    for i in range(num_cliques):
                        if mask & (1 << i):
                            cover_count[i] += 1
                
                # Find unique cliques for each v in S
                unique_cliques_for_v = {}
                for v in S:
                    mask = vertex_masks[v]
                    uniq = 0
                    for i in range(num_cliques):
                        if (mask & (1 << i)) and cover_count[i] == 1:
                            uniq |= (1 << i)
                    if uniq:
                        unique_cliques_for_v[v] = uniq
                
                # If there are no unique cliques, we can remove any vertex?
                # No, if no unique cliques, then cover_count[i] >= 2 for all i.
                # Then we can remove any vertex and still be covered.
                # But Step 1 would have caught that.
                # So if we are here, every vertex has at least one unique clique.
                
                # For each candidate v_add not in S:
                # We want to remove 2 vertices u1, u2 from S.
                # The new set is (S - {u1, u2}) U {v_add}.
                # This is valid if v_add covers all unique cliques of u1 and u2.
                
                # To be efficient, group vertices in S by their unique clique mask?
                # Or just iterate pairs? |S| is small.
                
                candidates_add = [v for v in range(n) if v not in S]
                
                # Optimization: Only consider v_add that covers "many" unique cliques?
                # Or just iterate all.
                
                found_move = False
                for v_add in candidates_add:
                    mask_add = vertex_masks[v_add]
                    
                    # Which vertices in S can be removed if we add v_add?
                    # A vertex u can be removed if its unique cliques are covered by v_add OR by other vertices in S (after removing u1, u2).
                    # This is complex.
                    
                    # Simpler: Iterate pairs (u1, u2) in S.
                    # Check if (S - {u1, u2}) U {v_add} is a valid cover.
                    # Valid if:
                    # 1. v_add covers all unique cliques of u1 that are NOT covered by S-{u1,u2,v_add}?
                    #    Actually, just check if the new set covers all_mask.
                    
                    # Precompute mask of S without u1, u2?
                    # mask_S = covered
                    # mask_new = (mask_S & ~vertex_masks[u1] & ~vertex_masks[u2]) | mask_add
                    # If mask_new == all_mask, valid.
                    
                    for i, u1 in enumerate(S):
                        mask_u1 = vertex_masks[u1]
                        for u2 in S:
                            if u1 == u2: continue
                            mask_u2 = vertex_masks[u2]
                            
                            # Calculate new coverage
                            # We remove u1 and u2 from S, add v_add.
                            # new_covered = 0
                            # for v in S:
                            #   if v != u1 and v != u2: new_covered |= vertex_masks[v]
                            # new_covered |= mask_add
                            
                            # Faster:
                            # base = covered
                            # We need to subtract u1 and u2.
                            # This is hard with bitmasks if they overlap.
                            # But |S| is small. Let's just recalculate for candidate pairs.
                            
                            # To speed up, only check pairs where u1 and u2 have "compatible" unique cliques.
                            # Unique cliques of u1 must be covered by v_add or other S vertices.
                            
                            # Let's just do the full check for now, it's O(|S|^2 * |S|) = O(|S|^3).
                            # |S| ~ 10. 10^3 = 1000.
                            # Candidates v_add ~ 30.
                            # Total 30000 checks per iteration. Very fast.
                            
                            new_covered = 0
                            for v in S:
                                if v != u1 and v != u2:
                                    new_covered |= vertex_masks[v]
                            new_covered |= mask_add
                            
                            if new_covered == all_mask:
                                # Valid move
                                S.remove(u1)
                                S.remove(u2)
                                S.add(v_add)
                                covered = new_covered
                                improved = True
                                no_improve = 0
                                found_move = True
                                break
                        if found_move:
                            break
                    if found_move:
                        break
                
                if not improved:
                    no_improve += 1
            
            return S

        # Solve
        best_S = greedy_init()
        best_S = local_search(best_S)
        best_size = len(best_S)
        
        # Iterated Local Search
        rng = random.Random(42)
        time_per_iter = (deadline - time.time()) / 20 if time.time() < deadline else 0
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Perturb
            S_pert = set(best_S)
            if S_pert:
                # Remove a random vertex
                v_remove = rng.choice(list(S_pert))
                S_pert.remove(v_remove)
                # Repair
                covered = 0
                for v in S_pert:
                    covered |= vertex_masks[v]
                while covered != all_mask:
                    # Add vertex with max new coverage
                    best_v = -1
                    max_new = -1
                    for v in range(n):
                        if v in S_pert:
                            continue
                        new_cover = vertex_masks[v] & ~covered
                        cnt = bin(new_cover).count('1')
                        if cnt > max_new:
                            max_new = cnt
                            best_v = v
                    if best_v == -1:
                        break
                    S_pert.add(best_v)
                    covered |= vertex_masks[best_v]
            
            # Local Search
            S_new = local_search(S_pert)
            if len(S_new) < best_size:
                best_S = S_new
                best_size = len(best_S)
        
        return sorted(best_S)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > deadline:
            # Fallback
            adj, edges = random_graph(n, p, seed)
            cliques = bron_kerbosch_maximal_cliques(adj)
            S = set()
            remaining = [set(c) for c in cliques]
            while remaining:
                best_v = None
                max_cov = -1
                for v in range(n):
                    if v in S: continue
                    cov = sum(1 for c in remaining if v in c)
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                if best_v is None: break
                S.add(best_v)
                remaining = [c for c in remaining if best_v not in c]
            out.append(sorted(S))
            continue
        
        adj, edges = random_graph(n, p, seed)
        res = solve_instance(adj, n, deadline)
        out.append(res)
        
    return out
