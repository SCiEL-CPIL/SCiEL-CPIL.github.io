"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import random
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def bron_kerbosch_maximal_cliques(adj, n):
        """Yield all maximal cliques using Bron-Kerbosch with pivot selection."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            union_px = P | X
            if union_px:
                pivot = max(union_px, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                P.discard(v)
                X.add(v)
                
                bk(R, P_new, X_new)
                
                P.add(v)
                X.discard(v)
                R.discard(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(cliques, n):
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        if not relevant_cliques:
            return []
        
        # Precompute vertex to cliques mapping
        v_to_c = defaultdict(list)
        for i, c in enumerate(relevant_cliques):
            for v in c:
                v_to_c[v].append(i)
        
        num_cliques = len(relevant_cliques)
        
        def is_covered(S, cliques_subset=None):
            # Check if S covers all cliques in cliques_subset (or all if None)
            if cliques_subset is None:
                for c in relevant_cliques:
                    if not (c & S):
                        return False
            else:
                for i in cliques_subset:
                    if not (relevant_cliques[i] & S):
                        return False
            return True

        def greedy_random(rng):
            S = set()
            uncovered = set(range(num_cliques))
            # Copy v_to_c counts for efficiency? Or just recalculate.
            # Recalculating is fine for small n.
            
            while uncovered:
                # Calculate coverage for each vertex not in S
                best_v = -1
                best_score = -1
                candidates = []
                
                for v in range(n):
                    if v in S:
                        continue
                    # Count how many uncovered cliques v hits
                    cnt = 0
                    for i in v_to_c[v]:
                        if i in uncovered:
                            cnt += 1
                    if cnt > 0:
                        # Add some randomness
                        score = cnt + rng.random() * 0.1
                        if score > best_score:
                            best_score = score
                            best_v = v
                
                if best_v == -1:
                    break # Should not happen if graph is connected enough, but safety
                    
                S.add(best_v)
                for i in v_to_c[best_v]:
                    uncovered.discard(i)
            
            return S

        def local_search(S):
            # 1. Remove redundant vertices
            changed = True
            while changed:
                changed = False
                for v in list(S):
                    # Check if v is critical
                    critical = False
                    for i in v_to_c[v]:
                        # If v is the only one in S for clique i
                        c = relevant_cliques[i]
                        if not (c & (S - {v})):
                            critical = True
                            break
                    if not critical:
                        S.discard(v)
                        changed = True
            
            # 2. Try swaps: Remove v, Add u, Remove w
            # We want to reduce size.
            # Move: S -> (S - {v}) + {u} - {w}
            # This reduces size by 1 if w != u (which it is) and w != v.
            # Condition: New set covers all cliques.
            
            improved = True
            while improved:
                improved = False
                # Try all v in S
                for v in list(S):
                    # If we remove v, which cliques are uncovered?
                    uncovered_by_v = set()
                    for i in v_to_c[v]:
                        c = relevant_cliques[i]
                        if not (c & (S - {v})):
                            uncovered_by_v.add(i)
                    
                    if not uncovered_by_v:
                        # Should have been caught by redundancy check, but just in case
                        S.discard(v)
                        improved = True
                        break
                    
                    # We need a u not in S that covers all cliques in uncovered_by_v
                    # And after adding u, we need to find a w in S-{v} that becomes redundant.
                    
                    # Optimization: If |uncovered_by_v| > 0, u must be in intersection of these cliques?
                    # No, u must be in ALL cliques in uncovered_by_v.
                    # So u must be in intersection of relevant_cliques[i] for i in uncovered_by_v.
                    # And u not in S.
                    
                    # Find candidates for u
                    candidates_u = set()
                    first = True
                    for i in uncovered_by_v:
                        c = relevant_cliques[i]
                        if first:
                            candidates_u = set(c)
                            first = False
                        else:
                            candidates_u &= c
                    
                    candidates_u -= S # u must not be in S
                    
                    for u in candidates_u:
                        # Tentative new set: S' = S - {v} + {u}
                        S_temp = S - {v}
                        S_temp.add(u)
                        
                        # Now check if any w in S_temp \ {u} is redundant
                        # i.e., is there w in S - {v} such that S_temp - {w} still covers all?
                        # This is equivalent to: is w redundant in S_temp?
                        # A vertex w is redundant in S_temp if every clique containing w has another vertex in S_temp.
                        
                        # We only need to check w in S - {v} because u is new.
                        # Actually, u might make other vertices redundant.
                        
                        for w in list(S_temp):
                            if w == u:
                                continue
                            
                            # Check if w is redundant in S_temp
                            redundant = True
                            for i in v_to_c[w]:
                                c = relevant_cliques[i]
                                if not (c & (S_temp - {w})):
                                    redundant = False
                                    break
                            
                            if redundant:
                                # We can do the move: Remove v, Add u, Remove w
                                S = S_temp - {w}
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
            return S

        best_sol = None
        best_size = float('inf')
        
        # Multiple restarts
        rng = random.Random(42) # Fixed seed for reproducibility in solver, or use instance seed?
        # Using a fixed seed for the solver's internal randomness ensures deterministic output for the same input.
        
        num_restarts = 50
        
        for _ in range(num_restarts):
            S = greedy_random(rng)
            S = local_search(S)
            if len(S) < best_size:
                best_size = len(S)
                best_sol = S.copy()
                
            # Early stop if we found a very good solution?
            # Minimum possible is at least 1 if there are cliques.
            # If best_size == 1, we can stop.
            if best_size == 1:
                break

        return sorted(best_sol) if best_sol else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        
        # Solve
        cover = solve_instance(cliques, n)
        
        out.append(cover)
    
    return out
# EVOLVE-BLOCK-END
