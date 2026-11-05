import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 145s to ensure return.
    deadline = 145.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

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
            # Pivot: choose u in P union X with max degree in P
            # Optimization: Precompute degrees if possible, but for n<=40, this is fine
            u = max(P_X, key=lambda v: len(adj[v].intersection(P)))
            
            # Iterate over P \ N(u)
            for v in list(P - adj[u]):
                R_new = R.union({v})
                P_new = P.intersection(adj[v])
                X_new = X.intersection(adj[v])
                bk(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj, time_budget):
        cliques = bron_kerbosch_maximal_cliques(adj)
        if not cliques:
            return []
        
        n = len(adj)
        num_cliques = len(cliques)
        
        # Precompute: For each vertex, which cliques does it cover?
        # vertex_clique_indices[v] = list of clique indices containing v
        vertex_clique_indices = [[] for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                vertex_clique_indices[v].append(i)
        
        # Helper: Calculate number of uncovered cliques for a set S
        # We maintain cover_counts for speed in SA
        def get_uncovered_count(S, cover_counts=None):
            if cover_counts is None:
                cc = [0] * num_cliques
                for v in S:
                    for idx in vertex_clique_indices[v]:
                        cc[idx] += 1
                return sum(1 for c in cc if c == 0)
            else:
                return sum(1 for c in cover_counts if c == 0)

        # Helper: Prune redundant vertices from S
        # Returns a smaller valid set if possible
        def prune(S):
            S = set(S)
            if not S:
                return S
            
            # Calculate cover counts
            cover_counts = [0] * num_cliques
            for v in S:
                for idx in vertex_clique_indices[v]:
                    cover_counts[idx] += 1
            
            changed = True
            while changed:
                changed = False
                # Try to remove vertices in a random order to break symmetry
                for v in list(S):
                    # Check if v is the only cover for any clique
                    can_remove = True
                    for idx in vertex_clique_indices[v]:
                        if cover_counts[idx] == 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        S.remove(v)
                        for idx in vertex_clique_indices[v]:
                            cover_counts[idx] -= 1
                        changed = True
            
            return S

        def score_with_penalty(S, cover_counts):
            unc = get_uncovered_count(S, cover_counts)
            if unc > 0:
                return len(S) + 1000 * unc
            return len(S)

        # 1. Greedy Initialization
        def greedy_init():
            S = set()
            uncovered = set(range(num_cliques))
            
            while uncovered:
                best_v = -1
                max_cov = -1
                for v in range(n):
                    if v in S:
                        continue
                    cov = 0
                    for idx in vertex_clique_indices[v]:
                        if idx in uncovered:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                
                if best_v == -1 or max_cov == 0:
                    break
                
                S.add(best_v)
                for idx in vertex_clique_indices[best_v]:
                    if idx in uncovered:
                        uncovered.remove(idx)
            
            return S

        # 2. Simulated Annealing Refinement
        def simulated_annealing(initial_S, time_limit):
            # Initialize state
            current_S = set(initial_S)
            
            # Calculate initial cover counts
            cover_counts = [0] * num_cliques
            for v in current_S:
                for idx in vertex_clique_indices[v]:
                    cover_counts[idx] += 1
            
            current_score = score_with_penalty(current_S, cover_counts)
            
            # Best valid solution tracking
            best_valid_S = set(current_S) if current_score == len(current_S) else set()
            best_valid_size = len(best_valid_S) if best_valid_S else float('inf')
            
            # If initial is valid, prune it first to get a better start
            if current_score == len(current_S):
                pruned_S = prune(current_S)
                if len(pruned_S) < len(current_S):
                    current_S = pruned_S
                    # Recalc cover counts
                    cover_counts = [0] * num_cliques
                    for v in current_S:
                        for idx in vertex_clique_indices[v]:
                            cover_counts[idx] += 1
                    current_score = score_with_penalty(current_S, cover_counts)
                    if current_score == len(current_S):
                        best_valid_S = set(current_S)
                        best_valid_size = len(current_S)
            
            T = 1.0
            min_T = 0.01
            cooling = 0.995
            
            rng = random.Random()
            
            while time.time() < time_limit and T > min_T:
                # Generate neighbor
                move_type = rng.random()
                neighbor_S = set(current_S)
                
                if move_type < 0.5:
                    # Swap: Remove one, Add one
                    if current_S:
                        v_in = rng.choice(list(current_S))
                        v_out = rng.randint(0, n - 1)
                        if v_in != v_out:
                            neighbor_S.remove(v_in)
                            neighbor_S.add(v_out)
                        else:
                            # If same, just skip or do a remove/add?
                            # Let's just try a remove if swap fails
                            neighbor_S.remove(v_in)
                elif move_type < 0.8:
                    # Remove
                    if current_S:
                        v = rng.choice(list(current_S))
                        neighbor_S.remove(v)
                else:
                    # Add
                    v = rng.randint(0, n - 1)
                    neighbor_S.add(v)
                
                # Calculate new score
                # Optimization: We can update cover_counts incrementally, 
                # but for n<=40 and cliques~100, recalculating is fast enough 
                # and less error-prone. 
                # However, SA runs many steps. Let's try incremental.
                
                # To keep it simple and robust, we recalculate. 
                # If it's too slow, we can optimize later. 
                # n=40, max cliques ~100. 100*40 = 4000 ops. 
                # 1000 steps/s is easy.
                
                new_cover_counts = [0] * num_cliques
                for v in neighbor_S:
                    for idx in vertex_clique_indices[v]:
                        new_cover_counts[idx] += 1
                        
                new_score = score_with_penalty(neighbor_S, new_cover_counts)
                
                delta = new_score - current_score
                
                if delta < 0:
                    # Accept
                    current_S = neighbor_S
                    cover_counts = new_cover_counts
                    current_score = new_score
                    
                    if new_score == len(current_S):
                        # Valid solution found
                        # Prune to see if we can improve immediately
                        pruned_S = prune(current_S)
                        if len(pruned_S) < len(current_S):
                            current_S = pruned_S
                            # Recalc
                            cover_counts = [0] * num_cliques
                            for v in current_S:
                                for idx in vertex_clique_indices[v]:
                                    cover_counts[idx] += 1
                            current_score = score_with_penalty(current_S, cover_counts)
                            
                        if len(current_S) < best_valid_size:
                            best_valid_size = len(current_S)
                            best_valid_S = set(current_S)
                else:
                    # Accept with probability exp(-delta/T)
                    # Avoid overflow/underflow
                    if delta == 0:
                        accept = True
                    else:
                        # Use math.exp
                        prob = math.exp(-delta / T)
                        accept = rng.random() < prob
                    
                    if accept:
                        current_S = neighbor_S
                        cover_counts = new_cover_counts
                        current_score = new_score
                        
                        # Even if invalid, we might find a valid one later.
                        # We don't update best_valid_S here unless it's valid.
                        if current_score == len(current_S):
                            pruned_S = prune(current_S)
                            if len(pruned_S) < len(current_S):
                                current_S = pruned_S
                                cover_counts = [0] * num_cliques
                                for v in current_S:
                                    for idx in vertex_clique_indices[v]:
                                        cover_counts[idx] += 1
                                current_score = score_with_penalty(current_S, cover_counts)
                                
                            if len(current_S) < best_valid_size:
                                best_valid_size = len(current_S)
                                best_valid_S = set(current_S)
                
                T *= cooling
                
            return best_valid_S

        # Main loop for this instance
        # We will do a few restarts if time permits, but one long SA is usually best.
        # Let's do 1 Greedy + 1 Long SA.
        # If time remains, we can do another Greedy + Shorter SA.
        
        greedy_S = greedy_init()
        # Prune the greedy result before SA to give it a head start
        greedy_S = prune(greedy_S)
        
        # Allocate time for SA
        # Leave 5s for next instances? No, we have global deadline.
        # Let's just run SA until time_limit.
        
        best_S = simulated_annealing(greedy_S, time_limit)
        
        # Final safety prune
        best_S = prune(best_S)
        
        return sorted(best_S)

    out = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Distribute time budget
    # Total time: 145s. 
    # Graph generation and clique enumeration is very fast (<0.1s).
    # SA is the bottleneck.
    # Let's give each instance roughly 1/3 of the time, but with a minimum.
    # 145 / 3 = ~48s per instance.
    # We will track elapsed time and calculate remaining time for each instance.
    
    for i, (n, p, seed) in enumerate(instances):
        current_time = time.time()
        elapsed = current_time - start_time
        remaining = deadline - elapsed
        
        if remaining <= 1.0:
            # Emergency: Just return empty or best effort? 
            # If we are out of time, we should return what we have.
            # But we haven't solved this instance yet.
            # Fallback: Quick greedy
            adj = random_graph(n, p, seed)
            cliques = bron_kerbosch_maximal_cliques(adj)
            S = set()
            uncovered = set(range(len(cliques)))
            # Quick greedy
            while uncovered:
                best_v = -1
                max_cov = -1
                for v in range(n):
                    if v in S: continue
                    cov = 0
                    for c in cliques:
                        if v in c and c in uncovered: # This check is slow, use indices
                            pass 
                    # Simplified fallback
                    break 
                # Just return empty if we are in emergency, or try a very fast greedy
                # Actually, let's just do a standard greedy quickly
                break
            
            # Proper quick greedy
            S = set()
            uncovered_indices = set(range(len(cliques)))
            # Precompute indices
            v_idx = [[] for _ in range(n)]
            for idx, c in enumerate(cliques):
                for v in c:
                    v_idx[v].append(idx)
                    
            while uncovered_indices:
                best_v = -1
                max_cov = -1
                for v in range(n):
                    if v in S: continue
                    cov = 0
                    for idx in v_idx[v]:
                        if idx in uncovered_indices:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                if best_v == -1: break
                S.add(best_v)
                for idx in v_idx[best_v]:
                    if idx in uncovered_indices:
                        uncovered_indices.remove(idx)
            
            out.append(sorted(S))
            continue
        
        # Allocate time for this instance
        # If it's the last instance, give it all remaining time.
        # If not, give it remaining / (instances left).
        instances_left = len(instances) - i
        time_budget = remaining / instances_left
        
        # Cap time budget to avoid one instance taking too long if others are easy?
        # No, SA is efficient. Just use the allocated time.
        
        adj = random_graph(n, p, seed)
        res = solve_instance(adj, time_budget)
        out.append(res)
        
    return out

