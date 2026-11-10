import time
import random
import sys

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        
        if P or X:
            # Pivot selection: choose u in P U X with max degree in P
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
            
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return list(bron_kerbosch(adj, set(), set(range(n)), set()))

    def is_valid_transversal(trans, cliques, v_to_cliques):
        # Check if every clique has at least one vertex in trans
        # We can do this by checking if the union of cliques hit by trans covers all cliques
        # But a simpler check given v_to_cliques is:
        # For each clique, is there a vertex in trans?
        # Actually, using v_to_cliques, we can just check if all cliques are hit.
        # Let's stick to a direct check for clarity and correctness in the inner loop, 
        # but we can optimize by precomputing the set of covered cliques.
        
        # Optimization: Precompute covered cliques
        covered = set()
        for v in trans:
            covered.update(v_to_cliques[v])
            
        # If the number of covered cliques equals total cliques, it's valid
        # But we need to be careful: v_to_cliques maps v to cliques containing v.
        # So if union of v_to_cliques[v] for v in trans == all_cliques_indices, it's valid.
        
        # Let's use a counter approach for speed if called frequently
        # However, for local search, set union is fast enough for small N.
        
        if len(covered) != len(cliques):
            return False
        return True

    def greedy_set_cover(cliques, n, rng, v_to_cliques):
        uncovered = set(range(len(cliques)))
        selected = set()
        
        while uncovered:
            best_v = -1
            best_count = -1
            ties = []
            
            # Only consider vertices that can cover at least one uncovered clique
            # We can iterate all vertices, or maintain a candidate set.
            # For small N (40), iterating all is fine.
            for v in range(n):
                count = len(v_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
                    ties = [v]
                elif count == best_count:
                    ties.append(v)
            
            if best_count == 0:
                # This shouldn't happen if cliques are non-empty and we have vertices
                break
                
            if len(ties) > 1:
                best_v = rng.choice(ties)
                
            selected.add(best_v)
            uncovered -= v_to_cliques[best_v]
            
        return selected

    def local_search(trans, cliques, rng, v_to_cliques, max_steps=1000):
        trans = set(trans)
        n = max(max(v for c in cliques for v in c) + 1, 1) if cliques else 0
        all_vertices = set(range(n))
        
        # 1. Remove redundant vertices
        changed = True
        while changed:
            changed = False
            for v in list(trans):
                new_trans = trans - {v}
                if is_valid_transversal(new_trans, cliques, v_to_cliques):
                    trans = new_trans
                    changed = True
        
        # 2. Try swaps and 2-for-1 moves
        for _ in range(max_steps):
            improved = False
            
            # Try 1-in, 1-out (Swap)
            if trans:
                v = rng.choice(list(trans))
                others = all_vertices - trans
                if others:
                    # Try a few random swaps
                    for _ in range(5):
                        w = rng.choice(list(others))
                        new_trans = (trans - {v}) | {w}
                        if is_valid_transversal(new_trans, cliques, v_to_cliques):
                            # Size is same, but might enable further reductions?
                            # Usually we don't accept same size unless it leads to smaller.
                            # For now, skip same-size swaps unless we are stuck.
                            pass
            
            # Try 2-for-1 move: Remove u, v and add w
            # This reduces size by 1.
            if len(trans) >= 2:
                # Pick two random vertices from trans
                u, v = rng.sample(list(trans), 2)
                trans_without = trans - {u, v}
                
                # Find a vertex w not in trans that covers all cliques that were ONLY covered by u or v
                # Or more simply: Find w such that (trans_without | {w}) is valid.
                # To be efficient, identify "critical" cliques: those that have no other vertex in trans_without.
                # A clique c is critical if (c & trans_without) is empty.
                # We need w to be in ALL critical cliques.
                
                critical_cliques = []
                for i, c in enumerate(cliques):
                    if not (set(c) & trans_without):
                        critical_cliques.append(set(c))
                
                if not critical_cliques:
                    # If no critical cliques, then trans_without is already valid?
                    # No, that would mean u and v were redundant. We should have caught that in step 1.
                    # So if no critical cliques, removing u and v leaves a valid set?
                    # Actually, if a clique is not critical, it has a vertex in trans_without.
                    # If ALL cliques are non-critical, then trans_without is valid.
                    # In that case, we can just remove u and v.
                    if is_valid_transversal(trans_without, cliques, v_to_cliques):
                        trans = trans_without
                        improved = True
                        break
                else:
                    # Find intersection of all critical cliques
                    # w must be in the intersection
                    candidates_w = set.intersection(*critical_cliques)
                    # w must not be in trans
                    candidates_w -= trans
                    
                    if candidates_w:
                        w = rng.choice(list(candidates_w))
                        new_trans = trans_without | {w}
                        if is_valid_transversal(new_trans, cliques, v_to_cliques):
                            trans = new_trans
                            improved = True
                            break
            
            # Try 1-in, 2-out is covered above.
            # Try 0-in, 1-out is covered in step 1.
            
            # If no improvement, we might try to shuffle slightly to escape local optima?
            # Or just stop if we've done many steps.
            
        return trans

    def simulated_annealing(trans, cliques, rng, v_to_cliques, start_temp=1.0, end_temp=0.01, steps=2000):
        trans = set(trans)
        current_cost = len(trans)
        best_trans = set(trans)
        best_cost = current_cost
        
        n = max(max(v for c in cliques for v in c) + 1, 1) if cliques else 0
        all_vertices = set(range(n))
        
        for i in range(steps):
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            
            # Generate a neighbor
            r = rng.random()
            new_trans = None
            
            if r < 0.4 and trans:
                # Remove a random vertex
                v = rng.choice(list(trans))
                new_trans = trans - {v}
            elif r < 0.7 and all_vertices - trans:
                # Add a random vertex
                v = rng.choice(list(all_vertices - trans))
                new_trans = trans | {v}
            else:
                # Swap
                if not trans:
                    continue
                v = rng.choice(list(trans))
                others = all_vertices - trans
                if not others:
                    continue
                w = rng.choice(list(others))
                new_trans = (trans - {v}) | {w}
                
            if new_trans is None:
                continue
                
            if not is_valid_transversal(new_trans, cliques, v_to_cliques):
                continue
                
            new_cost = len(new_trans)
            delta = new_cost - current_cost
            
            if delta <= 0 or rng.random() < pow(2.718, -delta / temp):
                trans = new_trans
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_trans = set(trans)
                    
        return best_trans

    out = []
    start_time = time.time()
    total_budget = 145.0 # Leave 5s margin for 150s limit
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        inst_start = time.time()
        remaining_time = total_budget - (time.time() - start_time)
        
        if remaining_time <= 1.0:
            # Fallback: all vertices
            out.append(list(range(n)))
            continue
            
        # Allocate time for this instance
        # Give roughly equal time to each, but capped by remaining
        time_limit = min(45.0, remaining_time) # Max 45s per instance
        
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            out.append([])
            continue
            
        # Pre-compute vertex to cliques mapping
        v_to_cliques = {v: set() for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                v_to_cliques[v].add(i)
                
        best_set = None
        best_size = n
        
        # 1. Greedy Initialization
        rng_init = random.Random(seed)
        greedy_sol = greedy_set_cover(cliques, n, rng_init, v_to_cliques)
        
        # 2. Local Search (Reduction + 2-for-1)
        ls_sol = local_search(greedy_sol, cliques, rng_init, v_to_cliques, max_steps=500)
        
        if best_set is None or len(ls_sol) < best_size:
            best_set = ls_sol
            best_size = len(ls_sol)
            
        # 3. Simulated Annealing
        # Run SA for the remaining time allocated to this instance
        sa_start = time.time()
        sa_deadline = inst_start + time_limit
        
        # Run multiple SA restarts if time permits
        # Start from the best found so far
        current_best = set(best_set)
        
        while time.time() < sa_deadline:
            rng_sa = random.Random(int(time.time() * 1000) + idx) # Vary seed slightly
            # Limit steps per SA run to avoid getting stuck in one long run if it's bad
            # Estimate steps based on remaining time? 
            # Just do a fixed number of steps, e.g., 1000, and loop.
            sa_sol = simulated_annealing(current_best, cliques, rng_sa, v_to_cliques, start_temp=0.5, end_temp=0.01, steps=1000)
            
            if len(sa_sol) < best_size:
                best_set = sa_sol
                best_size = len(sa_sol)
                current_best = set(sa_sol)
            
            # Small break to check time
            if time.time() >= sa_deadline:
                break
            
        out.append(sorted(best_set) if best_set else list(range(n)))

    return out
