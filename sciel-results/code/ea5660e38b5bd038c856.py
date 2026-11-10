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
            # Pivot selection
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

    def is_valid_transversal(trans, cliques):
        for c in cliques:
            hit = False
            for v in c:
                if v in trans:
                    hit = True
                    break
            if not hit:
                return False
        return True

    def local_search(trans, cliques, rng, max_steps=500):
        trans = set(trans)
        n = max(max(v for c in cliques for v in c) + 1, 1)
        all_vertices = set(range(n))
        
        # First, try to remove redundant vertices
        changed = True
        while changed:
            changed = False
            for v in list(trans):
                new_trans = trans - {v}
                if is_valid_transversal(new_trans, cliques):
                    trans = new_trans
                    changed = True
        
        # Then, try to swap vertices to reduce size
        for _ in range(max_steps):
            # Try removing a random vertex
            if not trans:
                break
            v = rng.choice(list(trans))
            new_trans = trans - {v}
            if is_valid_transversal(new_trans, cliques):
                trans = new_trans
                continue
            
            # Try swapping v with a random other vertex
            others = all_vertices - trans
            if not others:
                continue
            w = rng.choice(list(others))
            new_trans = (trans - {v}) | {w}
            if is_valid_transversal(new_trans, cliques):
                trans = new_trans
                continue
                
        return trans

    def greedy_set_cover(cliques, n, rng):
        uncovered = set(range(len(cliques)))
        selected = set()
        
        # Pre-compute incidence: for each vertex, which cliques it hits
        # cliques is a list of tuples
        # Let's use indices for cliques
        cliques_list = list(cliques)
        
        # Map vertex -> set of clique indices
        v_to_cliques = {v: set() for v in range(n)}
        for i, c in enumerate(cliques_list):
            for v in c:
                v_to_cliques[v].add(i)
                
        while uncovered:
            best_v = -1
            best_count = -1
            ties = []
            for v in range(n):
                # Count how many uncovered cliques v hits
                count = len(v_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
                    ties = [v]
                elif count == best_count:
                    ties.append(v)
            
            if best_count == 0:
                break # Should not happen if uncovered is not empty
                
            if len(ties) > 1:
                best_v = rng.choice(ties)
                
            selected.add(best_v)
            
            # Remove covered cliques
            uncovered -= v_to_cliques[best_v]
            
        return selected

    def simulated_annealing(trans, cliques, rng, start_temp=1.0, end_temp=0.1, steps=1000):
        trans = set(trans)
        current_cost = len(trans)
        best_trans = set(trans)
        best_cost = current_cost
        
        n = max(max(v for c in cliques for v in c) + 1, 1)
        all_vertices = set(range(n))
        
        for i in range(steps):
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            
            # Generate a neighbor
            # Option 1: Remove a random vertex
            # Option 2: Add a random vertex
            # Option 3: Swap a random vertex
            r = rng.random()
            if r < 0.3 and trans:
                v = rng.choice(list(trans))
                new_trans = trans - {v}
            elif r < 0.6 and all_vertices - trans:
                v = rng.choice(list(all_vertices - trans))
                new_trans = trans | {v}
            else:
                if not trans:
                    continue
                v = rng.choice(list(trans))
                others = all_vertices - trans
                if not others:
                    continue
                w = rng.choice(list(others))
                new_trans = (trans - {v}) | {w}
                
            if not is_valid_transversal(new_trans, cliques):
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
    total_budget = 140.0 # Leave 10s margin for 150s limit
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        inst_start = time.time()
        # Check if we have enough time for this instance
        remaining_time = total_budget - (time.time() - start_time)
        if remaining_time <= 0:
            # Fallback: return best possible (empty or all vertices) immediately
            # But we need to generate the graph at least? No, we can just return [] if we know it's invalid?
            # No, we must return a valid transversal.
            # If we run out of time, we should return a known valid transversal.
            # The set of all vertices is always a valid transversal.
            # So we can just return list(range(n)) as a fallback.
            out.append(list(range(n)))
            continue
            
        # Allocate time for this instance
        # If it's the first instance, give it more time?
        # Let's just use a fixed fraction of remaining time.
        # But we need to be careful.
        # Let's say we have T remaining, and K instances left.
        # Give T/K to each?
        # But some instances might be harder.
        # Let's just give 35s per instance as before, but capped by remaining time.
        time_limit = min(35.0, remaining_time)
        
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            out.append([])
            continue
            
        best_set = None
        best_size = n # Upper bound
        
        # Initial greedy solution
        rng_init = random.Random(seed)
        greedy_sol = greedy_set_cover(cliques, n, rng_init)
        greedy_sol = local_search(greedy_sol, cliques, rng_init)
        
        if len(greedy_sol) < best_size:
            best_set = greedy_sol
            best_size = len(greedy_sol)
        
        # Simulated Annealing
        remaining_for_sa = time_limit - (time.time() - inst_start)
        if remaining_for_sa > 5:
            rng_sa = random.Random(seed + 100)
            sa_sol = simulated_annealing(greedy_sol, cliques, rng_sa, steps=500)
            if len(sa_sol) < best_size:
                best_set = sa_sol
                best_size = len(sa_sol)
        
        # More local search iterations if time permits
        remaining_for_ls = time_limit - (time.time() - inst_start)
        if remaining_for_ls > 2 and best_set:
            rng_ls = random.Random(seed + 200)
            ls_sol = local_search(best_set, cliques, rng_ls, max_steps=200)
            if len(ls_sol) < best_size:
                best_set = ls_sol
                best_size = len(ls_sol)
        
        if best_set is None:
            best_set = set(range(n))
            
        out.append(sorted(best_set))
        
    return out
