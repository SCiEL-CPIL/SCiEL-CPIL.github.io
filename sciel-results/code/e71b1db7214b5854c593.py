import time
import random
from itertools import combinations

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        if P or X:
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

    def greedy_initial_solution(cliques, n):
        """Greedy set cover initialization."""
        uncovered = list(range(len(cliques)))
        selected = set()
        
        # Precompute vertex to clique indices
        vertex_to_cliques = {v: set() for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].add(i)
        
        while uncovered:
            # Find vertex covering most uncovered cliques
            best_v = -1
            best_count = 0
            # Only consider vertices that appear in some uncovered clique
            candidates = set()
            for i in uncovered:
                candidates.update(cliques[i])
            
            for v in candidates:
                count = sum(1 for i in uncovered if v in cliques[i])
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1:
                break
                
            selected.add(best_v)
            # Remove covered cliques
            covered_by_best = vertex_to_cliques[best_v]
            uncovered = [i for i in uncovered if i not in covered_by_best]
            
        return selected

    def is_minimal(selected, cliques):
        """Check if selected is an inclusion-minimal transversal."""
        # For each vertex in selected, check if it's essential
        # A vertex v is essential if there exists a clique C such that 
        # v is the ONLY vertex in C that is in selected.
        
        # Optimization: Precompute for each clique, which vertices are in selected
        # Actually, we can just iterate.
        
        # To speed up, we can check: for each v in selected, is there a clique 
        # where selected & clique == {v}?
        
        # Let's create a map: clique_index -> list of vertices in selected
        # But n is small.
        
        for v in selected:
            # Check if v is the only one in selected for any clique
            for c in cliques:
                if v in c:
                    # Check if any other vertex in c is in selected
                    other_in_selected = False
                    for u in c:
                        if u != v and u in selected:
                            other_in_selected = True
                            break
                    if not other_in_selected:
                        return False, v
        return True, None

    def reduce_to_minimal(selected, cliques):
        """Remove all non-essential vertices."""
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                # Check if v is essential
                is_essential = False
                for c in cliques:
                    if v in c:
                        # Check if any other vertex in c is in selected
                        other_in_selected = False
                        for u in c:
                            if u != v and u in selected:
                                other_in_selected = True
                                break
                        if not other_in_selected:
                            is_essential = True
                            break
                
                if not is_essential:
                    selected.remove(v)
                    changed = True
        return selected

    def local_search_swap(selected, cliques, n, rng, max_iters=100):
        """Try to improve size via swaps (remove 1, add 1) if possible, 
        though usually we just want to minimize size. 
        Actually, swaps keep size same. We want to REDUCE size.
        The reduction happens in reduce_to_minimal.
        However, a swap might lead to a state where a subsequent removal is possible.
        So: Swap -> Reduce -> if size decreased, accept.
        """
        current_size = len(selected)
        for _ in range(max_iters):
            if not selected:
                break
            v_out = rng.choice(list(selected))
            # Pick a random v_in not in selected
            # To be efficient, pick from vertices that are in some clique
            # But just picking random 0..n-1 is fine for small n
            v_in = rng.randint(0, n - 1)
            if v_in == v_out or v_in in selected:
                continue
            
            candidate = (selected - {v_out}) | {v_in}
            
            # Try to reduce candidate
            candidate = reduce_to_minimal(candidate, cliques)
            
            if len(candidate) < current_size:
                selected = candidate
                current_size = len(selected)
            # If size is same, we might accept with probability to diversify, 
            # but since we only care about final best, we can stick to improving or neutral.
            # Let's accept neutral swaps with low probability to escape local minima of "minimal" sets.
            elif len(candidate) == current_size and rng.random() < 0.1:
                selected = candidate
        return selected

    def solve_instance(n, p, seed, time_budget):
        start_time = time.time()
        edges, adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            return []
        
        # Initial solution
        best_sol = greedy_initial_solution(cliques, n)
        best_sol = reduce_to_minimal(best_sol, cliques)
        best_size = len(best_sol)
        
        rng = random.Random(seed + 12345)
        
        # Iterated Local Search
        current_sol = set(best_sol)
        
        while time.time() - start_time < time_budget:
            # Perturb: remove k random vertices and re-add greedily
            k = rng.randint(1, max(1, len(current_sol) // 3))
            if k == 0:
                k = 1
            
            # Remove k vertices
            to_remove = rng.sample(list(current_sol), min(k, len(current_sol)))
            perturbed = set(current_sol)
            for v in to_remove:
                perturbed.discard(v)
            
            # Re-add greedily to cover uncovered cliques
            uncovered = []
            for i, c in enumerate(cliques):
                if not any(v in perturbed for v in c):
                    uncovered.append(i)
            
            if uncovered:
                # Greedy add
                # Precompute vertex to uncovered clique indices for speed
                v_to_uncov = {v: set() for v in range(n)}
                for i in uncovered:
                    for v in cliques[i]:
                        v_to_uncov[v].add(i)
                
                while uncovered:
                    best_v = -1
                    best_count = 0
                    for v in range(n):
                        if v in perturbed:
                            continue
                        count = len(v_to_uncov[v] & set(uncovered))
                        if count > best_count:
                            best_count = count
                            best_v = v
                    if best_v == -1:
                        break
                    perturbed.add(best_v)
                    # Update uncovered
                    covered_by_best = v_to_uncov[best_v]
                    uncovered = [i for i in uncovered if i not in covered_by_best]
            
            # Reduce to minimal
            perturbed = reduce_to_minimal(perturbed, cliques)
            
            # Local Search (Swaps)
            # Limit iterations based on time
            remaining = time_budget - (time.time() - start_time)
            if remaining > 0.01:
                # Simple swap search
                max_ls_iters = int(remaining * 50) # Scale with time
                if max_ls_iters > 0:
                    perturbed = local_search_swap(perturbed, cliques, n, rng, max_iters=max_ls_iters)
            
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_sol = perturbed
                current_sol = set(best_sol)
            else:
                # Accept with probability if equal or worse? 
                # Usually ILS accepts worse with probability exp(-delta/T).
                # Here, let's just accept equal to diversify, or reject worse.
                if len(perturbed) == best_size and rng.random() < 0.5:
                    current_sol = perturbed
                else:
                    current_sol = set(best_sol) # Reset to best to stay in basin? 
                    # Or keep current to allow drift?
                    # Let's keep current_sol as the perturbed one to allow drift, 
                    # but track best_sol separately.
                    # Actually, standard ILS keeps current.
                    pass 

        return sorted(best_sol)

    out = []
    start_time = time.time()
    total_deadline = 145.0  # Leave 5s margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = total_deadline - elapsed
        if remaining <= 0:
            # Fallback: just return something valid? 
            # If we run out of time, we must return a valid solution.
            # The greedy solution from the previous step (if any) or a trivial one.
            # Since we solve sequentially, if we run out, we haven't solved this one.
            # We should allocate time carefully.
            # Let's ensure each instance gets at least 10s if possible.
            inst_budget = max(1.0, remaining / (3 - idx))
        else:
            inst_budget = remaining / (3 - idx)
        
        # Add a small buffer
        inst_budget = min(inst_budget, 45.0) # Cap per instance
        
        sol = solve_instance(n, p, seed, inst_budget)
        out.append(sol)
        
        # Check if we are running out of time for the next ones
        if time.time() - start_time > total_deadline - 5.0:
             # Emergency: if we don't have enough time for remaining instances,
             # we need to return valid solutions for them quickly.
             # The solve_instance function handles time, but if we call it with 0 budget, it might fail.
             # Let's force a quick greedy for remaining if time is tight.
             pass

    # Final check: ensure we returned 3 lists
    while len(out) < 3:
        out.append([]) # This is invalid, but better than crash? 
        # Actually, if we run out of time, we should have allocated enough.
        # Let's trust the budgeting.
        
    return out

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    for i, r in enumerate(res):
        print(f"Instance {i}: size {len(r)}")
