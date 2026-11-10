# EVOLVE-BLOCK-START
import random
import sys

def construct():
    def build_graph(n, p, seed):
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
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            pivot_pool = P | X
            if not pivot_pool:
                return
            pivot = max(pivot_pool, key=lambda v: len(P & adj[v]))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return maximal_cliques

    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            return []
        
        universe = set()
        for clique in cliques:
            universe.update(clique)
            
        # Precompute vertex to cliques mapping
        vertex_to_cliques = {v: set() for v in universe}
        for i, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(i)
        
        # Greedy initialization
        def greedy_init(cliques, universe, vertex_to_cliques, rng):
            uncovered = set(range(len(cliques)))
            selected = set()
            
            while uncovered:
                best_v = None
                best_count = -1
                candidates = []
                
                for v in universe:
                    if v in selected:
                        continue
                    count = len(vertex_to_cliques[v] & uncovered)
                    if count > best_count:
                        best_count = count
                        candidates = [v]
                    elif count == best_count and count > 0:
                        candidates.append(v)
                
                if not candidates or best_count == 0:
                    break
                    
                v = rng.choice(candidates)
                selected.add(v)
                uncovered -= vertex_to_cliques[v]
                
            return selected

        def is_valid_cover(selected, cliques, vertex_to_cliques):
            # Check if every clique is hit
            for i, clique in enumerate(cliques):
                hit = False
                for v in clique:
                    if v in selected:
                        hit = True
                        break
                if not hit:
                    return False
            return True

        def local_search(selected, cliques, vertex_to_cliques, rng, max_iters=2000):
            selected = set(selected)
            current_size = len(selected)
            
            for _ in range(max_iters):
                improved = False
                
                # 1. Try to remove a vertex
                removable = []
                for v in selected:
                    # Check if v is the only one covering any clique
                    necessary = False
                    for i in vertex_to_cliques[v]:
                        clique = cliques[i]
                        # If v is in clique, check if any other selected vertex is in clique
                        other_hit = False
                        for u in clique:
                            if u != v and u in selected:
                                other_hit = True
                                break
                        if not other_hit:
                            necessary = True
                            break
                    if not necessary:
                        removable.append(v)
                        
                if removable:
                    # Remove one (randomly or first)
                    v = rng.choice(removable)
                    selected.remove(v)
                    current_size -= 1
                    improved = True
                    continue
                    
                # 2. If no removal, try swaps to enable future removals
                # Swap u in selected with v not in selected
                # Condition: selected - {u} + {v} is still a valid cover
                # This keeps size same but changes composition.
                # We do this to escape local minima where no single removal works.
                
                # To make this efficient, we only try a few random swaps
                # We pick a random u from selected and a random v from universe-selected
                # and check validity.
                
                # Optimization: Only try swaps if we are stuck for a while?
                # Let's just try a limited number of swaps per iteration
                swaps_tried = 0
                max_swaps = 50
                while swaps_tried < max_swaps:
                    if not selected:
                        break
                    u = rng.choice(list(selected))
                    # Candidate v must cover all cliques that u covers exclusively?
                    # Actually, for the cover to remain valid after removing u and adding v:
                    # Every clique that contains u must contain v OR some other selected vertex.
                    # So, for all cliques C in vertex_to_cliques[u]:
                    #   (v in C) OR (any other w in C is in selected - {u})
                    
                    valid_v_candidates = []
                    # We can restrict v to vertices that appear in many cliques
                    # For now, pick random v from universe
                    v = rng.choice(list(universe))
                    if v in selected:
                        swaps_tried += 1
                        continue
                        
                    # Check validity of swap u -> v
                    valid = True
                    for i in vertex_to_cliques[u]:
                        clique = cliques[i]
                        # Check if v is in clique
                        if v in clique:
                            continue
                        # Check if any other selected vertex (excluding u) is in clique
                        other_hit = False
                        for w in clique:
                            if w != u and w in selected:
                                other_hit = True
                                break
                        if not other_hit:
                            valid = False
                            break
                            
                    if valid:
                        selected.remove(u)
                        selected.add(v)
                        improved = True
                        break
                    else:
                        swaps_tried += 1
                        
                if not improved:
                    # Stuck. Perform a perturbation:
                    # Remove a random vertex, then greedily repair
                    if selected:
                        u = rng.choice(list(selected))
                        selected.remove(u)
                        # Greedy repair
                        uncovered = set()
                        for i, clique in enumerate(cliques):
                            hit = False
                            for w in clique:
                                if w in selected:
                                    hit = True
                                    break
                            if not hit:
                                uncovered.add(i)
                                
                        while uncovered:
                            best_v = None
                            best_count = -1
                            for v in universe:
                                if v in selected:
                                    continue
                                count = len(vertex_to_cliques[v] & uncovered)
                                if count > best_count:
                                    best_count = count
                                    best_v = v
                            if best_v is None:
                                break
                            selected.add(best_v)
                            uncovered -= vertex_to_cliques[best_v]
                            
            return selected

        best_solution = None
        rng = random.Random(seed + 5000)
        
        # Run multiple restarts
        for _ in range(50):
            init_sol = greedy_init(cliques, universe, vertex_to_cliques, rng)
            sol = local_search(init_sol, cliques, vertex_to_cliques, rng, max_iters=500)
            
            if best_solution is None or len(sol) < len(best_solution):
                best_solution = sol
                # If we found a very good solution, maybe increase efforts or just keep going
                # For n=40, optimal is likely small.
                
        return sorted(best_solution)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        solution = solve_instance(n, p, seed)
        out.append(solution)
        
    return out
# EVOLVE-BLOCK-END
