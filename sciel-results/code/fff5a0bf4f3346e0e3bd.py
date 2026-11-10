import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, P, X, C, maximal_cliques):
        if not P and not X:
            if len(C) >= 2:
                maximal_cliques.append(frozenset(C))
            return
        
        P_union_X = P | X
        if P_union_X:
            # Pivot selection: choose vertex u in P union X with max degree in P
            pivot = None
            max_inter = -1
            for v in P_union_X:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    pivot = v
            
            # Branch on P \ N(pivot)
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_P = P & adj[v]
                new_X = X & adj[v]
                new_C = C | {v}
                bron_kerbosch(adj, new_P, new_X, new_C, maximal_cliques)
                P.remove(v)
                X.add(v)

    def find_min_transversal(n, edges):
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        maximal_cliques = []
        # Bron-Kerbosch
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)
        
        if not maximal_cliques:
            return []

        # Convert cliques to lists of integers for faster indexing if needed, 
        # but frozensets are fine for set operations.
        # We will use a branch-and-bound approach for the hitting set problem.
        
        # Precompute: for each vertex, which cliques does it hit?
        # This allows quick updates.
        # However, with small n and moderate clique count, we can just use sets.
        
        unhit = list(maximal_cliques)
        transversal = set()
        
        # 1. Greedy initialization
        while unhit:
            if time.time() > deadline:
                break
            freq = {}
            for clique in unhit:
                for v in clique:
                    freq[v] = freq.get(v, 0) + 1
            
            if not freq:
                break
                
            best_v = None
            best_score = None
            for v, f in freq.items():
                # Score: frequency is primary. Degree is secondary.
                score = (f, len(adj[v]), -v)
                if best_score is None or score > best_score:
                    best_score = score
                    best_v = v
            
            transversal.add(best_v)
            
            # Remove all cliques containing best_v
            new_unhit = []
            for clique in unhit:
                if best_v not in clique:
                    new_unhit.append(clique)
            unhit = new_unhit

        # Ensure validity if broken by time
        if unhit:
            for c in unhit:
                transversal.add(next(iter(c)))
        
        # 2. Branch and Bound Refinement
        # We want to minimize |transversal|.
        # We can try to replace subsets of the current transversal with smaller sets.
        # A simpler B&B: 
        # Sort vertices in transversal. 
        # Try to remove a vertex v. If the remaining set still covers all cliques, keep it removed.
        # This is the "greedy removal" which is local search.
        
        # To go further, we can do a limited depth-first search for swaps.
        # Given the small size (n<=40, transversal likely < 15), a full B&B might be feasible if pruned well.
        
        # Let's implement a more aggressive local search / B&B hybrid.
        # We'll try to find a smaller set by iteratively trying to remove vertices.
        
        # First, standard greedy removal (already present in previous versions, let's keep it)
        changed = True
        while changed:
            changed = False
            for v in list(transversal):
                if time.time() > deadline:
                    return sorted(transversal)
                redundant = True
                for c in maximal_cliques:
                    if v in c:
                        others = c - {v}
                        if not (others & transversal):
                            redundant = False
                            break
                if redundant:
                    transversal.remove(v)
                    changed = True
        
        # Now, try to improve by swapping or replacing.
        # A simple strategy: 
        # For each vertex v in transversal, try removing it.
        # If the set becomes invalid, try to add ONE other vertex w to restore validity.
        # If |transversal \ {v} | {w}| < |transversal|, we'd have improved, but size is same.
        # We want size reduction.
        
        # Let's try a depth-limited search for a smaller hitting set.
        # Current size K. Can we find a hitting set of size K-1?
        
        # This is NP-hard, but K is small.
        # We can use a recursive search:
        # search(remaining_cliques, current_set, limit)
        
        # To make it efficient, we only search for a solution of size < current_size.
        # We'll start by trying to find a solution of size current_size - 1.
        
        # Actually, a simpler and effective heuristic for small instances:
        # Randomized local search with restarts?
        # Or just exhaustive search for size reduction?
        
        # Let's try to find if a smaller set exists by checking subsets of the current transversal?
        # No, the optimal set might contain vertices NOT in the current greedy transversal.
        
        # However, for these specific random graphs, the greedy solution is often very close to optimal.
        # The main gain from B&B is usually removing a few redundant vertices that greedy removal missed due to ordering, or finding a completely different smaller structure.
        
        # Let's implement a proper Branch and Bound for the Minimum Hitting Set.
        # State: (unhit_cliques, chosen_vertices)
        # We want to find a hitting set of size <= current_best - 1.
        
        # To avoid exponential blowup, we limit the search depth or time.
        
        best_size = len(transversal)
        
        # We will try to find a set of size best_size - 1, then best_size - 2, etc.
        # This is expensive. 
        
        # Alternative: Use the greedy solution as a starting point and perform local moves.
        # Move 1: Remove v if possible (done above).
        # Move 2: Swap v for w? (Size stays same, might allow future removals).
        # Move 3: Replace {v1, v2} with {w}? (Size decreases by 1).
        
        # Let's try Move 3: 
        # For every pair of vertices (v1, v2) in transversal:
        #   Remove v1, v2.
        #   Find cliques that are now unhit.
        #   Find a single vertex w that hits ALL those unhit cliques.
        #   If such w exists, and w is not v1 or v2, replace {v1, v2} with {w}.
        #   This reduces size by 1.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            trans_list = list(transversal)
            for i in range(len(trans_list)):
                for j in range(i + 1, len(trans_list)):
                    v1 = trans_list[i]
                    v2 = trans_list[j]
                    
                    # Candidate set: transversal without v1, v2
                    candidate = transversal - {v1, v2}
                    
                    # Find cliques not hit by candidate
                    unhit_after_removal = []
                    for c in maximal_cliques:
                        if not (c & candidate):
                            unhit_after_removal.append(c)
                    
                    if not unhit_after_removal:
                        # We removed v1 and v2 and everything is still hit?
                        # That means v1 and v2 were redundant. Greedy removal should have caught this, 
                        # unless it was a combined redundancy.
                        transversal = candidate
                        improved = True
                        break
                    
                    # Check if a single vertex w can cover all unhit_after_removal
                    # w must be in the union of these cliques.
                    # Also w should ideally be in the original graph vertices.
                    
                    potential_w = set()
                    for c in unhit_after_removal:
                        potential_w.update(c)
                    
                    # Filter potential_w: 
                    # 1. w must hit ALL cliques in unhit_after_removal.
                    # 2. w should not be v1 or v2 (otherwise we just removed them and added one back, net -1).
                    #    Actually if w == v1, we just removed v2. That's a single removal.
                    #    If we can remove v2 alone, greedy removal would have found it.
                    #    So we look for w != v1, w != v2.
                    
                    found_w = None
                    for w in potential_w:
                        if w == v1 or w == v2:
                            continue
                        # Check if w hits all unhit_after_removal
                        ok = True
                        for c in unhit_after_removal:
                            if w not in c:
                                ok = False
                                break
                        if ok:
                            # Also check: does adding w cause any issues? 
                            # We are replacing {v1, v2} with {w}.
                            # The new set is candidate U {w}.
                            # We know candidate hits all cliques EXCEPT unhit_after_removal.
                            # w hits all unhit_after_removal.
                            # So candidate U {w} hits all cliques.
                            # Size is len(transversal) - 2 + 1 = len(transversal) - 1.
                            found_w = w
                            break
                    
                    if found_w is not None:
                        transversal = candidate | {found_w}
                        improved = True
                        break
                
                if improved:
                    break
        
        # One more pass of greedy removal after swaps
        changed = True
        while changed:
            changed = False
            for v in list(transversal):
                if time.time() > deadline:
                    break
                redundant = True
                for c in maximal_cliques:
                    if v in c:
                        others = c - {v}
                        if not (others & transversal):
                            redundant = False
                            break
                if redundant:
                    transversal.remove(v)
                    changed = True

        return sorted(transversal)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        sol = find_min_transversal(n, edges)
        results.append(sol)
        if time.time() > deadline:
             # If we ran out of time, ensure we return what we have
             # But we need to return 3 lists. 
             # If we didn't finish all 3, we should return best effort or empty?
             # Better to return valid sets for completed ones and try fast heuristic for others?
             # Actually, 120s is plenty for N=40. The loop is very fast.
             pass

    return results
