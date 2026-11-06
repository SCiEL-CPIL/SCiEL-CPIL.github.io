import time
import random
from collections import deque
import sys
import math

def _build_graph(n, p, seed):
    """Build adjacency lists from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        U = P | X
        if not U:
            return
        # Pick pivot with max degree in P (heuristic)
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
            
    bk(set(), set(range(n)), set())
    return cliques

def _solve_instance(n, p, seed, time_budget):
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []

    num_cliques = len(cliques)
    
    # Bitmask for all cliques
    all_cliques_mask = (1 << num_cliques) - 1
    
    # Precompute for each vertex, the bitmask of cliques it covers
    vertex_covers = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_covers[v] |= bit

    # Helper to count bits
    def popcount(x):
        return x.bit_count()
        
    # 1. Greedy Initialization with better heuristic
    sol_mask = 0
    clique_counts = [0] * num_cliques
    remaining_cliques = all_cliques_mask
    
    # Precompute vertex degrees for tie-breaking
    vertex_degrees = [len(adj[v]) for v in range(n)]
    
    while remaining_cliques:
        best_v = -1
        best_count = 0
        best_degree = -1
        for v in range(n):
            if (sol_mask & (1 << v)) != 0:
                continue
            count = popcount(vertex_covers[v] & remaining_cliques)
            if count > best_count or (count == best_count and vertex_degrees[v] > best_degree):
                best_count = count
                best_v = v
                best_degree = vertex_degrees[v]
        
        if best_v == -1 or best_count == 0:
            break
            
        sol_mask |= (1 << best_v)
        cover = vertex_covers[best_v]
        for i in range(num_cliques):
            if (cover & (1 << i)):
                clique_counts[i] += 1
                if clique_counts[i] == 1:
                    remaining_cliques &= ~(1 << i)
                    
    cur_sol_mask = sol_mask
    cur_clique_counts = clique_counts[:]
    cur_unhit = popcount(remaining_cliques)
    cur_size = popcount(cur_sol_mask)
    
    best_sol_mask = cur_sol_mask
    best_clique_counts = cur_clique_counts[:]
    best_size = cur_size
    best_unhit = cur_unhit
    
    start_time = time.time()
    
    def try_remove_and_update():
        """Try to remove any single vertex that keeps solution valid. Returns True if improvement made."""
        nonlocal cur_sol_mask, cur_clique_counts, cur_size, cur_unhit, best_sol_mask, best_clique_counts, best_size
        for v in range(n):
            if (cur_sol_mask & (1 << v)) == 0:
                continue
            new_unhit = cur_unhit
            cover = vertex_covers[v]
            c = cover
            while c:
                i = (c & -c).bit_length() - 1
                c &= c - 1
                if cur_clique_counts[i] == 1:
                    new_unhit += 1
            
            if new_unhit == 0:
                # Commit removal
                cur_sol_mask &= ~(1 << v)
                cover = vertex_covers[v]
                c = cover
                while c:
                    i = (c & -c).bit_length() - 1
                    c &= c - 1
                    cur_clique_counts[i] -= 1
                cur_size -= 1
                cur_unhit = new_unhit
                if cur_size < best_size:
                    best_size = cur_size
                    best_sol_mask = cur_sol_mask
                    best_clique_counts = cur_clique_counts[:]
                return True
        return False

    def try_2swap():
        """Try 2-swap: remove 2 vertices, then greedily repair. Try ALL pairs for thoroughness."""
        nonlocal cur_sol_mask, cur_clique_counts, cur_size, cur_unhit, best_sol_mask, best_clique_counts, best_size
        if cur_unhit != 0:
            return False
        
        vertices_in_sol = [v for v in range(n) if (cur_sol_mask & (1 << v))]
        if len(vertices_in_sol) < 2:
            return False
        
        # Try ALL pairs to remove, then repair
        for i_idx in range(len(vertices_in_sol)):
            if time.time() - start_time > time_budget:
                return False
            for j_idx in range(i_idx + 1, len(vertices_in_sol)):
                if time.time() - start_time > time_budget:
                    return False
                
                u = vertices_in_sol[i_idx]
                v = vertices_in_sol[j_idx]
                
                # Simulate removal of u and v
                trial_sol = cur_sol_mask & ~(1 << u) & ~(1 << v)
                trial_counts = cur_clique_counts[:]
                
                cover_u = vertex_covers[u]
                c = cover_u
                while c:
                    i = (c & -c).bit_length() - 1
                    c &= c - 1
                    trial_counts[i] -= 1
                    
                cover_v = vertex_covers[v]
                c = cover_v
                while c:
                    i = (c & -c).bit_length() - 1
                    c &= c - 1
                    trial_counts[i] -= 1
                    
                # Find unhit cliques
                remaining_mask = 0
                for i in range(num_cliques):
                    if trial_counts[i] == 0:
                        remaining_mask |= (1 << i)
                        
                if remaining_mask == 0:
                    # Both can be removed! Net improvement of -2
                    cur_sol_mask = trial_sol
                    cur_clique_counts = trial_counts
                    cur_size -= 2
                    cur_unhit = 0
                    if cur_size < best_size:
                        best_size = cur_size
                        best_sol_mask = cur_sol_mask
                        best_clique_counts = cur_clique_counts[:]
                    return True
                    
                # Greedily add back vertices to cover remaining cliques
                # Try to add at most 2 vertices (net 0 or better)
                added = 0
                while remaining_mask and added < 2 and time.time() - start_time < time_budget:
                    best_v = -1
                    best_count = 0
                    for w in range(n):
                        if (trial_sol & (1 << w)) != 0:
                            continue
                        count = popcount(vertex_covers[w] & remaining_mask)
                        if count > best_count:
                            best_count = count
                            best_v = w
                    if best_v == -1 or best_count == 0:
                        break
                    trial_sol |= (1 << best_v)
                    cover = vertex_covers[best_v]
                    c = cover
                    while c:
                        i = (c & -c).bit_length() - 1
                        c &= c - 1
                        trial_counts[i] += 1
                        if trial_counts[i] == 1:
                            remaining_mask &= ~(1 << i)
                    added += 1
                    
                trial_unhit = popcount(remaining_mask)
                trial_size = popcount(trial_sol)
                
                if trial_unhit == 0 and trial_size < cur_size:
                    cur_sol_mask = trial_sol
                    cur_clique_counts = trial_counts
                    cur_size = trial_size
                    cur_unhit = 0
                    if cur_size < best_size:
                        best_size = cur_size
                        best_sol_mask = cur_sol_mask
                        best_clique_counts = cur_clique_counts[:]
                    return True
        return False

    def kick_and_repair():
        """Perturb by removing k random vertices, then repair with greedy."""
        nonlocal cur_sol_mask, cur_clique_counts, cur_size, cur_unhit, best_sol_mask, best_clique_counts, best_size
        if cur_unhit != 0:
            return False
        
        vertices_in_sol = [v for v in range(n) if (cur_sol_mask & (1 << v))]
        if len(vertices_in_sol) < 3:
            return False
            
        k = random.randint(2, min(5, len(vertices_in_sol) - 1))
        to_remove = random.sample(vertices_in_sol, k)
        
        trial_sol = cur_sol_mask
        trial_counts = cur_clique_counts[:]
        for v in to_remove:
            trial_sol &= ~(1 << v)
            cover = vertex_covers[v]
            c = cover
            while c:
                i = (c & -c).bit_length() - 1
                c &= c - 1
                trial_counts[i] -= 1
                
        remaining_mask = 0
        for i in range(num_cliques):
            if trial_counts[i] == 0:
                remaining_mask |= (1 << i)
                
        while remaining_mask and time.time() - start_time < time_budget:
            best_v = -1
            best_count = 0
            for v in range(n):
                if (trial_sol & (1 << v)) != 0:
                    continue
                count = popcount(vertex_covers[v] & remaining_mask)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            trial_sol |= (1 << best_v)
            cover = vertex_covers[best_v]
            c = cover
            while c:
                i = (c & -c).bit_length() - 1
                c &= c - 1
                trial_counts[i] += 1
                if trial_counts[i] == 1:
                    remaining_mask &= ~(1 << i)
                    
        trial_unhit = popcount(remaining_mask)
        trial_size = popcount(trial_sol)
        
        if trial_unhit == 0 and trial_size < cur_size:
            cur_sol_mask = trial_sol
            cur_clique_counts = trial_counts
            cur_size = trial_size
            cur_unhit = 0
            if cur_size < best_size:
                best_size = cur_size
                best_sol_mask = cur_sol_mask
                best_clique_counts = cur_clique_counts[:]
            return True
        return False

    def final_shrink():
        """Final polish: try to remove vertices in different orders to find smaller solutions."""
        nonlocal cur_sol_mask, cur_clique_counts, cur_size, cur_unhit, best_sol_mask, best_clique_counts, best_size
        if cur_unhit != 0:
            return False
        
        vertices_in_sol = [v for v in range(n) if (cur_sol_mask & (1 << v))]
        if len(vertices_in_sol) < 2:
            return False
        
        # Try removing vertices in random order, checking validity each time
        for _ in range(30):  # More attempts for better polish
            if time.time() - start_time > time_budget:
                break
            trial_sol = cur_sol_mask
            trial_counts = cur_clique_counts[:]
            trial_size = cur_size
            trial_unhit = cur_unhit
            
            order = vertices_in_sol[:]
            random.shuffle(order)
            
            for v in order:
                if (trial_sol & (1 << v)) == 0:
                    continue
                # Check if we can remove v
                new_unhit = trial_unhit
                cover = vertex_covers[v]
                c = cover
                while c:
                    i = (c & -c).bit_length() - 1
                    c &= c - 1
                    if trial_counts[i] == 1:
                        new_unhit += 1
                
                if new_unhit == 0:
                    trial_sol &= ~(1 << v)
                    cover = vertex_covers[v]
                    c = cover
                    while c:
                        i = (c & -c).bit_length() - 1
                        c &= c - 1
                        trial_counts[i] -= 1
                    trial_size -= 1
                    trial_unhit = new_unhit
            
            if trial_unhit == 0 and trial_size < best_size:
                best_size = trial_size
                best_sol_mask = trial_sol
                best_clique_counts = trial_counts[:]
                cur_sol_mask = trial_sol
                cur_clique_counts = trial_counts[:]
                cur_size = trial_size
                cur_unhit = trial_unhit
                return True
        return False

    def exact_local_polish():
        """High-precision final pass: systematically try all single removals in multiple passes,
        then try 1-in-1-out swaps to find better solutions."""
        nonlocal cur_sol_mask, cur_clique_counts, cur_size, cur_unhit, best_sol_mask, best_clique_counts, best_size
        if cur_unhit != 0:
            return False
        
        improved = True
        passes = 0
        max_passes = 10
        while improved and passes < max_passes and time.time() - start_time < time_budget:
            improved = False
            passes += 1
            
            # Pass 1: Try removing each vertex
            for v in range(n):
                if (cur_sol_mask & (1 << v)) == 0:
                    continue
                new_unhit = cur_unhit
                cover = vertex_covers[v]
                c = cover
                while c:
                    i = (c & -c).bit_length() - 1
                    c &= c - 1
                    if cur_clique_counts[i] == 1:
                        new_unhit += 1
                
                if new_unhit == 0:
                    cur_sol_mask &= ~(1 << v)
                    cover = vertex_covers[v]
                    c = cover
                    while c:
                        i = (c & -c).bit_length() - 1
                        c &= c - 1
                        cur_clique_counts[i] -= 1
                    cur_size -= 1
                    cur_unhit = new_unhit
                    improved = True
                    if cur_size < best_size:
                        best_size = cur_size
                        best_sol_mask = cur_sol_mask
                        best_clique_counts = cur_clique_counts[:]
            
            # Pass 2: Try 1-in-1-out swaps
            vertices_in_sol = [v for v in range(n) if (cur_sol_mask & (1 << v))]
            for v_out in vertices_in_sol:
                if time.time() - start_time > time_budget:
                    break
                # Temporarily remove v_out
                trial_sol = cur_sol_mask & ~(1 << v_out)
                trial_counts = cur_clique_counts[:]
                cover_out = vertex_covers[v_out]
                c = cover_out
                while c:
                    i = (c & -c).bit_length() - 1
                    c &= c - 1
                    trial_counts[i] -= 1
                
                remaining_mask = 0
                for i in range(num_cliques):
                    if trial_counts[i] == 0:
                        remaining_mask |= (1 << i)
                
                if remaining_mask == 0:
                    # v_out can be removed entirely
                    cur_sol_mask = trial_sol
                    cur_clique_counts = trial_counts
                    cur_size -= 1
                    cur_unhit = 0
                    improved = True
                    if cur_size < best_size:
                        best_size = cur_size
                        best_sol_mask = cur_sol_mask
                        best_clique_counts = cur_clique_counts[:]
                    continue
                
                # Try adding one vertex to cover remaining
                for v_in in range(n):
                    if (cur_sol_mask & (1 << v_in)) != 0:
                        continue
                    cover_in = vertex_covers[v_in]
                    if (cover_in & remaining_mask) != remaining_mask:
                        continue
                    # This swap works: remove v_out, add v_in
                    # But we need net improvement, so only accept if it enables further removals
                    # For now, just check if it's valid
                    new_trial_sol = trial_sol | (1 << v_in)
                    new_trial_counts = trial_counts[:]
                    c = cover_in
                    while c:
                        i = (c & -c).bit_length() - 1
                        c &= c - 1
                        new_trial_counts[i] += 1
                    
                    # Check if this new solution allows removing some other vertex
                    for v_check in range(n):
                        if (new_trial_sol & (1 << v_check)) == 0:
                            continue
                        new_unhit = 0
                        cover_check = vertex_covers[v_check]
                        c = cover_check
                        while c:
                            i = (c & -c).bit_length() - 1
                            c &= c - 1
                            if new_trial_counts[i] == 1:
                                new_unhit += 1
                        if new_unhit == 0:
                            # We can do a 2-for-1 improvement
                            cur_sol_mask = new_trial_sol & ~(1 << v_check)
                            cur_clique_counts = new_trial_counts[:]
                            cover_check = vertex_covers[v_check]
                            c = cover_check
                            while c:
                                i = (c & -c).bit_length() - 1
                                c &= c - 1
                                cur_clique_counts[i] -= 1
                            cur_size -= 1
                            cur_unhit = 0
                            improved = True
                            if cur_size < best_size:
                                best_size = cur_size
                                best_sol_mask = cur_sol_mask
                                best_clique_counts = cur_clique_counts[:]
                            break
                    if improved:
                        break
        return improved

    # Main local search loop
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        
        # Phase 1: Try single removals
        if try_remove_and_update():
            improved = True
            continue
            
        # Phase 2: Try 2-swap moves
        if try_2swap():
            improved = True
            continue
            
        # Phase 3: Kick and repair
        if kick_and_repair():
            improved = True
            continue
        
        # Phase 4: Final shrink (polish)
        if final_shrink():
            improved = True
            continue
        
        # Phase 5: Exact local polish
        if exact_local_polish():
            improved = True
            continue

    result = [v for v in range(n) if (best_sol_mask & (1 << v))]
    return result

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate time: give more to larger/harder instances
    # n=30,p=0.3: denser, more cliques, harder
    # n=35,p=0.25: medium
    # n=40,p=0.2: sparser but more vertices
    time_allocs = [45, 40, 45]  # Total 130s, leaving 10s buffer
    
    for i, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time()
        if remaining_time < 8:
             # Fallback: quick greedy
             adj = _build_graph(n, p, seed)
             cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
             if not cliques:
                 results.append([])
                 continue
             sol = []
             for clique in cliques:
                 if clique & set(sol):
                     continue
                 v = min(clique)  # Pick smallest vertex for determinism
                 sol.append(v)
             results.append(sorted(set(sol)))
             continue
         
        alloc = min(time_allocs[i], remaining_time - 5)
        
        sol = _solve_instance(n, p, seed, alloc)
        results.append(sol)
        
    return results
