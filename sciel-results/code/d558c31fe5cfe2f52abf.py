import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))  # 448
    
    # Generate all edges
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    m = len(edges)  # 448
    
    # Precompute all C4s (squares)
    # A square is defined by two distinct dimensions i, j and a base vertex x
    # Vertices: x, x^i, x^j, x^i^j
    # Edges: (x, x^i), (x^i, x^i^j), (x^i^j, x^j), (x^j, x)
    squares = []
    edge_to_squares = {e: [] for e in edges}
    processed_squares = set()
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                sq_edges = [e1, e2, e3, e4]
                sq_key = frozenset(sq_edges)
                
                if sq_key in processed_squares:
                    continue
                processed_squares.add(sq_key)
                
                sq_idx = len(squares)
                squares.append(sq_edges)
                for e in sq_edges:
                    edge_to_squares[e].append(sq_idx)
    
    # Build conflict graph: two edges conflict if they share a square
    # Actually, for the local search, we need to know which edges are "blocked"
    # by the current selection.
    
    # 1. Greedy Construction with random ordering
    def greedy_construct():
        selected = set()
        edge_order = list(edges)
        random.shuffle(edge_order)
        
        # Track for each square how many edges are selected
        sq_count = [0] * len(squares)
        
        for e in edge_order:
            ok = True
            for sq_idx in edge_to_squares[e]:
                if sq_count[sq_idx] == 3:
                    ok = False
                    break
            if ok:
                selected.add(e)
                for sq_idx in edge_to_squares[e]:
                    sq_count[sq_idx] += 1
        return selected
    
    # 2. Local Search: remove one edge, add multiple edges
    def local_search(selected, time_limit):
        # Precompute for each edge, the list of other edges in its squares
        # This is for fast candidate generation
        edge_conflicts = {}
        for e in edges:
            conflicts = set()
            for sq_idx in edge_to_squares[e]:
                for se in squares[sq_idx]:
                    if se != e:
                        conflicts.add(se)
            edge_conflicts[e] = conflicts
        
        # Track square counts for fast checking
        sq_count = [0] * len(squares)
        for e in selected:
            for sq_idx in edge_to_squares[e]:
                sq_count[sq_idx] += 1
        
        def can_add(e):
            for sq_idx in edge_to_squares[e]:
                if sq_count[sq_idx] == 3:
                    return False
            return True
        
        def add_edge(e):
            selected.add(e)
            for sq_idx in edge_to_squares[e]:
                sq_count[sq_idx] += 1
        
        def remove_edge(e):
            selected.remove(e)
            for sq_idx in edge_to_squares[e]:
                sq_count[sq_idx] -= 1
        
        edge_list = list(edges)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(edge_list)
            
            for e_out in edge_list:
                if time.time() > time_limit:
                    break
                if e_out not in selected:
                    continue
                
                # Remove e_out
                remove_edge(e_out)
                
                # Get candidates: edges that conflict with e_out and are not selected
                candidates = [c for c in edge_conflicts[e_out] if c not in selected]
                random.shuffle(candidates)
                
                added_count = 0
                for c in candidates:
                    if can_add(c):
                        add_edge(c)
                        added_count += 1
                
                if added_count > 1:
                    improved = True
                elif added_count == 1:
                    # Neutral move - accept with some probability to escape local optima
                    if random.random() < 0.3:
                        improved = True
                    else:
                        # Revert
                        # We added one edge, need to remove it and re-add e_out
                        # But we don't know which one was added. Let's just revert by
                        # removing the last added and re-adding e_out.
                        # Actually, this is tricky. Let's just not do neutral moves
                        # unless we track them properly.
                        # For simplicity, revert:
                        # We need to undo the add. Let's track it.
                        # Actually, let's just not accept neutral moves for now.
                        # Revert:
                        # Find the edge we added and remove it
                        # This is inefficient. Let's just skip neutral moves.
                        # Revert by re-adding e_out and removing the added edge
                        # But we don't know which edge was added.
                        # Let's just not do this.
                        # Actually, let's just accept neutral moves always to explore.
                        improved = True
                else:
                    # No improvement, revert
                    add_edge(e_out)
        
        return selected
    
    # 3. Simulated Annealing for escaping local optima
    def simulated_annealing(selected, time_limit):
        sq_count = [0] * len(squares)
        for e in selected:
            for sq_idx in edge_to_squares[e]:
                sq_count[sq_idx] += 1
        
        def can_add(e):
            for sq_idx in edge_to_squares[e]:
                if sq_count[sq_idx] == 3:
                    return False
            return True
        
        def add_edge(e):
            selected.add(e)
            for sq_idx in edge_to_squares[e]:
                sq_count[sq_idx] += 1
        
        def remove_edge(e):
            selected.remove(e)
            for sq_idx in edge_to_squares[e]:
                sq_count[sq_idx] -= 1
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while time.time() < time_limit and T > T_min:
            # Pick a random edge to remove
            e_out = random.choice(list(selected))
            remove_edge(e_out)
            
            # Try to add random edges
            added = []
            for _ in range(3):
                e_in = random.choice(edges)
                if e_in not in selected and can_add(e_in):
                    add_edge(e_in)
                    added.append(e_in)
            
            delta = len(added) - 1  # Removed 1, added len(added)
            
            if delta >= 0 or random.random() < np.exp(delta / T):
                # Accept
                pass
            else:
                # Revert
                for e in added:
                    remove_edge(e)
                add_edge(e_out)
            
            T *= cooling
        
        return selected
    
    # Run multiple restarts
    best_selected = None
    best_count = 0
    
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Greedy
        selected = greedy_construct()
        
        # Local search
        ls_time = min(10, deadline - time.time() - 5)
        if ls_time > 1:
            selected = local_search(selected, time.time() + ls_time)
        
        # Simulated annealing
        sa_time = min(5, deadline - time.time() - 5)
        if sa_time > 1:
            selected = simulated_annealing(selected, time.time() + sa_time)
        
        # Final local search polish
        if time.time() < deadline - 2:
            selected = local_search(selected, deadline - 2)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = set(selected)
    
    if best_selected is None:
        best_selected = greedy_construct()
    
    out = []
    for e in best_selected:
        out.append([e[0], e[1]])
    
    return out
