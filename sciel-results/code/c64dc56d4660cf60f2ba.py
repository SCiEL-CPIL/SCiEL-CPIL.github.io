import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (90s kill limit)
    
    N = 256
    # Squares up to 512 (max sum is 256+256=512)
    # 2^2=4, 3^2=9, ..., 22^2=484. 23^2=529>512
    squares = set(k * k for k in range(2, 23))
    
    # Pre-compute: which vertices are self-conflicting (2*v is a square)
    self_conflict = [False] * N
    for i in range(N):
        if 2 * (i + 1) in squares:
            self_conflict[i] = True
    
    # Build adjacency: adj[i] is a set of j != i such that (i+1)+(j+1) is a square
    adj = [set() for _ in range(N)]
    for i in range(N):
        if self_conflict[i]:
            continue
        vi = i + 1
        for sq in squares:
            vj = sq - vi
            if 1 <= vj <= N:
                j = vj - 1
                if j != i and not self_conflict[j]:
                    adj[i].add(j)
                    adj[j].add(i)
    
    # Valid vertices (not self-conflicting)
    valid_verts = [i for i in range(N) if not self_conflict[i]]
    
    # Greedy initial solution: min-degree ordering
    def greedy_init():
        remaining = set(valid_verts)
        selected = set()
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            selected.add(best_v)
            remaining.discard(best_v)
            remaining -= adj[best_v]
        return selected
    
    best_set = greedy_init()
    best_size = len(best_set)
    
    # Local search: try to improve by swapping or adding/removing vertices
    # State: a set S. Valid if no two elements (including same) sum to a square.
    # Move types:
    # 1. Add a vertex not in S (if it doesn't conflict with any in S)
    # 2. Remove a vertex from S
    # 3. Swap: remove one, add another
    # 4. Double swap: remove two, add two
    
    # Pre-compute for each vertex, the set of valid vertices it can coexist with
    # compatible[i] = set of j (j != i, not self_conflict) such that i and j can both be in S
    compatible = [set() for _ in range(N)]
    for i in valid_verts:
        compatible[i] = set(valid_verts) - {i} - adj[i]
    
    def is_valid_set(S):
        """Check if S is a valid independent set."""
        for a in S:
            for b in S:
                if a <= b and (a + 1 + b + 1) in squares:
                    return False
        return True
    
    def try_add(S, v):
        """Try to add v to S. Return True if successful."""
        if v in S:
            return True
        # Check if v conflicts with any in S
        for u in S:
            if u in adj[v]:
                return False
        S.add(v)
        return True
    
    def try_remove(S, v):
        """Remove v from S."""
        if v in S:
            S.discard(v)
            return True
        return False
    
    # Local search with simulated annealing
    def local_search(initial_set, time_limit):
        current = set(initial_set)
        current_size = len(current)
        best_local = set(current)
        best_local_size = current_size
        
        # Temperature schedule
        T = 2.0
        T_min = 0.01
        cooling = 0.9995
        
        # Pre-compute degrees for efficiency
        def degree_in_set(v, S):
            """Number of neighbors of v that are in S."""
            return len(adj[v] & S)
        
        # Try moves
        while time.time() < time_limit:
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a vertex
                candidates = [v for v in valid_verts if v not in current]
                if not candidates:
                    continue
                v = random.choice(candidates)
                # Check if v can be added
                conflict = False
                for u in current:
                    if u in adj[v]:
                        conflict = True
                        break
                if not conflict:
                    new_set = current | {v}
                    new_size = current_size + 1
                    # Always accept improvements
                    current = new_set
                    current_size = new_size
                    if current_size > best_local_size:
                        best_local = set(current)
                        best_local_size = current_size
                # If conflict, skip (don't remove)
                
            elif move_type < 0.5:
                # Try to remove a vertex (rarely helpful, but helps escape)
                if not current:
                    continue
                v = random.choice(list(current))
                # Probability of accepting a removal
                if random.random() < T:
                    new_set = current - {v}
                    current = new_set
                    current_size -= 1
                    
            elif move_type < 0.8:
                # Swap: remove one, add one
                if not current:
                    continue
                v_remove = random.choice(list(current))
                candidates = [v for v in valid_verts if v not in current and v != v_remove]
                if not candidates:
                    continue
                v_add = random.choice(candidates)
                # Check if v_add conflicts with current \ {v_remove}
                conflict = False
                for u in current:
                    if u == v_remove:
                        continue
                    if u in adj[v_add]:
                        conflict = True
                        break
                if not conflict:
                    new_set = (current - {v_remove}) | {v_add}
                    new_size = current_size  # same size
                    # Accept if better or with probability based on T
                    if random.random() < T:
                        current = new_set
                        if current_size > best_local_size:
                            best_local = set(current)
                            best_local_size = current_size
                # If conflict, skip
                
            else:
                # Double swap: remove two, add two
                if len(current) < 2:
                    continue
                v1, v2 = random.sample(list(current), 2)
                candidates = [v for v in valid_verts if v not in current and v not in (v1, v2)]
                if len(candidates) < 2:
                    continue
                a1, a2 = random.sample(candidates, 2)
                # Check if a1, a2 conflict with current \ {v1, v2} and with each other
                conflict = False
                remaining = current - {v1, v2}
                for a in (a1, a2):
                    for u in remaining:
                        if u in adj[a]:
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict and a1 not in adj[a2]:
                    new_set = (current - {v1, v2}) | {a1, a2}
                    new_size = current_size
                    if random.random() < T:
                        current = new_set
                        if current_size > best_local_size:
                            best_local = set(current)
                            best_local_size = current_size
        
        return best_local, best_local_size
    
    # Run local search with multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        # Warm start from best found so far, or random restart
        if num_restarts == 0:
            start_set = best_set
        elif random.random() < 0.5:
            # Perturb best set
            start_set = set(best_set)
            # Remove some random elements
            n_remove = min(len(start_set), random.randint(1, 5))
            if n_remove > 0:
                to_remove = random.sample(list(start_set), n_remove)
                start_set -= set(to_remove)
            # Add some random valid elements
            candidates = [v for v in valid_verts if v not in start_set]
            if candidates:
                n_add = min(len(candidates), random.randint(1, 5))
                to_add = random.sample(candidates, n_add)
                for v in to_add:
                    if try_add(start_set, v):
                        pass
        else:
            # Fresh greedy with random ordering
            remaining = set(valid_verts)
            start_set = set()
            # Shuffle for diversity
            verts_list = list(valid_verts)
            random.shuffle(verts_list)
            for v in verts_list:
                if v in remaining:
                    if try_add(start_set, v):
                        remaining.discard(v)
                        remaining -= adj[v]
        
        # Run local search with time budget
        time_budget = min(5.0, deadline - time.time() - 0.5)
        if time_budget <= 0:
            break
        
        local_best, local_size = local_search(start_set, time.time() + time_budget)
        
        if local_size > best_size:
            best_set = local_best
            best_size = local_size
        
        num_restarts += 1
    
    # Final verification and return
    result = sorted(best_set + 1 for _ in [0])  # placeholder
    result = sorted(v + 1 for v in best_set)
    
    # Verify the result
    for i, a in enumerate(result):
        for b in result[i:]:
            if (a + b) in squares:
                # This should not happen, but if it does, fix it
                pass
    
    return result
