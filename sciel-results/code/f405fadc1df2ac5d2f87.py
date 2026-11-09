import random
import time
from itertools import combinations

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115
    
    # Precompute all triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # For fast updates, we need to know which triples contain a given vertex
    # and which pairs of vertices are in the triple with a third vertex.
    # Actually, it's easier to just iterate over relevant triples when updating.
    # For vertex u, the triples containing u are those where u is one of the 3.
    # We can precompute for each vertex u, the list of triple indices containing u.
    triples_containing = [[] for _ in range(n)]
    for i, (a, b, c) in enumerate(triples):
        triples_containing[a].append(i)
        triples_containing[b].append(i)
        triples_containing[c].append(i)
        
    # common_count[t] = number of common neighbors of the triple t
    # adj_mask[u] = bitmask of neighbors of u
    
    def initialize_graph():
        adj_mask = [0] * n
        # Start with a random graph, maybe p=0.5
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Compute initial common counts
        common_count = [0] * num_triples
        for t_idx, (u, v, w) in enumerate(triples):
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            common_count[t_idx] = bin(common).count('1')
            
        return adj_mask, common_count

    def is_valid(common_count):
        return all(c < 3 for c in common_count)
        
    def add_edge(adj_mask, common_count, u, v):
        """
        Tries to add edge (u, v).
        Returns True if added and graph remains valid, False otherwise.
        If False, state is unchanged.
        """
        # Check if adding (u, v) creates a violation.
        # A violation occurs if there is a triple (a, b, c) such that
        # the number of common neighbors becomes >= 3.
        # Adding edge (u, v) increases the common neighbor count for triples
        # that contain u or v, but only if the *other* two vertices in the triple
        # are neighbors of both u and v? No.
        
        # Let's think carefully.
        # common_count(t) for t=(a,b,c) is |N(a) ∩ N(b) ∩ N(c)|.
        # If we add edge (u, v), N(u) gains v, N(v) gains u.
        # This affects common counts for triples containing u or v.
        
        # Case 1: Triple contains u but not v. t = (u, x, y).
        # N(u) changes. New N(u) = Old N(u) ∪ {v}.
        # New intersection N(u)∩N(x)∩N(y) = Old Intersection ∪ ({v} ∩ N(x) ∩ N(y)).
        # So count increases by 1 if v is neighbor of x and y.
        
        # Case 2: Triple contains v but not u. t = (v, x, y).
        # Similarly, count increases by 1 if u is neighbor of x and y.
        
        # Case 3: Triple contains both u and v. t = (u, v, x).
        # N(u) gains v, N(v) gains u.
        # New N(u)∩N(v)∩N(x) = (Old N(u)∪{v}) ∩ (Old N(v)∪{u}) ∩ N(x)
        # = (Old N(u)∩Old N(v)∩N(x)) ∪ (Old N(u)∩{u}∩N(x)) ... wait u is not in N(u).
        # Let's use set logic.
        # A = N(u), B = N(v), C = N(x).
        # New A' = A ∪ {v}, New B' = B ∪ {u}.
        # A' ∩ B' ∩ C = (A ∪ {v}) ∩ (B ∩ C) ∪ ({v} ∩ B ∩ C) ... this is getting complex.
        # Simpler: 
        # Old common = A ∩ B ∩ C.
        # New common = (A ∪ {v}) ∩ (B ∪ {u}) ∩ C
        # = (A ∩ B ∩ C) ∪ (A ∩ {u} ∩ C) ∪ ({v} ∩ B ∩ C) ∪ ({v} ∩ {u} ∩ C)
        # Note: u ∉ A, v ∉ B.
        # A ∩ {u} = empty.
        # {v} ∩ {u} = empty.
        # So New common = Old common ∪ ({v} ∩ B ∩ C).
        # {v} ∩ B ∩ C is non-empty iff v ∈ B and v ∈ C.
        # i.e., v is neighbor of v (false) ... wait.
        # B is N(v). v is not in N(v). So {v} ∩ B is empty.
        # Therefore, for triples containing both u and v, the common count does NOT change?
        # Let's re-verify.
        # If t=(u,v,x), common neighbors are vertices w such that w~u, w~v, w~x.
        # Adding edge u-v:
        # Does u become a common neighbor? No, u is not neighbor of u.
        # Does v become a common neighbor? No, v is not neighbor of v.
        # Do any other vertices become common neighbors?
        # The condition for w to be common is w~u AND w~v AND w~x.
        # Adding u-v doesn't change w~u for w≠v, nor w~v for w≠u.
        # For w=v: v~u is now true. v~v is false. So v is not a common neighbor.
        # For w=u: u~v is now true. u~u is false. So u is not a common neighbor.
        # So yes, adding edge (u,v) does NOT change common counts for triples containing both u and v.
        
        # So we only need to check triples containing exactly one of u or v.
        
        # Check triples containing u (but not v)
        potential_violation = False
        changes = [] # List of (triple_idx, delta)
        
        for t_idx in triples_containing[u]:
            t = triples[t_idx]
            if v in t:
                continue # Skip if v is also in triple
            
            # t = (u, x, y)
            # Count increases if v is neighbor of x and y.
            # Check if v is neighbor of x and y.
            # x, y are the other two vertices in t.
            if t[0] == u:
                x, y = t[1], t[2]
            elif t[1] == u:
                x, y = t[0], t[2]
            else:
                x, y = t[0], t[1]
                
            if (adj_mask[v] & (1 << x)) and (adj_mask[v] & (1 << y)):
                # v is neighbor of x and y.
                # So v becomes a new common neighbor of u, x, y.
                # Current count for t_idx increases by 1.
                if common_count[t_idx] >= 2:
                    potential_violation = True
                changes.append((t_idx, 1))
                
        # Check triples containing v (but not u)
        if not potential_violation:
            for t_idx in triples_containing[v]:
                t = triples[t_idx]
                if u in t:
                    continue
                
                if t[0] == v:
                    x, y = t[1], t[2]
                elif t[1] == v:
                    x, y = t[0], t[2]
                else:
                    x, y = t[0], t[1]
                    
                if (adj_mask[u] & (1 << x)) and (adj_mask[u] & (1 << y)):
                    if common_count[t_idx] >= 2:
                        potential_violation = True
                    changes.append((t_idx, 1))
                    
        if potential_violation:
            return False
            
        # Apply changes
        for t_idx, delta in changes:
            common_count[t_idx] += delta
            
        # Update adjacency
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        return True

    def remove_edge(adj_mask, common_count, u, v):
        """
        Removes edge (u, v) and updates common counts.
        """
        # Reverse of add_edge logic.
        # Decrement counts for triples containing u (not v) where v was neighbor of x,y.
        # And triples containing v (not u) where u was neighbor of x,y.
        
        for t_idx in triples_containing[u]:
            t = triples[t_idx]
            if v in t:
                continue
            if t[0] == u:
                x, y = t[1], t[2]
            elif t[1] == u:
                x, y = t[0], t[2]
            else:
                x, y = t[0], t[1]
                
            # Before removal, v was neighbor of u.
            # If v is neighbor of x and y, it was a common neighbor.
            if (adj_mask[v] & (1 << x)) and (adj_mask[v] & (1 << y)):
                common_count[t_idx] -= 1
                
        for t_idx in triples_containing[v]:
            t = triples[t_idx]
            if u in t:
                continue
            if t[0] == v:
                x, y = t[1], t[2]
            elif t[1] == v:
                x, y = t[0], t[2]
            else:
                x, y = t[0], t[1]
                
            if (adj_mask[u] & (1 << x)) and (adj_mask[u] & (1 << y)):
                common_count[t_idx] -= 1
                
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    best_edges = None
    best_count = 0

    # Run multiple trials
    for trial in range(50):
        if time.time() > deadline:
            break
            
        adj_mask, common_count = initialize_graph()
        
        # Repair phase: remove edges until valid
        while not is_valid(common_count):
            if time.time() > deadline:
                break
            # Find a violating triple
            viol_idx = -1
            for i, c in enumerate(common_count):
                if c >= 3:
                    viol_idx = i
                    break
            if viol_idx == -1:
                break # Should not happen if is_valid returned False
                
            u, v, w = triples[viol_idx]
            # We need to remove an edge to break this violation.
            # The common neighbors are vertices z such that z~u, z~v, z~w.
            common_mask = adj_mask[u] & adj_mask[v] & adj_mask[w]
            common_neighbors = [i for i in range(n) if (common_mask >> i) & 1]
            
            # Pick a random common neighbor z and remove a random edge from (u,v,w) to z
            z = random.choice(common_neighbors)
            # Remove edge between z and one of u,v,w
            # Removing z-u, z-v, or z-w will reduce the common count for this triple by 1.
            # It might affect other triples.
            target = random.choice([u, v, w])
            remove_edge(adj_mask, common_count, z, target)
            
        if time.time() > deadline:
            break
            
        # Optimization phase: Hill Climbing
        current_count = bin(sum(adj_mask)).count('1') // 2
        improved = True
        while improved:
            improved = False
            # Get non-edges
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj_mask[i] >> j) & 1]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if add_edge(adj_mask, common_count, u, v):
                    current_count += 1
                    improved = True
                    
            if current_count > best_count:
                best_count = current_count
                best_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_mask[i] >> j) & 1:
                            best_edges.append([i, j])
                            
        # Kick phase
        for _ in range(10):
            if time.time() > deadline:
                break
            # Remove 2-3 random edges
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if (adj_mask[i] >> j) & 1]
            if not edges_list:
                break
            for _ in range(2):
                u, v = random.choice(edges_list)
                remove_edge(adj_mask, common_count, u, v)
                current_count -= 1
                edges_list.remove((u, v)) # Avoid removing same edge twice immediately if possible, or just recompute
                # Recompute edges list is expensive, but n is small.
                # Actually, just pick random from current adj_mask
                # Let's just re-pick from adj_mask
                all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj_mask[i] >> j) & 1]
                if not all_edges: break
                u, v = random.choice(all_edges)
                remove_edge(adj_mask, common_count, u, v)
                current_count -= 1
                
            # Re-optimize
            improved = True
            while improved:
                improved = False
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj_mask[i] >> j) & 1]
                random.shuffle(non_edges)
                for u, v in non_edges:
                    if time.time() > deadline:
                        break
                    if add_edge(adj_mask, common_count, u, v):
                        current_count += 1
                        improved = True
                        
            if current_count > best_count:
                best_count = current_count
                best_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_mask[i] >> j) & 1:
                            best_edges.append([i, j])

    if best_edges is None:
        # Fallback: Empty graph or simple path
        return [[i, i+1] for i in range(19)]
        
    return best_edges
