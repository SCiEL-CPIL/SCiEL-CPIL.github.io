import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute for each vertex, which triples it belongs to
    triples_by_vertex = [[] for _ in range(n)]
    for idx, tri in enumerate(all_triples):
        for v in tri:
            triples_by_vertex[v].append(idx)
    
    # For each edge (u,v), precompute which triples are "affected" 
    # A triple (i,j,k) is affected by edge (u,v) if adding/removing (u,v)
    # changes whether u or v is a common neighbor of (i,j,k)
    # u is a common neighbor of (i,j,k) iff u != i,j,k and u is adjacent to all of i,j,k
    # So edge (u,v) affects triple (i,j,k) if u in {i,j,k} or v in {i,j,k}
    # OR if u/v is a potential common neighbor (adjacent to all three)
    # Actually, the common neighbors of (i,j,k) are vertices adjacent to all three.
    # Adding edge (u,v) changes adjacency of u and v.
    # u's adjacency to i,j,k determines if u is a common neighbor of (i,j,k).
    # But u is only a common neighbor if u != i,j,k.
    # So edge (u,v) affects triple (i,j,k) if:
    #   - u is in {i,j,k}: then u's adjacency to the other two changes
    #   - v is in {i,j,k}: same
    #   - u != i,j,k and u is adjacent to at least 2 of {i,j,k}: adding/removing (u,v) 
    #     doesn't affect u's status as common neighbor of (i,j,k) unless v is one of i,j,k
    # Wait, I need to think more carefully.
    # 
    # Common neighbors of triple (i,j,k) = {x : x != i,j,k and x~i and x~j and x~k}
    # When we add/remove edge (u,v):
    #   - If u is in {i,j,k}: u is no longer/now a candidate (but u can't be its own neighbor)
    #     Actually u is excluded from common neighbors if u in {i,j,k}.
    #     The change: v's adjacency to i,j,k changes. If v != i,j,k, then v might gain/lose
    #     status as common neighbor of (i,j,k).
    #   - Similarly if v in {i,j,k}: u might gain/lose status.
    #   - If u not in {i,j,k} and v not in {i,j,k}: 
    #     u's adjacency to i,j,k unchanged, v's adjacency to i,j,k unchanged.
    #     So no triple's common neighbor set changes!
    # 
    # Wait that's wrong. Adding edge (u,v) changes u's neighbor set and v's neighbor set.
    # For triple (i,j,k), the common neighbors are determined by adjacency to i, j, k.
    # If u is not in {i,j,k}, then u's adjacency to i,j,k is unchanged by adding (u,v).
    # So u's status as common neighbor of (i,j,k) is unchanged.
    # Same for v.
    # So only triples where u or v is a member are affected!
    
    # Precompute: for each edge (u,v), list of triple indices where u or v is in the triple
    edge_affected = {}
    for u, v in all_edges:
        affected = set(triples_by_vertex[u]) | set(triples_by_vertex[v])
        edge_affected[(u,v)] = list(affected)
    
    def build_greedy_seed():
        """Build a valid graph greedily, trying edges in random order."""
        adj = [0] * n
        # triple_common_count[t] = number of common neighbors of triple t
        triple_cn = [0] * num_triples
        
        # For efficiency, when we add edge (u,v), we need to update triple_cn
        # for triples where u or v is a member.
        # For triple (i,j,k) where u is a member (say u=i):
        #   v becomes a common neighbor iff v != j,k and v~j and v~k
        #   (v's adjacency to j and k is what matters, and v != i,j,k)
        #   Since u=i, we need v != j,k.
        #   v~j and v~k: check current adjacency.
        
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        
        for u, v in edge_order:
            # Check if adding (u,v) creates any violation
            # Only need to check triples where u or v is a member
            affected = edge_affected[(u,v)]
            can_add = True
            for t_idx in affected:
                i, j, k = all_triples[t_idx]
                # Adding edge (u,v): does this make any vertex a 3rd common neighbor?
                # We need to check if after adding, any triple has >= 3 common neighbors
                # Currently triple_cn[t_idx] is the count before adding.
                # After adding, the count might increase.
                # Which vertices' status as common neighbor changes?
                # u's status: u is common neighbor of (i,j,k) iff u notin {i,j,k} and u~i,j,k
                # v's status: v is common neighbor of (i,j,k) iff v notin {i,j,k} and v~i,j,k
                # 
                # If u in {i,j,k}: u is excluded, so u's status doesn't matter.
                #   But v's status changes: v gains adjacency to u (which is in {i,j,k}).
                #   v is common neighbor iff v notin {i,j,k} and v~i,j,k.
                #   Before: v~i,j,k (excluding the new edge to u).
                #   After: v~i,j,k including edge to u.
                #   So if u is in {i,j,k}, say u=i, then v~i is now true.
                #   v is common neighbor iff v != j,k and v~j and v~k.
                #   (v~i is now true automatically)
                # 
                # Let's just compute the delta.
                
                # Determine if u is in the triple
                u_in = (u == i or u == j or u == k)
                v_in = (v == i or v == j or v == k)
                
                delta = 0
                if not u_in and not v_in:
                    # Neither u nor v is in the triple.
                    # u's status as common neighbor: unchanged (u's adj to i,j,k unchanged)
                    # v's status as common neighbor: unchanged
                    # So no change.
                    delta = 0
                elif u_in and not v_in:
                    # u is in triple, v is not.
                    # v's status changes: v gains edge to u.
                    # v is common neighbor iff v notin {i,j,k} (true) and v~i,j,k.
                    # Before adding: v~i,j,k where i,j,k include u.
                    #   v~u was false (edge not present).
                    #   So before: v~i,j,k requires v~u AND v~(other two).
                    #   Since v~u was false, v was NOT a common neighbor before.
                    # After adding: v~u is true.
                    #   v is common neighbor iff v~(the other two in {i,j,k} besides u).
                    # Let's find the other two.
                    others = [x for x in (i,j,k) if x != u]
                    if (adj[v] & (1 << others[0])) and (adj[v] & (1 << others[1])):
                        delta = 1  # v becomes a common neighbor
                    else:
                        delta = 0
                elif v_in and not u_in:
                    # Symmetric
                    others = [x for x in (i,j,k) if x != v]
                    if (adj[u] & (1 << others[0])) and (adj[u] & (1 << others[1])):
                        delta = 1
                    else:
                        delta = 0
                else:
                    # Both u and v in triple. But u != v and triple has 3 distinct vertices.
                    # So both u and v are in {i,j,k}. The third vertex w.
                    # u's status: u is in triple, so excluded. No change to u's status.
                    # v's status: v is in triple, so excluded. No change to v's status.
                    # Wait, but the edge (u,v) is between two members of the triple.
                    # This doesn't affect whether any OTHER vertex is a common neighbor.
                    # And u,v are excluded from being common neighbors.
                    # So delta = 0.
                    delta = 0
                
                if triple_cn[t_idx] + delta >= 3:
                    can_add = False
                    break
            
            if can_add:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                # Update triple_cn for affected triples
                for t_idx in affected:
                    i, j, k = all_triples[t_idx]
                    u_in = (u == i or u == j or u == k)
                    v_in = (v == i or v == j or v == k)
                    
                    delta = 0
                    if not u_in and not v_in:
                        delta = 0
                    elif u_in and not v_in:
                        others = [x for x in (i,j,k) if x != u]
                        if (adj[v] & (1 << others[0])) and (adj[v] & (1 << others[1])):
                            delta = 1
                    elif v_in and not u_in:
                        others = [x for x in (i,j,k) if x != v]
                        if (adj[u] & (1 << others[0])) and (adj[u] & (1 << others[1])):
                            delta = 1
                    else:
                        delta = 0
                    triple_cn[t_idx] += delta
        
        return adj, triple_cn
    
    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2
    
    def is_valid(triple_cn):
        return all(c < 3 for c in triple_cn)
    
    def tabu_search(adj, triple_cn, max_time):
        """Tabu search with single-edge flips."""
        ts_start = time.time()
        current_adj = adj[:]
        current_cn = triple_cn[:]
        current_edges = count_edges(current_adj)
        
        best_adj = current_adj[:]
        best_cn = current_cn[:]
        best_edges = current_edges
        
        # Tabu list: (edge, time_added)
        tabu = {}
        tabu_tenure = 50
        
        # Precompute edge index
        edge_set = set()
        for u, v in all_edges:
            if current_adj[u] & (1 << v):
                edge_set.add((u,v))
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() - ts_start < max_time:
            if no_improve > max_no_improve:
                break
            
            # Choose random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            edge = (min(u,v), max(u,v))
            u, v = edge
            
            is_present = (u, v) in edge_set
            
            # Check tabu
            if edge in tabu and tabu[edge] > len(tabu) - tabu_tenure:
                continue
            
            # Compute effect
            affected = edge_affected[edge]
            
            # Determine if flip is allowed
            # If adding: check no triple exceeds 2 common neighbors
            # If removing: always valid (reduces common neighbors)
            
            if is_present:
                # Removing edge
                new_cn_check = True
                for t_idx in affected:
                    i, j, k = all_triples[t_idx]
                    u_in = (u == i or u == j or u == k)
                    v_in = (v == i or v == j or v == k)
                    
                    delta = 0
                    if not u_in and not v_in:
                        delta = 0
                    elif u_in and not v_in:
                        others = [x for x in (i,j,k) if x != u]
                        if (current_adj[v] & (1 << others[0])) and (current_adj[v] & (1 << others[1])):
                            delta = -1  # v was a common neighbor, now not
                    elif v_in and not u_in:
                        others = [x for x in (i,j,k) if x != v]
                        if (current_adj[u] & (1 << others[0])) and (current_adj[u] & (1 << others[1])):
                            delta = -1
                    else:
                        delta = 0
                    
                    if current_cn[t_idx] + delta < 0:
                        new_cn_check = False
                        break
                
                if new_cn_check:
                    # Apply removal
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    edge_set.discard((u,v))
                    current_edges -= 1
                    
                    for t_idx in affected:
                        i, j, k = all_triples[t_idx]
                        u_in = (u == i or u == j or u == k)
                        v_in = (v == i or v == j or v == k)
                        
                        delta = 0
                        if not u_in and not v_in:
                            delta = 0
                        elif u_in and not v_in:
                            others = [x for x in (i,j,k) if x != u]
                            if (current_adj[v] & (1 << others[0])) and (current_adj[v] & (1 << others[1])):
                                delta = -1
                        elif v_in and not u_in:
                            others = [x for x in (i,j,k) if x != v]
                            if (current_adj[u] & (1 << others[0])) and (current_adj[u] & (1 << others[1])):
                                delta = -1
                        else:
                            delta = 0
                        current_cn[t_idx] += delta
                    
                    tabu[edge] = len(tabu)
                    no_improve += 1
            else:
                # Adding edge
                can_add = True
                for t_idx in affected:
                    i, j, k = all_triples[t_idx]
                    u_in = (u == i or u == j or u == k)
                    v_in = (v == i or v == j or v == k)
                    
                    delta = 0
                    if not u_in and not v_in:
                        delta = 0
                    elif u_in and not v_in:
                        others = [x for x in (i,j,k) if x != u]
                        if (current_adj[v] & (1 << others[0])) and (current_adj[v] & (1 << others[1])):
                            delta = 1
                    elif v_in and not u_in:
                        others = [x for x in (i,j,k) if x != v]
                        if (current_adj[u] & (1 << others[0])) and (current_adj[u] & (1 << others[1])):
                            delta = 1
                    else:
                        delta = 0
                    
                    if current_cn[t_idx] + delta >= 3:
                        can_add = False
                        break
                
                if can_add:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    edge_set.add((u,v))
                    current_edges += 1
                    
                    for t_idx in affected:
                        i, j, k = all_triples[t_idx]
                        u_in = (u == i or u == j or u == k)
                        v_in = (v == i or v == j or v == k)
                        
                        delta = 0
                        if not u_in and not v_in:
                            delta = 0
                        elif u_in and not v_in:
                            others = [x for x in (i,j,k) if x != u]
                            if (current_adj[v] & (1 << others[0])) and (current_adj[v] & (1 << others[1])):
                                delta = 1
                        elif v_in and not u_in:
                            others = [x for x in (i,j,k) if x != v]
                            if (current_adj[u] & (1 << others[0])) and (current_adj[u] & (1 << others[1])):
                                delta = 1
                        else:
                            delta = 0
                        current_cn[t_idx] += delta
                    
                    tabu[edge] = len(tabu)
                    no_improve = 0
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        best_cn = current_cn[:]
                else:
                    no_improve += 1
        
        return best_adj, best_edges
    
    # Main
    best_adj = [0] * n
    best_edges = 0
    
    # Run multiple greedy seeds with different random orders
    num_seeds = 10
    time_per_seed = (deadline - start_time - 5) / num_seeds
    
    for s in range(num_seeds):
        if time.time() > deadline - 5:
            break
        
        adj, triple_cn = build_greedy_seed()
        edges = count_edges(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
        
        # Tabu search
        remaining = deadline - time.time() - 2
        ts_time = max(0.5, remaining / (num_seeds - s))
        adj, edges = tabu_search(adj, triple_cn, ts_time)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Final result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
