import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    total_pairs = n * (n - 1) // 2
    
    # Pairs[v] is a set of frozensets {x, y} where x, y are neighbours of v
    # We use lists of tuples for speed, or sets of integers encoded as x*n+y
    
    def is_valid_addition(adj, Pairs, u, v):
        if adj[u] & (1 << v):
            return False
        
        # Check triples containing u (and not v)
        # We need to check if adding v to N(u) causes any triple {u, x, y} to have 3 common neighbours.
        # This happens if v is in N(x) and N(y) (i.e., {x,y} in Pairs[v]) 
        # AND the current common neighbours of u, x, y is exactly 2.
        
        # Iterate over pairs in Pairs[v]
        for x, y in Pairs[v]:
            if x == u or y == u:
                continue
            # Count common neighbours of u, x, y in current graph
            # N(u) & N(x) & N(y)
            count = (adj[u] & adj[x] & adj[y]).bit_count()
            if count == 2:
                return False
                
        # Check triples containing v (and not u)
        for x, y in Pairs[u]:
            if x == v or y == v:
                continue
            count = (adj[v] & adj[x] & adj[y]).bit_count()
            if count == 2:
                return False
                
        return True

    def add_edge(adj, Pairs, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update Pairs
        # v is added to N(u). So for any z in N(v), {u, z} is a new pair in Pairs[v]? No.
        # Pairs[w] contains pairs of neighbours of w.
        # Since u is now a neighbour of v, and v is now a neighbour of u.
        
        # 1. For any z in N(v) (old), u and z are both neighbours of v? No, u is new neighbour of v.
        #    So {u, z} is a pair of neighbours of v. Add {u, z} to Pairs[v].
        for z in range(n):
            if z == u or z == v: continue
            if adj[v] & (1 << z): # z is neighbour of v (including new u? No, we updated adj first)
                # We need to be careful with order. 
                # Let's compute neighbours before update or use masks.
                pass
        
        # Let's do it cleanly:
        # Old N(u), Old N(v)
        # New N(u) = Old N(u) U {v}
        # New N(v) = Old N(v) U {u}
        
        # Pairs[v] needs to include pairs {u, z} for all z in Old N(v).
        # And {z1, z2} for z1, z2 in Old N(v) are already there? Yes.
        # So just add {u, z} for z in Old N(v).
        
        # Pairs[u] needs to include pairs {v, z} for all z in Old N(u).
        
        # For any w in Old N(u) INTERSECT Old N(v):
        # u and v are both neighbours of w. So {u, v} is a new pair in Pairs[w].
        
        old_Nu = adj[u] & ~(1 << v)
        old_Nv = adj[v] & ~(1 << u)
        
        # Add to Pairs[v]
        for z in range(n):
            if z == u or z == v: continue
            if old_Nv & (1 << z):
                Pairs[v].append((u, z))
                
        # Add to Pairs[u]
        for z in range(n):
            if z == u or z == v: continue
            if old_Nu & (1 << z):
                Pairs[u].append((v, z))
                
        # Add {u, v} to Pairs[w] for w in old_Nu & old_Nv
        common = old_Nu & old_Nv
        for w in range(n):
            if common & (1 << w):
                Pairs[w].append((u, v))

    def remove_edge(adj, Pairs, u, v):
        # This is harder. We need to remove specific pairs from Pairs.
        # Given the frequency of removals, it might be better to recompute Pairs or use a different structure.
        # For now, let's avoid removals in the main loop or recompute.
        # Actually, local search involves swaps (remove then add).
        # Recomputing Pairs takes O(n^2 * avg_degree). n=20, so 400 * 10 = 4000 ops. Very fast.
        
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Recompute Pairs
        new_Pairs = [[] for _ in range(n)]
        for w in range(n):
            # Get neighbours of w
            neighs = [x for x in range(n) if x != w and (adj[w] & (1 << x))]
            for i in range(len(neighs)):
                for j in range(i+1, len(neighs)):
                    new_Pairs[w].append((neighs[i], neighs[j]))
        return new_Pairs

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        Pairs = [[] for _ in range(n)]
        
        # Initialize with random edges? Or start empty?
        # Starting empty and adding is safe.
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, Pairs, u, v):
                add_edge(adj, Pairs, u, v)
                
        return adj, Pairs

    def local_search(adj, Pairs, max_steps=1000):
        current_adj = adj[:]
        current_Pairs = Pairs
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_Pairs = current_Pairs
        best_edges = current_edges
        
        # Phase 1: Greedy Add
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_adj, current_Pairs, u, v):
                    add_edge(current_adj, current_Pairs, u, v)
                    current_edges += 1
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        best_Pairs = current_Pairs
        
        # Phase 2: Swaps
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            if not edges:
                break
            random.shuffle(edges)
            
            found_swap = False
            for u, v in edges:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                # Try removing u-v
                # We need to check if we can add 2 edges after removing u-v.
                # Instead of actually removing, we can simulate or just remove and recompute.
                # Removal is expensive if we recompute Pairs.
                # Let's just remove, recompute, try to add, and if fail, put back.
                
                # Remove u-v
                temp_adj = current_adj[:]
                temp_adj[u] &= ~(1 << v)
                temp_adj[v] &= ~(1 << u)
                
                # Recompute Pairs for temp_adj
                temp_Pairs = [[] for _ in range(n)]
                for w in range(n):
                    neighs = [x for x in range(n) if x != w and (temp_adj[w] & (1 << x))]
                    for i in range(len(neighs)):
                        for j in range(i+1, len(neighs)):
                            temp_Pairs[w].append((neighs[i], neighs[j]))
                            
                added = 0
                # Try to add missing edges
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (temp_adj[i] & (1 << j))]
                random.shuffle(missing)
                
                for a, b in missing:
                    if added == 2:
                        break
                    if is_valid_addition(temp_adj, temp_Pairs, a, b):
                        add_edge(temp_adj, temp_Pairs, a, b)
                        added += 1
                        
                if added == 2:
                    # Success
                    current_adj = temp_adj
                    current_Pairs = temp_Pairs
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        best_Pairs = current_Pairs
                    found_swap = True
                # If added < 2, we discard temp_adj and keep current_adj
        return best_adj, best_Pairs, best_edges

    best_adj = [0] * n
    best_Pairs = [[] for _ in range(n)]
    best_edges = 0
    
    # Run multiple restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj, Pairs = greedy_construct(i * 7 + 1)
        adj, Pairs, edges = local_search(adj, Pairs, max_steps=200)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            best_Pairs = Pairs
            
    # Final polish
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            adj, Pairs, edges = local_search(best_adj, best_Pairs, max_steps=500)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
                best_Pairs = Pairs
            else:
                break

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
