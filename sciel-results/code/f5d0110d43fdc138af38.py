import time
import random
import sys

def construct():
    """
    Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 18
    
    def get_edges_from_adj(adj_mask):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def is_k4_free(adj_mask):
        """
        Check if graph represented by adj_mask is K4-free.
        """
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    # Common neighbors of u and v
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    # Check if any two vertices in common are connected
                    c1 = common
                    while c1:
                        lsb = c1 & -c1
                        c1 ^= lsb
                        w = lsb.bit_length() - 1
                        # Check if w is connected to any other vertex in common (excluding w)
                        w_neigh_in_common = adj_mask[w] & common
                        if w_neigh_in_common:
                            return False
        return True

    def eval_max_tf(adj_mask):
        """
        Find the largest triangle-free induced subgraph size.
        """
        # Precompute forbidden pairs for each vertex
        # forbidden_pairs[v] is a list of bitmasks representing pairs {u, w}
        # such that (u, v, w) is a triangle.
        forbidden_pairs = [[] for _ in range(n)]
        
        for v in range(n):
            neigh_v = adj_mask[v]
            # Iterate through neighbors u of v
            u_bit = neigh_v
            while u_bit:
                lsb_u = u_bit & -u_bit
                u_bit ^= lsb_u
                u = lsb_u.bit_length() - 1
                # For neighbor u, find common neighbors w of u and v
                # w must be in neigh_v and adj_mask[u]
                common_w = neigh_v & adj_mask[u]
                common_w &= ~(1 << u)
                common_w &= ~(1 << v)
                
                # For each w in common_w, (u, v, w) is a triangle
                w_bit = common_w
                while w_bit:
                    lsb_w = w_bit & -w_bit
                    w_bit ^= lsb_w
                    w = lsb_w.bit_length() - 1
                    pair_mask = (1 << u) | (1 << w)
                    forbidden_pairs[v].append(pair_mask)
        
        # DP over all masks
        valid = [False] * (1 << n)
        valid[0] = True
        max_size = 0
        
        for mask in range(1, 1 << n):
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            prev_mask = mask ^ lsb
            
            if not valid[prev_mask]:
                valid[mask] = False
                continue
            
            # Check if adding v creates a triangle
            # This happens if prev_mask contains any pair from forbidden_pairs[v]
            conflict_found = False
            for pair_mask in forbidden_pairs[v]:
                if (prev_mask & pair_mask) == pair_mask:
                    conflict_found = True
                    break
            
            if not conflict_found:
                valid[mask] = True
                size = mask.bit_count()
                if size > max_size:
                    max_size = size
            else:
                valid[mask] = False
                
        return max_size

    def generate_random_k4_free():
        """
        Generate a random K4-free graph by starting with a random graph
        and removing edges from K4s until none remain.
        """
        adj = [0] * n
        # Start with random graph p=0.7 (denser is better for minimizing TF subgraph)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.7:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        changed = True
        while changed:
            changed = False
            if is_k4_free(adj):
                break
            
            # Find a K4
            k4_found = None
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        c1 = common
                        while c1:
                            lsb = c1 & -c1
                            c1 ^= lsb
                            w = lsb.bit_length() - 1
                            w_neigh_in_common = adj[w] & common
                            if w_neigh_in_common:
                                lsb_x = w_neigh_in_common & -w_neigh_in_common
                                x = lsb_x.bit_length() - 1
                                k4_found = (u, v, w, x)
                                break
                        if k4_found:
                            break
                if k4_found:
                    break
            
            if k4_found:
                u, v, w, x = k4_found
                # Remove a random edge from the K4
                edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                eu, ev = random.choice(edges)
                adj[eu] &= ~(1 << ev)
                adj[ev] &= ~(1 << eu)
                changed = True
        
        return adj

    def local_search_swap(adj_mask, time_limit):
        """
        Local search using edge swaps (remove one, add another).
        """
        current_adj = adj_mask[:]
        current_score = eval_max_tf(current_adj)
        best_adj = current_adj[:]
        best_score = current_score
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while time.time() < time_limit:
            improved = False
            # Try a few random swaps
            for _ in range(50):
                if time.time() > time_limit:
                    break
                
                # Pick random edge to remove
                rem_edge = random.choice(all_edges)
                if not ((current_adj[rem_edge[0]] >> rem_edge[1]) & 1):
                    continue # Only remove existing edges
                
                # Pick random edge to add (non-edge)
                add_edge = random.choice(all_edges)
                if (current_adj[add_edge[0]] >> add_edge[1]) & 1:
                    continue # Only add non-edges
                if add_edge == rem_edge or (add_edge[0], add_edge[1]) == (rem_edge[1], rem_edge[0]):
                    continue # Can't remove and add same edge in a swap effectively (net zero)
                
                # Apply swap
                u_rem, v_rem = rem_edge
                u_add, v_add = add_edge
                
                current_adj[u_rem] &= ~(1 << v_rem)
                current_adj[v_rem] &= ~(1 << u_rem)
                current_adj[u_add] |= (1 << v_add)
                current_adj[v_add] |= (1 << u_add)
                
                # Check K4-free
                if not is_k4_free(current_adj):
                    # Revert
                    current_adj[u_rem] |= (1 << v_rem)
                    current_adj[v_rem] |= (1 << u_rem)
                    current_adj[u_add] &= ~(1 << v_add)
                    current_adj[v_add] &= ~(1 << u_add)
                    continue
                
                new_score = eval_max_tf(current_adj)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                    improved = True
                    # Keep the swap
                else:
                    # Revert
                    current_adj[u_rem] |= (1 << v_rem)
                    current_adj[v_rem] |= (1 << u_rem)
                    current_adj[u_add] &= ~(1 << v_add)
                    current_adj[v_add] &= ~(1 << u_add)
                    
            if not improved:
                # If no improvement in 50 swaps, try a larger jump or restart?
                # For now, just continue, hoping for a lucky swap later.
                # Or accept equal score moves?
                pass

        return best_adj, best_score

    # Main search loop
    best_adj = None
    best_score = n + 1
    
    # Try a few random starts
    num_starts = 10
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        adj = generate_random_k4_free()
        
        # Evaluate initial
        score = eval_max_tf(adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
        # Local search
        remaining = deadline - time.time() - 5
        if remaining > 2:
            adj_ls, score_ls = local_search_swap(adj, remaining)
            if score_ls < best_score:
                best_score = score_ls
                best_adj = adj_ls
                
    # Fallback: Turan graph T(18,3)
    turan_adj = [0] * n
    for i in range(6):
        for j in range(6, 12):
            turan_adj[i] |= (1 << j)
            turan_adj[j] |= (1 << i)
        for j in range(12, 18):
            turan_adj[i] |= (1 << j)
            turan_adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            turan_adj[i] |= (1 << j)
            turan_adj[j] |= (1 << i)
            
    turan_score = eval_max_tf(turan_adj)
    if turan_score < best_score:
        best_score = turan_score
        best_adj = turan_adj
        
    if best_adj is None:
        best_adj = turan_adj
        
    return get_edges_from_adj(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
