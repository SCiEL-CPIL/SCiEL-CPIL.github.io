import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n = 31
    m = 91  # number of edges required
    q = 5
    
    # Build PG(2,5) points as 1D subspaces of GF(5)^3
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], q-2, q)
                vec = [(x * inv) % q for x in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    if len(points) != 31:
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        return [[i, j] for i, j in all_possible[:m]]
    
    def perp(p, qpt):
        return (p[0]*qpt[0] + p[1]*qpt[1] + p[2]*qpt[2]) % q == 0
    
    # Build adjacency for the full polarity graph
    # Use bitmasks for fast neighbor operations
    adj_mask = [0] * n
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if perp(points[i], points[j]):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                edges.append((i, j))
    
    polarity_edges = len(edges)
    
    if polarity_edges < m:
        random.seed(42)
        existing = set(edges)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for e in all_possible:
            if e not in existing:
                edges.append(e)
                if len(edges) == m:
                    break
        return [[u, v] for u, v in edges]
    
    # Helper to count C4s containing edge (u, v) in current graph defined by adj_mask
    # A C4 is u-w-x-v-u. 
    # w is neighbor of u, x is neighbor of v.
    # w != x, w != v, x != u.
    # w and x must be adjacent.
    def count_c4s_for_edge(u, v):
        count = 0
        # Get neighbors of u and v
        # We can iterate over bits
        nu = adj_mask[u]
        nv = adj_mask[v]
        
        # Remove u and v from consideration if they are in each other's masks (they are)
        # But we need w != v and x != u.
        # Also w != x.
        
        # Iterate over w in N(u)
        # Mask of neighbors of u excluding v
        nu_excl_v = nu & ~(1 << v)
        
        # Iterate over w
        w = 0
        while nu_excl_v:
            # Extract lowest set bit
            lsb = nu_excl_v & -nu_excl_v
            w = lsb.bit_length() - 1
            nu_excl_v ^= lsb
            
            # Now we need x in N(v) such that x != u, x != w, and w ~ x
            nv_excl_u = nv & ~(1 << u)
            
            # Check intersection of N(w) and N(v) excluding u and w?
            # Actually, we just need x in N(v) \ {u, w} such that w ~ x.
            
            # Let's iterate x in N(v)
            # Or better: common neighbors of w and v?
            # No, x is neighbor of v. w is neighbor of u.
            # We need x in N(v) and x in N(w).
            
            # Let's just iterate x in N(v) excluding u and w
            nv_excl_u_w = nv_excl_u & ~(1 << w)
            
            # Check which of these are neighbors of w
            # N(w) & nv_excl_u_w
            common = adj_mask[w] & nv_excl_u_w
            count += bin(common).count('1')
            
        return count

    # Helper to count C4s created by adding edge (u, v)
    # This is the same as count_c4s_for_edge(u, v) in the graph WITHOUT the edge (u,v)
    # Since (u,v) is not an edge currently, N(u) and N(v) do not include each other.
    # So the formula is identical.
    def count_c4s_created_by_adding(u, v):
        return count_c4s_for_edge(u, v)

    # Initialize state
    current_adj_mask = adj_mask[:]
    current_edges = set(edges)
    
    # Greedily remove edges to reach m
    to_remove = polarity_edges - m
    for _ in range(to_remove):
        best_edge = None
        best_count = -1
        for e in current_edges:
            u, v = e
            count = count_c4s_for_edge(u, v)
            if count > best_count:
                best_count = count
                best_edge = e
        
        if best_edge:
            current_edges.remove(best_edge)
            u, v = best_edge
            current_adj_mask[u] &= ~(1 << v)
            current_adj_mask[v] &= ~(1 << u)

    # Calculate initial C4 count
    # Sum C4s for all edges / 4 (each C4 has 4 edges)
    def total_c4_count():
        total = 0
        for u, v in current_edges:
            total += count_c4s_for_edge(u, v)
        return total // 4

    current_c4 = total_c4_count()
    best_c4 = current_c4
    best_adj_mask = current_adj_mask[:]
    best_edges = set(current_edges)

    # Precompute all possible non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local Search Loop
    # We will try to improve by swapping edges.
    # To be efficient, we can just iterate through all possible swaps in a random order or sorted order.
    # Given the small size, we can do a full pass.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Collect all current edges and non-edges
        edge_list = list(current_edges)
        non_edge_list = [e for e in all_pairs if e not in current_edges]
        
        # Calculate C4 contribution for each current edge
        # And C4 creation cost for each non-edge
        # We want to find a swap (e_out, e_in) that minimizes C4.
        # Delta = C4s_created(e_in) - C4s_removed(e_out)
        # We want Delta < 0.
        
        # To speed up, we can just pick the edge with max C4 and the non-edge with min C4 creation?
        # That's a greedy step. It might get stuck.
        # Let's try all pairs of (worst edges, best non-edges) first, then if stuck, randomize.
        
        # Sort edges by C4 count descending
        edge_c4s = [(count_c4s_for_edge(u, v), (u, v)) for u, v in edge_list]
        edge_c4s.sort(reverse=True)
        
        # Sort non-edges by C4 creation cost ascending
        non_edge_c4s = [(count_c4s_created_by_adding(u, v), (u, v)) for u, v in non_edge_list]
        non_edge_c4s.sort()
        
        # Try top K swaps
        K = 10
        best_swap = None
        best_delta = 0
        
        for c_out, e_out in edge_c4s[:K]:
            for c_in, e_in in non_edge_c4s[:K]:
                delta = c_in - c_out
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e_out, e_in)
                    
        if best_swap:
            e_out, e_in = best_swap
            u_out, v_out = e_out
            u_in, v_in = e_in
            
            # Apply swap
            current_edges.remove(e_out)
            current_edges.add(e_in)
            current_adj_mask[u_out] &= ~(1 << v_out)
            current_adj_mask[v_out] &= ~(1 << u_out)
            current_adj_mask[u_in] |= (1 << v_in)
            current_adj_mask[v_in] |= (1 << u_in)
            
            current_c4 += best_delta
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj_mask = current_adj_mask[:]
                best_edges = set(current_edges)
            improved = True
        else:
            # No improvement found with greedy top-K. 
            # Try random swaps to escape local minimum.
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge and a random non-edge
            # But to be efficient, let's just do a few random attempts
            attempts = 0
            found_improvement = False
            while attempts < 100 and time.time() < deadline:
                attempts += 1
                e_out = random.choice(edge_list)
                e_in = random.choice(non_edge_list)
                
                c_out = count_c4s_for_edge(e_out[0], e_out[1])
                c_in = count_c4s_created_by_adding(e_in[0], e_in[1])
                delta = c_in - c_out
                
                if delta < 0:
                    # Apply
                    current_edges.remove(e_out)
                    current_edges.add(e_in)
                    u_out, v_out = e_out
                    u_in, v_in = e_in
                    current_adj_mask[u_out] &= ~(1 << v_out)
                    current_adj_mask[v_out] &= ~(1 << u_out)
                    current_adj_mask[u_in] |= (1 << v_in)
                    current_adj_mask[v_in] |= (1 << u_in)
                    
                    current_c4 += delta
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_adj_mask = current_adj_mask[:]
                        best_edges = set(current_edges)
                    
                    found_improvement = True
                    # Restart the loop to re-evaluate with new state
                    break
            
            if found_improvement:
                improved = True
            else:
                # Stuck. Restart with a different random perturbation of the initial graph?
                # Or just break.
                # Let's try a few random restarts of the local search from the current state
                # by randomly swapping 2-3 edges even if it increases C4, then re-optimizing.
                # This is a simple simulated annealing step.
                
                if random.random() < 0.5:
                    # Perturb
                    for _ in range(2):
                        e_out = random.choice(list(current_edges))
                        e_in = random.choice(non_edge_list)
                        current_edges.remove(e_out)
                        current_edges.add(e_in)
                        u_out, v_out = e_out
                        u_in, v_in = e_in
                        current_adj_mask[u_out] &= ~(1 << v_out)
                        current_adj_mask[v_out] &= ~(1 << u_out)
                        current_adj_mask[u_in] |= (1 << v_in)
                        current_adj_mask[v_in] |= (1 << u_in)
                    
                    # Recalculate C4
                    current_c4 = total_c4_count()
                    
                    # Continue local search
                    improved = True
                else:
                    break

    final_edges = list(best_edges)
    
    if len(final_edges) != m:
        # Fallback
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        final_edges = all_possible[:m]
        
    result = [[u, v] for u, v in final_edges]
    return result
