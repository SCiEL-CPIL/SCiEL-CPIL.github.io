import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (or 150s if evaluator is stricter)
    
    # --- Data Structures ---
    # Use bitmasks for adjacency for fast intersection checks
    adj = [0] * n
    degs = [0] * n
    
    def get_edges_from_adj():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def calc_score():
        """Calculate the maximum triangle degree sum."""
        max_sum = 0
        for u in range(n):
            if adj[u] == 0:
                continue
            # Iterate over neighbors v > u to avoid double counting triangles
            # We need common neighbors w such that w > v? No, just distinct.
            # For efficiency, iterate u < v, then common = adj[u] & adj[v] & ~(1<<u) & ~(1<<v)
            # Actually adj[u] doesn't contain u. adj[v] doesn't contain v.
            # So common = adj[u] & adj[v] is the set of w connected to both.
            # We just need to ensure w is distinct from u and v, which is true for simple graphs.
            # To avoid double counting triangles (u,v,w), (u,w,v), etc., we can just check all triples?
            # No, score is max over all triangles. A triangle is a set {u,v,w}.
            # If we iterate u < v, and find common neighbor w, we have a triangle.
            # The sum is deg[u]+deg[v]+deg[w].
            # We don't need to enforce w > v for the *score* calculation, as long as we visit the triangle.
            # However, visiting it multiple times is fine for max, just slower.
            # Let's optimize by iterating u < v, and checking common neighbors.
            mask_v = adj[u]
            # Only consider v > u
            mask_v &= ~((1 << (u + 1)) - 1) # Clear bits 0..u
            while mask_v:
                lsb = mask_v & -mask_v
                v = lsb.bit_length() - 1
                mask_v ^= lsb
                
                common = adj[u] & adj[v]
                # Iterate over common neighbors w
                while common:
                    lsb_w = common & -common
                    w = lsb_w.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    common ^= lsb_w
        return max_sum

    def make_swap(u1, v1, u2, v2):
        """Remove edge (u1,v1), add edge (u2,v2). Updates adj and degs in place."""
        # Remove
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        degs[u1] -= 1
        degs[v1] -= 1
        
        # Add
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        degs[u2] += 1
        degs[v2] += 1

    def unmake_swap(u1, v1, u2, v2):
        """Revert the swap."""
        # Remove added
        adj[u2] &= ~(1 << v2)
        adj[v2] &= ~(1 << u2)
        degs[u2] -= 1
        degs[v2] -= 1
        
        # Add removed
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)
        degs[u1] += 1
        degs[v1] += 1

    def random_graph_init():
        """Initialize a random graph with exactly target_edges edges."""
        for i in range(n):
            adj[i] = 0
            degs[i] = 0
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs[:target_edges]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1

    # --- Search Strategy ---
    # We combine the robust random restarts of Inspiration 1 with the SA logic of the Current Program.
    # Key improvement: Use bitmasks for O(1) neighbor intersection checks, which is much faster than list-based checks.
    # This allows for many more iterations within the time limit.
    
    best_score = float('inf')
    best_adj = [0] * n
    best_degs = [0] * n
    best_edges = None
    
    # We will run several restarts.
    # Each restart: Random init -> Local Search (SA) -> Record best.
    
    # Pre-compute list of all possible edges for random selection?
    # There are 16*15/2 = 120 edges.
    all_edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    num_restarts = 20 # Will break early if time runs out
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Random Initialization
        random_graph_init()
        current_score = calc_score()
        
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
            best_degs = degs[:]
            best_edges = get_edges_from_adj()
            
        # 2. Local Search (Simulated Annealing)
        # Start with a moderate temperature, cool down.
        T = 2.0
        cooling = 0.995
        no_improve = 0
        max_no_improve = 2000 # Reset or stop if stuck too long
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Pick random edge to remove
            # We need a list of current edges to pick from efficiently?
            # Or just pick random u,v and check if edge exists.
            # Since graph is dense (~65/120 = 54%), picking random pair and checking is okay.
            # But we need to pick a NON-edge to add.
            
            # Pick edge (u1, v1) to remove
            while True:
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 < v1 and (adj[u1] & (1 << v1)):
                    break
            
            # Pick non-edge (u2, v2) to add
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 < v2 and not (adj[u2] & (1 << v2)):
                    # Ensure we don't pick the same edge we just removed (it's now a non-edge, but we want a different one usually? 
                    # Actually, swapping an edge with itself is a no-op. 
                    # If (u2,v2) == (u1,v1), we revert. 
                    # It's better to ensure they are distinct to make progress.
                    if u2 == u1 and v2 == v1:
                        continue
                    break
            
            # Apply swap
            make_swap(u1, v1, u2, v2)
            new_score = calc_score()
            
            delta = new_score - current_score
            
            if delta <= 0:
                # Improvement or equal
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
                    best_degs = degs[:]
                    best_edges = get_edges_from_adj()
            else:
                # Worse
                # Accept with probability exp(-delta / T)
                if T > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                        no_improve = 0 # Reset counter on accepted worse move? 
                        # Usually in SA, we reset on any accepted move.
                    else:
                        # Revert
                        unmake_swap(u1, v1, u2, v2)
                        no_improve += 1
                else:
                    # Revert
                    unmake_swap(u1, v1, u2, v2)
                    no_improve += 1
            
            T *= cooling
            
            # Periodic check for best
            # (Already done in improvement branch)
            
        # End of local search for this restart
        
    # Fallback if no valid graph found (should not happen)
    if best_edges is None:
        # Construct seed graph
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': n, 'edges': edges}
        
    return {'n': n, 'edges': best_edges}
