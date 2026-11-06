import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (10s buffer)
    
    # We aim for n=35 first. If found, return immediately.
    # If not found in time, we might try n=34 or fallback.
    # Given the seed score is 1.0 (likely n=17 or similar small valid graph, or just 1.0 baseline), 
    # and the task says "Exoo's 35-vertex graph is the published record", 
    # a valid n=35 graph is the target.
    
    # Actually, the seed program returns n=17 as fallback. 
    # The score is 1.0. If I return n=35, score should be 35/17 ~ 2.05? 
    # Wait, the prompt says "search score is value/seed". 
    # If seed value is 17 (from Paley), and I return 35, score is 35/17.
    # But the prompt says "score 1.0000" for the current program. 
    # This implies the "seed" value in the denominator is likely the value of the *current* program's output?
    # Or is "seed" a fixed baseline? 
    # "The evaluator measures the raw value (higher is better). The search score is value/seed"
    # Usually in these tasks, "seed" refers to the initial value provided by the seed program.
    # The current program returns n=17 in fallback. If it finds n=35, it returns 35.
    # If the current program's *best* output is n=17 (because it fails to find 35), then seed=17.
    # If I find 35, score = 35/17 = 2.05.
    # If I find 36, score = 36/17 = 2.11.
    
    # Let's focus on finding a valid graph for n=35.
    
    n_target = 35
    
    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def build_masks(adj, n):
        masks = [0] * n
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    m |= (1 << j)
            masks[i] = m
        return masks

    def build_comp_masks(masks, n):
        comp_masks = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            # Complement: not self, and not in adj
            # adj_mask has bits for neighbors.
            # self bit is not in adj_mask (usually).
            # So comp_mask[i] = full_mask & ~(masks[i] | (1 << i))
            comp_masks[i] = full_mask & ~(masks[i] | (1 << i))
        return comp_masks

    def find_k4(masks, n):
        """Returns a list of 4 vertices forming a K4, or None."""
        for i in range(n):
            mi = masks[i]
            for j in range(i + 1, n):
                if mi & (1 << j):
                    mj = masks[j]
                    common = mi & mj
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    
                    c_mask = common
                    while c_mask:
                        lsb = c_mask & -c_mask
                        k = lsb.bit_length() - 1
                        c_mask ^= lsb
                        rest = common & ~(1 << k)
                        if masks[k] & rest:
                            r_mask = masks[k] & rest
                            lsb2 = r_mask & -r_mask
                            l = lsb2.bit_length() - 1
                            return [i, j, k, l]
        return None

    def find_i6(comp_masks, n):
        """Returns a list of 6 vertices forming an I6 in G (K6 in complement), or None."""
        found = [None]
        
        def dfs(candidates, depth, path):
            if found[0]:
                return
            if depth == 6:
                found[0] = path[:]
                return
            if bin(candidates).count('1') < (6 - depth):
                return
                
            c_mask = candidates
            while c_mask:
                if found[0]:
                    return
                lsb = c_mask & -c_mask
                v = lsb.bit_length() - 1
                c_mask ^= lsb
                path.append(v)
                new_candidates = c_mask & comp_masks[v]
                dfs(new_candidates, depth + 1, path)
                path.pop()
                
        dfs((1 << n) - 1, 0, [])
        return found[0]

    def check_and_fix(adj, n, time_limit):
        """
        Attempts to repair the graph adj to be valid.
        Returns True if valid, False otherwise.
        Modifies adj in place.
        """
        # Pre-compute masks initially
        masks = build_masks(adj, n)
        comp_masks = build_comp_masks(masks, n)
        
        # Check initial state
        k4 = find_k4(masks, n)
        if k4:
            pass # Handle in loop
        i6 = find_i6(comp_masks, n)
        if not k4 and not i6:
            return True
            
        # Repair Loop
        # We will alternate between fixing K4s and I6s.
        # To avoid oscillation, we keep a history of recent flips? 
        # Simple strategy: Randomly pick a violation, flip a random edge involved.
        
        # To speed up, we can try to be smart about which edge to flip.
        # For K4 [a,b,c,d], removing any edge breaks it. 
        # For I6 [a,b,c,d,e,f], adding any edge breaks it.
        
        # We recompute masks only when necessary? 
        # Flipping one edge (u,v) changes masks[u] and masks[v].
        # It changes comp_masks[u] and comp_masks[v].
        
        # For simplicity and robustness within time, we might recompute masks every few steps 
        # or just check validity from scratch if the local update is too complex.
        # Given n=35, recomputing masks is O(n^2) = 1225 ops. Very fast.
        # Finding K4 is O(E * deg) roughly. 
        # Finding I6 is backtracking.
        
        # Let's use a "soft" check: 
        # 1. Update masks for the flipped edge.
        # 2. Try to find a violation.
        
        # Actually, finding a violation might be expensive if the graph is "close" to valid 
        # but has many small violations.
        # However, if we fix one, we might create another.
        
        # Strategy:
        # While time remains:
        #   1. Find a K4. If found, pick random edge in it, remove it. Update masks.
        #   2. Else, find an I6. If found, pick random non-edge in it, add it. Update masks.
        #   3. Else, return True.
        
        # Optimization: If we get stuck in a loop (same violations appearing), 
        # we can randomize the choice of edge or do a "kick" (flip random edges).
        
        # Let's implement this loop.
        
        last_flip_time = time.time()
        stagnation_counter = 0
        last_violation_sig = None
        
        while time.time() < time_limit:
            # Check for K4
            k4 = find_k4(masks, n)
            if k4:
                sig = ('K4', tuple(sorted(k4)))
                if sig == last_violation_sig:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    last_violation_sig = sig
                
                # If stuck, do a random kick
                if stagnation_counter > 20:
                    # Flip 5 random edges
                    for _ in range(5):
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u != v:
                            # Flip
                            if adj[u][v]:
                                adj[u][v] = 0
                                adj[v][u] = 0
                                masks[u] &= ~(1 << v)
                                masks[v] &= ~(1 << u)
                                comp_masks[u] |= (1 << v)
                                comp_masks[v] |= (1 << u)
                            else:
                                adj[u][v] = 1
                                adj[v][u] = 1
                                masks[u] |= (1 << v)
                                masks[v] |= (1 << u)
                                comp_masks[u] &= ~(1 << v)
                                comp_masks[v] &= ~(1 << u)
                    stagnation_counter = 0
                    last_violation_sig = None
                    continue
                
                # Pick random edge in K4 to remove
                edges_in_k4 = list(combinations(k4, 2))
                u, v = random.choice(edges_in_k4)
                if adj[u][v] == 1:
                    adj[u][v] = 0
                    adj[v][u] = 0
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    comp_masks[u] |= (1 << v)
                    comp_masks[v] |= (1 << u)
                # else: shouldn't happen
                
            else:
                # Check for I6
                i6 = find_i6(comp_masks, n)
                if i6:
                    sig = ('I6', tuple(sorted(i6)))
                    if sig == last_violation_sig:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                        last_violation_sig = sig
                        
                    if stagnation_counter > 20:
                        for _ in range(5):
                            u = random.randint(0, n-1)
                            v = random.randint(0, n-1)
                            if u != v:
                                if adj[u][v]:
                                    adj[u][v] = 0
                                    adj[v][u] = 0
                                    masks[u] &= ~(1 << v)
                                    masks[v] &= ~(1 << u)
                                    comp_masks[u] |= (1 << v)
                                    comp_masks[v] |= (1 << u)
                                else:
                                    adj[u][v] = 1
                                    adj[v][u] = 1
                                    masks[u] |= (1 << v)
                                    masks[v] |= (1 << u)
                                    comp_masks[u] &= ~(1 << v)
                                    comp_masks[v] &= ~(1 << u)
                        stagnation_counter = 0
                        last_violation_sig = None
                        continue

                    # Pick random non-edge in I6 to add
                    pairs = list(combinations(i6, 2))
                    u, v = random.choice(pairs)
                    if adj[u][v] == 0:
                        adj[u][v] = 1
                        adj[v][u] = 1
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        comp_masks[u] &= ~(1 << v)
                        comp_masks[v] &= ~(1 << u)
                else:
                    # No violations found
                    return True
                    
            # Periodic full re-check to ensure masks are consistent (paranoia)
            # Not strictly necessary if updates are correct, but good for safety.
            # Let's skip to save time, the updates are simple.
            
        return False

    # Multi-start strategy
    # We will run several independent repair processes in parallel? 
    # Python GIL makes true parallelism hard for CPU-bound tasks.
    # Instead, we can run them sequentially but with different initializations, 
    # allocating a time slice to each.
    
    # Or, we can just run one very robust repair process with random restarts.
    # The "stagnation kick" above acts as a restart mechanism.
    
    # Let's try 5 different random initializations, giving each 20 seconds.
    # If any succeeds, return immediately.
    
    best_n = 0
    best_edges = None
    
    # Try n=35
    n = 35
    for restart in range(10):
        if time.time() > deadline - 5:
            break
            
        # Random init with p=0.5
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    
        # Allocate time for this restart
        restart_deadline = time.time() + 10.0 # 10s per restart
        
        if check_and_fix(adj, n, restart_deadline):
            edges = get_edges(adj, n)
            return {'n': n, 'edges': edges}
            
    # If n=35 failed, try n=34
    n = 34
    for restart in range(5):
        if time.time() > deadline - 5:
            break
            
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    
        restart_deadline = time.time() + 10.0
        
        if check_and_fix(adj, n, restart_deadline):
            edges = get_edges(adj, n)
            return {'n': n, 'edges': edges}
            
    # Fallback: n=17 Paley Graph
    n_p = 17
    squares = {x * x % n_p for x in range(1, n_p)}
    p_edges = [[i, j] for i in range(n_p) for j in range(i + 1, n_p) if (j - i) % n_p in squares]
    return {'n': n_p, 'edges': p_edges}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
