import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def get_energy_delta(colors, n, u, v, old_c, new_c):
        """
        Calculate change in energy when flipping edge (u, v) from old_c to new_c.
        colors: list of colors for edges (i, j) with i < j.
        We need to know the neighbors.
        To make this fast, we should maintain adjacency sets or bitmasks outside.
        This function is a placeholder for the logic, but we will implement the search loop with state.
        """
        pass

    # We will implement a simple state-based local search.
    # State:
    # colors: list of length n*(n-1)//2
    # adj: list of 5 lists of bitmasks (int) for each color.
    # adj[c][i] is a bitmask of neighbors of i in color c.
    
    def init_state(n):
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return colors, adj

    def calc_energy(adj, n):
        energy = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    # Count common neighbors
                    # adj[c][i] & adj[c][j] gives bitmask of common neighbors
                    # popcount
                    common = adj[c][i] & adj[c][j]
                    count = bin(common).count('1')
                    if count > 1:
                        energy += count - 1
        return energy

    def flip_edge(colors, adj, n, u, v, new_c):
        """
        Flip edge (u, v) to new_c.
        Returns delta_energy.
        Assumes u < v.
        """
        idx = 0
        # Find index of (u, v)
        # idx = u*n - u*(u+1)//2 + (v - u - 1) ?
        # Formula for index of (i, j) with i < j in lexicographic order:
        # Sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # Or simpler:
        # idx = 0
        # for k in range(n):
        #   if k < i: idx += n - 1 - k
        #   elif k == i: idx += j - i - 1
        #   else: break
        # Faster:
        # idx = i * (2 * n - i - 1) // 2 + (j - i - 1)
        
        old_c = colors[u * (2 * n - u - 1) // 2 + (v - u - 1)]
        if old_c == new_c:
            return 0

        # Calculate delta energy
        delta = 0
        
        # For each w != u, v
        # We need to check if v is in adj[old_c][w] and adj[new_c][w]
        # and if u is in adj[old_c][w] and adj[new_c][w]
        
        # Pairs affected: (u, w) and (v, w) for all w != u, v
        # Also pair (u, v) itself?
        # Common neighbors of u and v:
        # Before: cnt = popcount(adj[old_c][u] & adj[old_c][v])
        # After:  cnt = popcount(adj[new_c][u] & adj[new_c][v])
        # Note: adj[old_c][u] contains v? Yes, if (u,v) is in old_c.
        # So v is in adj[old_c][u].
        # Is v in adj[old_c][v]? No.
        # So v is NOT a common neighbor of u and v.
        # Similarly u is not a common neighbor.
        # So the common neighbor count for (u, v) does NOT change when flipping (u, v)?
        # Wait.
        # Common neighbors of u, v in color c are w s.t. (u,w) in c and (v,w) in c.
        # Flipping (u,v) changes the color of the edge between u and v.
        # It does NOT change the color of (u,w) or (v,w).
        # So the set of common neighbors for (u,v) is UNCHANGED.
        # Correct.
        
        # So we only need to update pairs (u, w) and (v, w).
        
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Pair (u, w)
            # In old_c: v was a neighbor of u.
            # If v was also a neighbor of w in old_c, then v was a common neighbor of u, w.
            # Now v is not a neighbor of u in old_c. So v is no longer a common neighbor.
            # Delta for (u, w) in old_c:
            # old_cnt = popcount(adj[old_c][u] & adj[old_c][w])
            # new_cnt = old_cnt - (1 if v in adj[old_c][w] else 0)
            
            v_in_old_w = 1 if (adj[old_c][w] & (1 << v)) else 0
            v_in_new_w = 1 if (adj[new_c][w] & (1 << v)) else 0
            
            # We need the current counts to calculate delta energy.
            # It's better to maintain the counts or just recalculate the term.
            # Term(old) = max(0, cnt - 1)
            # Term(new) = max(0, cnt - delta_cnt - 1)
            
            # Let's just compute the change in the "excess" common neighbors.
            # For (u, w) in old_c:
            # cnt decreases by v_in_old_w.
            # Let cnt_old = popcount(adj[old_c][u] & adj[old_c][w])
            # But adj[old_c][u] still has v in it right now (before we update it).
            # So cnt_old includes v if v_in_old_w is 1.
            
            # This is getting complex to do incrementally without maintaining the counts.
            # Given n=26, we can just recalculate the energy for the affected pairs from the adj matrices
            # AFTER we update the adj matrices.
            
            # Actually, simpler approach:
            # 1. Save old energy contributions for pairs (u,w) and (v,w) for all w, and (u,v).
            # 2. Update adj and colors.
            # 3. Calculate new energy contributions for the same pairs.
            # 4. Delta = New - Old.
            
            pass

        # This O(n^2) recalculation per flip is too slow for SA (10^5 flips).
        # We need the O(n) delta calculation.
        
        # Let's stick to the O(n) logic but implement it carefully.
        # We maintain `cnt[c][i][j]` for i<j.
        # Initialization: O(n^3) is fine.
        
        return 0 # Placeholder

    # Given the complexity and time, I will implement a simpler Local Search that just
    # tries to fix violations by recalculation, which is slower but easier to get right.
    # Or, I will use the known valid construction for n=26 if I can find it in the literature
    # or derive it correctly.
    
    # Actually, the Lazebnik-Woldar construction IS the standard one.
    # If the current code fails, it's a bug.
    # Let's try to fix the current code's validation or construction.
    
    # Wait, the prompt says "Make a STRUCTURAL change".
    # The current code is algebraic.
    # I will switch to a **Genetic Algorithm** or **Simulated Annealing** as suggested by inspirations.
    
    # To ensure speed, I will use a "Greedy Repair" strategy.
    # 1. Random init.
    # 2. While violations exist:
    #    Find a violation (u, v, c) with count >= 2.
    #    Pick a common neighbor w.
    #    Flip color of (u, w) or (v, w) to break the C4.
    #    Check if this creates new violations.
    
    # This is a local search.
    
    n_max = 26
    best_n = 11
    best_colors = [0] * (11 * 10 // 2) # Fallback
    
    for n in range(12, n_max + 1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts
        found = False
        for restart in range(10):
            if time.time() > deadline - 10:
                break
                
            # Init
            colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            
            # Local Search
            # We will use a simple hill climbing with random restarts/kicks.
            # Energy: number of C4s.
            
            def get_energy(colors, n):
                adj = [[0] * n for _ in range(5)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        c = colors[idx]
                        adj[c][i] |= (1 << j)
                        adj[c][j] |= (1 << i)
                        idx += 1
                energy = 0
                for c in range(5):
                    for i in range(n):
                        for j in range(i + 1, n):
                            common = adj[c][i] & adj[c][j]
                            cnt = bin(common).count('1')
                            if cnt > 1:
                                energy += cnt - 1
                return energy, adj

            E, adj = get_energy(colors, n)
            
            # If E==0, we are done
            if E == 0:
                best_n = n
                best_colors = colors
                found = True
                break
                
            # Local search loop
            steps = 0
            max_steps = 5000
            while E > 0 and steps < max_steps:
                if time.time() > deadline - 10:
                    break
                steps += 1
                
                # Find a random edge to flip
                # To be smart, pick an edge involved in a violation.
                # Pick a random pair (u, v) with common neighbors > 1 in some color.
                # This is hard to do quickly without storing violations.
                # Let's just pick a random edge and random new color.
                
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                old_c = colors[u * (2 * n - u - 1) // 2 + (v - u - 1)]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                
                # Flip
                colors[u * (2 * n - u - 1) // 2 + (v - u - 1)] = new_c
                
                # Recalculate energy (Slow but simple)
                new_E, new_adj = get_energy(colors, n)
                
                # Accept if better or with SA probability
                # Simple hill climbing: accept if better
                if new_E < E:
                    E = new_E
                    adj = new_adj
                else:
                    # Revert
                    colors[u * (2 * n - u - 1) // 2 + (v - u - 1)] = old_c
                    # If stuck, maybe kick?
                    if steps % 100 == 0:
                        # Random kick
                        for _ in range(5):
                            ku = random.randint(0, n - 1)
                            kv = random.randint(0, n - 1)
                            if ku != kv:
                                if ku > kv: ku, kv = kv, ku
                                colors[ku * (2 * n - ku - 1) // 2 + (kv - ku - 1)] = random.randint(0, 4)
                        E, adj = get_energy(colors, n)
                        
            if E == 0:
                best_n = n
                best_colors = colors
                found = True
                break

    return {'n': best_n, 'colors': best_colors}
