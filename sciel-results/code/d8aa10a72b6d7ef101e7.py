import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute all triples of vertices for penalty calculation
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
                
    def get_penalty(masks):
        """
        Calculate the penalty for K3,3 subgraphs.
        Penalty is sum over all triples {u,v,w} of C(|N(u)&N(v)&N(w)|, 3).
        If penalty is 0, the graph is K3,3-free.
        """
        penalty = 0
        for u, v, w in triples:
            common = masks[u] & masks[v] & masks[w]
            # Count bits
            c = bin(common).count('1')
            if c >= 3:
                # Number of K3,3s with this core is C(c, 3)
                # C(3,3)=1, C(4,3)=4, C(5,3)=10...
                if c == 3:
                    penalty += 1
                elif c == 4:
                    penalty += 4
                elif c == 5:
                    penalty += 10
                elif c == 6:
                    penalty += 20
                elif c == 7:
                    penalty += 35
                elif c == 8:
                    penalty += 56
                else:
                    # Rare for n=20 to have >8 common neighbors in a K3,3-free context if we are optimizing
                    # But just in case:
                    penalty += math.comb(c, 3)
        return penalty

    def get_energy(masks, num_edges, penalty_weight=1000):
        """
        Energy = -num_edges + penalty_weight * penalty
        We want to minimize energy.
        """
        return -num_edges + penalty_weight * get_penalty(masks)

    def flip_edge(masks, u, v):
        """
        Returns new masks after flipping edge (u,v).
        Also returns the change in edge count.
        """
        # We need to update masks for u and v
        # To be efficient, we should probably work with a list of ints and update in place 
        # or use a copy-on-write strategy if copying is fast enough.
        # n=20, masks is list of 20 ints. Copying is very fast.
        new_masks = masks[:]
        bit_v = 1 << v
        bit_u = 1 << u
        
        if new_masks[u] & bit_v:
            # Edge exists, remove it
            new_masks[u] ^= bit_v
            new_masks[v] ^= bit_u
            return new_masks, -1
        else:
            # Edge does not exist, add it
            new_masks[u] |= bit_v
            new_masks[v] |= bit_u
            return new_masks, 1

    # Initialize with Turan Graph T(20, 3)
    # Parts: 7, 7, 6
    # This is a known good starting point.
    def init_turan(k):
        parts = [[] for _ in range(k)]
        for i in range(n):
            parts[i % k].append(i)
        masks = [0] * n
        for i in range(k):
            for j in range(i + 1, k):
                for u in parts[i]:
                    for v in parts[j]:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
        return masks

    best_masks = None
    best_energy = float('inf')
    best_edge_count = 0
    
    # Run multiple restarts with different initializations
    restarts = 5
    
    for r in range(restarts):
        if time.time() - start_time >= time_limit:
            break
            
        if r == 0:
            current_masks = init_turan(3)
        elif r == 1:
            current_masks = init_turan(4)
        elif r == 2:
            current_masks = init_turan(5)
        else:
            # Random dense graph
            current_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.6:
                        current_masks[i] |= (1 << j)
                        current_masks[j] |= (1 << i)
        
        current_edges = sum(bin(m).count('1') for m in current_masks) // 2
        current_penalty = get_penalty(current_masks)
        current_energy = get_energy(current_masks, current_edges)
        
        if current_penalty == 0:
            if current_edges > best_edge_count:
                best_edge_count = current_edges
                best_masks = current_masks[:]
        
        # Simulated Annealing parameters
        T_start = 50.0
        T_end = 0.1
        iterations = 50000
        steps_per_iter = 100 # Total 5 million steps per restart? Might be too slow.
        # Let's adjust: 50k iterations, 1 step each? No, SA needs many steps.
        # Let's do 200,000 steps per restart.
        total_steps = 200000
        decay = (T_end / T_start) ** (1.0 / total_steps)
        
        T = T_start
        last_improve_step = 0
        local_best_energy = current_energy
        local_best_masks = current_masks[:]
        local_best_edges = current_edges
        
        for step in range(total_steps):
            if time.time() - start_time >= time_limit:
                break
                
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add or Remove single edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                new_masks, delta_edges = flip_edge(current_masks, u, v)
                new_penalty = get_penalty(new_masks)
                new_energy = get_energy(new_masks, current_edges + delta_edges)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    current_masks = new_masks
                    current_edges += delta_edges
                    current_energy = new_energy
                    
                    if current_penalty == 0 and current_edges > local_best_edges:
                        local_best_energy = current_energy
                        local_best_masks = current_masks[:]
                        local_best_edges = current_edges
                        last_improve_step = step
                        
            elif move_type < 0.8:
                # Swap: Remove (u,v), Add (w,z)
                # Find an existing edge to remove
                # Efficiently pick a random edge:
                # Sum of degrees is 2*E. Pick a random vertex u, then random neighbor v.
                # Or just pick random u, v and check if edge exists.
                # For n=20, picking random u,v is fast enough.
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if not (current_masks[u] & (1 << v)):
                    # Edge (u,v) doesn't exist, so we can't remove it. 
                    # Let's try to pick a valid edge to remove.
                    # Retry logic or just skip this step?
                    # Let's just skip to next step to keep it simple, 
                    # or pick another pair.
                    # To be efficient, let's just try a few times.
                    found = False
                    for _ in range(5):
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u != v and (current_masks[u] & (1 << v)):
                            found = True
                            break
                    if not found:
                        continue
                
                # Now (u,v) exists. Remove it.
                new_masks, _ = flip_edge(current_masks, u, v)
                
                # Now pick (w,z) to add. It must not be (u,v) and not exist.
                w = random.randint(0, n - 1)
                z = random.randint(0, n - 1)
                if w == z or (w == u and z == v) or (w == v and z == u):
                    continue
                # Check if (w,z) exists in new_masks
                if new_masks[w] & (1 << z):
                    continue
                
                # Add (w,z)
                new_masks, _ = flip_edge(new_masks, w, z)
                
                # Edge count is unchanged
                new_penalty = get_penalty(new_masks)
                new_energy = get_energy(new_masks, current_edges)
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                    current_masks = new_masks
                    current_energy = new_energy
                    # Edges count same
                    
                    if new_penalty == 0 and current_edges > local_best_edges:
                        local_best_energy = current_energy
                        local_best_masks = current_masks[:]
                        local_best_edges = current_edges
                        last_improve_step = step
                        
            else:
                # 2-flip: Remove (u,v), Add (w,z) where (w,z) might be (u,v) reversed? No, distinct.
                # Actually, let's just stick to Add/Remove and Swap.
                pass

            T *= decay
            
            # Optional: Reheat if stuck?
            # If no improvement for a long time, increase T slightly?
            # For now, standard cooling.
            
        # End of restart
        if local_best_edges > best_edge_count:
            best_edge_count = local_best_edges
            best_masks = local_best_masks[:]
            
    # Convert best_masks to list of edges
    if best_masks is None:
        # Fallback: Return empty or a trivial valid graph
        return []
        
    edges = []
    for u in range(n):
        mask = best_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                edges.append([u, v])
                
    return edges
