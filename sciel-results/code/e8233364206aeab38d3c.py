"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm[i][j] = i * j // gcd(i, j)
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm[a][b]
            for c in range(b+1, N+1):
                if lcm[a][c] == lab and lcm[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    # For each element x, store list of (y, z) pairs that form bad triple with x
    conflicts = [[] for _ in range(N+1)]
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Precompute for each element x: a list of bitmask pairs (1<<y | 1<<z)
    # where {x, y, z} is a bad triple
    conflict_masks = [[] for _ in range(N+1)]
    for x in range(1, N+1):
        for (y, z) in conflicts[x]:
            mask = (1 << y) | (1 << z)
            conflict_masks[x].append(mask)
    
    # Precompute largest odd divisor for structural ordering
    def largest_odd_divisor(n):
        while n % 2 == 0:
            n //= 2
        return n
    
    lod = [0] + [largest_odd_divisor(i) for i in range(1, N+1)]
    
    # Group elements by largest odd divisor
    lod_groups = {}
    for x in range(1, N+1):
        lod_groups.setdefault(lod[x], []).append(x)
    
    def can_add_mask(x, S_mask):
        """Check if x can be added to set represented by S_mask (bitmask)."""
        for cm in conflict_masks[x]:
            if (S_mask & cm) == cm:
                return False
        return True
    
    # Greedy fill with bitmask
    def greedy_fill_mask(S_mask, order):
        for x in order:
            if (S_mask >> x) & 1 == 0 and can_add_mask(x, S_mask):
                S_mask |= (1 << x)
        return S_mask
    
    # QUBO-style Simulated Annealing with sparse conflict updates
    def qubo_annealing(S_mask, max_iters=8000, t_start=3.0, t_end=0.01, cooling=None):
        """QUBO-style SA: minimize energy = -|S| + penalty * bad_triples.
        Exploits sparse conflict matrix for fast energy deltas."""
        if cooling is None:
            cooling = (t_end / t_start) ** (1.0 / max_iters) if max_iters > 0 else 0.999
        current = S_mask
        current_size = current.bit_count()
        current_bad = 0
        # Count initial bad triples
        for (a, b, c) in bad_triples:
            if (current >> a) & 1 and (current >> b) & 1 and (current >> c) & 1:
                current_bad += 1
        current_energy = -current_size + 100 * current_bad
        
        best = current
        best_size = current_size
        best_energy = current_energy
        
        t = t_start
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Pick a random element to flip
            x = random.randint(1, N)
            in_set = (current >> x) & 1
            
            if in_set:
                # Remove x: compute energy delta
                new_bad = 0
                for (y, z) in conflicts[x]:
                    if (current >> y) & 1 and (current >> z) & 1:
                        new_bad += 1
                delta_size = -1
                delta_bad = -new_bad
                delta_energy = -delta_size + 100 * delta_bad
            else:
                # Add x: compute energy delta
                new_bad = 0
                for (y, z) in conflicts[x]:
                    if (current >> y) & 1 and (current >> z) & 1:
                        new_bad += 1
                delta_size = 1
                delta_bad = new_bad
                delta_energy = -delta_size + 100 * delta_bad
            
            # Metropolis acceptance
            if delta_energy <= 0:
                accept = True
            elif t > 1e-10:
                accept = random.random() < math.exp(-delta_energy / t)
            else:
                accept = False
            
            if accept:
                if in_set:
                    current = current & ~(1 << x)
                else:
                    current = current | (1 << x)
                current_size += delta_size
                current_bad += delta_bad
                current_energy += delta_energy
                
                if current_size > best_size or (current_size == best_size and current_energy < best_energy):
                    best = current
                    best_size = current_size
                    best_energy = current_energy
            
            t *= cooling
        
        return best
    
    # Local search: 1-removal and re-greedy-fill
    def local_search(S_mask, max_rounds=80):
        """Try 1-removal moves to improve."""
        current = S_mask
        current_size = current.bit_count()
        best = current
        best_size = current_size
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 3:
                break
            
            improved = False
            in_set = [x for x in range(1, N+1) if (current >> x) & 1]
            random.shuffle(in_set)
            
            for u in in_set:
                if time.time() > deadline - 3:
                    break
                
                # Remove u
                temp = current & ~(1 << u)
                
                # Re-fill greedily
                for x in range(1, N+1):
                    if (temp >> x) & 1 == 0 and can_add_mask(x, temp):
                        temp |= (1 << x)
                
                new_size = temp.bit_count()
                if new_size > current_size:
                    current = temp
                    current_size = new_size
                    if current_size > best_size:
                        best = current
                        best_size = current_size
                    improved = True
                    break
                # If same size, accept with some probability to escape local minima
                elif new_size == current_size and random.random() < 0.15:
                    current = temp
                    current_size = new_size
                    improved = True
                    break
        
        return best
    
    # High-precision polish: exhaustive 1-flip and 2-flip moves
    def high_precision_polish(S_mask, max_rounds=150):
        """Exhaustive local search with 1-flip and 2-flip moves."""
        current = S_mask
        current_size = current.bit_count()
        best = current
        best_size = current_size
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 3:
                break
            
            improved = False
            
            # Try all 1-flip moves
            for x in range(1, N+1):
                if time.time() > deadline - 3:
                    break
                in_set = (current >> x) & 1
                if in_set:
                    temp = current & ~(1 << x)
                    # Try to refill
                    for y in range(1, N+1):
                        if (temp >> y) & 1 == 0 and can_add_mask(y, temp):
                            temp |= (1 << y)
                    new_size = temp.bit_count()
                    if new_size > current_size:
                        current = temp
                        current_size = new_size
                        if current_size > best_size:
                            best = current
                            best_size = current_size
                        improved = True
                        break
                else:
                    if can_add_mask(x, current):
                        temp = current | (1 << x)
                        new_size = temp.bit_count()
                        if new_size > current_size:
                            current = temp
                            current_size = new_size
                            if current_size > best_size:
                                best = current
                                best_size = current_size
                            improved = True
                            break
            
            # Try 2-flip moves if no improvement
            if not improved:
                in_set = [x for x in range(1, N+1) if (current >> x) & 1]
                not_in = [x for x in range(1, N+1) if (current >> x) & 1 == 0]
                random.shuffle(in_set)
                random.shuffle(not_in)
                
                for u in in_set[:30]:
                    if time.time() > deadline - 3:
                        break
                    for v in not_in[:30]:
                        if time.time() > deadline - 3:
                            break
                        temp = (current & ~(1 << u)) | (1 << v)
                        # Check validity
                        valid = True
                        for cm in conflict_masks[v]:
                            if (temp & cm) == cm:
                                valid = False
                                break
                        if valid:
                            # Try to add more elements
                            for w in range(1, N+1):
                                if (temp >> w) & 1 == 0 and can_add_mask(w, temp):
                                    temp |= (1 << w)
                            new_size = temp.bit_count()
                            if new_size > current_size:
                                current = temp
                                current_size = new_size
                                if current_size > best_size:
                                    best = current
                                    best_size = current_size
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best
    
    # Main search loop
    best_mask = 0
    best_size = 0
    
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random restart
        S_mask = 0
        
        # Sometimes seed with a small random set
        if r % 3 == 0:
            seed_size = random.randint(1, 5)
            seed_elements = random.sample(elements, seed_size)
            for x in seed_elements:
                if can_add_mask(x, S_mask):
                    S_mask |= (1 << x)
        
        # Greedy fill with structural ordering: group by largest odd divisor
        if r % 4 == 0:
            # Order by largest odd divisor (structural heuristic)
            order = []
            for lod_val in sorted(lod_groups.keys()):
                group = lod_groups[lod_val][:]
                random.shuffle(group)
                order.extend(group)
        else:
            order = elements[:]
            random.shuffle(order)
        
        S_mask = greedy_fill_mask(S_mask, order)
        
        # Local search
        S_mask = local_search(S_mask)
        
        # QUBO-style simulated annealing to escape local minima
        S_mask = qubo_annealing(S_mask)
        
        # High-precision polish
        S_mask = high_precision_polish(S_mask)
        
        # Final polish: try to add any remaining element
        for x in elements:
            if (S_mask >> x) & 1 == 0 and can_add_mask(x, S_mask):
                S_mask |= (1 << x)
        
        # Update best
        current_size = S_mask.bit_count()
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
    
    # Convert best_mask back to list
    best_set = [x for x in range(1, N+1) if (best_mask >> x) & 1]
    
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
