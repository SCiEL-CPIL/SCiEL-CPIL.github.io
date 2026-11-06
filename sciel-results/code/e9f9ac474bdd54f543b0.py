import time
import random
import math

DEADLINE = 140  # Safety margin for 150s limit

def score_masks(masks):
    n = len(masks)
    deg = [bin(m).count('1') for m in masks]
    max_s = 0
    for i in range(n):
        mi = masks[i]
        # j > i
        mask_j = mi & ~((1 << (i + 1)) - 1)
        while mask_j:
            lsb = mask_j & -mask_j
            j = lsb.bit_length() - 1
            mask_j ^= lsb
            
            # Common neighbors of i and j
            common = mi & masks[j]
            # Exclude i and j themselves
            common &= ~(1 << i) & ~(1 << j)
            
            while common:
                lsb_k = common & -common
                k = lsb_k.bit_length() - 1
                common ^= lsb_k
                s = deg[i] + deg[j] + deg[k]
                if s > max_s:
                    max_s = s
    return max_s

def _masks_to_edges(masks, n):
    edges = []
    for i in range(n):
        mi = masks[i]
        mask_j = mi & ~((1 << (i + 1)) - 1)
        while mask_j:
            lsb = mask_j & -mask_j
            j = lsb.bit_length() - 1
            edges.append([i, j])
            mask_j ^= lsb
    return edges

def construct():
    start_time = time.time()
    n = 16
    target_edges = 65
    random.seed(42)

    # Initial Population
    pop_size = 100
    population = []
    
    # 1. Turan graph T_2(16) + 1 edge
    masks = [0] * n
    for u in range(8):
        for v in range(8, 16):
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
    # Add 1 edge within a partition
    u = random.randint(0, 7)
    v = random.randint(0, 7)
    if u != v:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    population.append(masks)
    
    # 2. Random graphs with 65 edges
    for _ in range(pop_size - 1):
        masks = [0] * n
        edges = set()
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                if (u, v) not in edges:
                    edges.add((u, v))
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
        population.append(masks)

    best_score = float('inf')
    best_masks = None
    
    scores = [score_masks(m) for m in population]
    best_idx = max(range(len(scores)), key=lambda i: -scores[i]) # Wait, min score
    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_masks = population[best_idx]
    
    generations = 0
    max_generations = 5000
    
    while time.time() - start_time < DEADLINE:
        generations += 1
        
        # Selection: Tournament
        next_pop = []
        
        # Elitism: Keep best
        next_pop.append(best_masks)
        
        while len(next_pop) < pop_size:
            # Pick 2 random parents
            i1 = random.randint(0, pop_size - 1)
            i2 = random.randint(0, pop_size - 1)
            p1 = population[i1]
            p2 = population[i2]
            
            # Crossover: Uniform on edges
            e1 = set()
            e2 = set()
            for i in range(n):
                mi = p1[i]
                mask_j = mi & ~((1 << (i + 1)) - 1)
                while mask_j:
                    lsb = mask_j & -mask_j
                    j = lsb.bit_length() - 1
                    e1.add((i, j))
                    mask_j ^= lsb
                mj = p2[i]
                mask_j = mj & ~((1 << (i + 1)) - 1)
                while mask_j:
                    lsb = mask_j & -mask_j
                    j = lsb.bit_length() - 1
                    e2.add((i, j))
                    mask_j ^= lsb
            
            child_edges = set()
            for e in e1:
                if random.random() < 0.5:
                    child_edges.add(e)
            for e in e2:
                if random.random() < 0.5:
                    child_edges.add(e)
            
            # Repair to 65 edges
            if len(child_edges) > target_edges:
                to_remove = len(child_edges) - target_edges
                for _ in range(to_remove):
                    if child_edges:
                        rem = random.choice(list(child_edges))
                        child_edges.remove(rem)
            elif len(child_edges) < target_edges:
                to_add = target_edges - len(child_edges)
                all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
                for _ in range(to_add):
                    while True:
                        e = random.choice(all_possible)
                        if e not in child_edges:
                            child_edges.add(e)
                            break
            
            # Convert back to masks
            child_masks = [0] * n
            for u, v in child_edges:
                child_masks[u] |= (1 << v)
                child_masks[v] |= (1 << u)
            
            # Mutation: Swap edge
            if random.random() < 0.5:
                if child_edges:
                    rem = random.choice(list(child_edges))
                    child_edges.remove(rem)
                    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
                    while True:
                        add = random.choice(all_possible)
                        if add not in child_edges:
                            child_edges.add(add)
                            break
                    u, v = rem
                    child_masks[u] &= ~(1 << v)
                    child_masks[v] &= ~(1 << u)
                    u, v = add
                    child_masks[u] |= (1 << v)
                    child_masks[v] |= (1 << u)
            
            next_pop.append(child_masks)
        
        population = next_pop
        
        # Evaluate
        scores = [score_masks(m) for m in population]
        best_idx = min(range(len(scores)), key=lambda i: scores[i])
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_masks = population[best_idx]
            
        # Local Search Polish
        # Take best_masks and do a few SA steps
        cur_masks = best_masks[:]
        cur_score = best_score
        T = 5.0
        for _ in range(50): # 50 steps
            if time.time() - start_time > DEADLINE - 5:
                break
            # Pick random edge to remove
            cur_edges = set()
            for i in range(n):
                mi = cur_masks[i]
                mask_j = mi & ~((1 << (i + 1)) - 1)
                while mask_j:
                    lsb = mask_j & -mask_j
                    j = lsb.bit_length() - 1
                    cur_edges.add((i, j))
                    mask_j ^= lsb
            
            if not cur_edges:
                break
            rem = random.choice(list(cur_edges))
            # Pick random non-edge to add
            all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
            non_edges = [e for e in all_possible if e not in cur_edges]
            if not non_edges:
                break
            add = random.choice(non_edges)
            
            # Apply swap
            u, v = rem
            cur_masks[u] &= ~(1 << v)
            cur_masks[v] &= ~(1 << u)
            u, v = add
            cur_masks[u] |= (1 << v)
            cur_masks[v] |= (1 << u)
            
            new_score = score_masks(cur_masks)
            delta = new_score - cur_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_masks = cur_masks[:]
            else:
                # Revert
                u, v = add
                cur_masks[u] &= ~(1 << v)
                cur_masks[v] &= ~(1 << u)
                u, v = rem
                cur_masks[u] |= (1 << v)
                cur_masks[v] |= (1 << u)
            
            T *= 0.9
            if T < 0.1:
                T = 0.1

        if best_score <= 23:
            break

    best_edges = _masks_to_edges(best_masks, n)
    return {'n': n, 'edges': best_edges}
