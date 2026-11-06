import time
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START

def _paley_orientation(p):
    """Return (n, orientation list) for Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    n = p
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % p in squares:
                orient.append(0)
            else:
                orient.append(1)
    return n, orient

def _orient_to_outbits(orient, n):
    """Convert orientation list to list of out-degree bitmasks (Python ints)."""
    outbits = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orient[idx]
            idx += 1
            if bit == 0:
                outbits[i] |= (1 << j)
            else:
                outbits[j] |= (1 << i)
    return outbits

def _outbits_to_orient(outbits, n):
    """Convert out-degree bitmasks to orientation list."""
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            orient.append(0 if (outbits[i] >> j) & 1 else 1)
    return orient

def _count_violations_fast(outbits, n):
    """
    Count violating 4-sets using bitmask operations for speed.
    A 4-set {a,b,c,d} is valid if some vertex has arcs to the other 3.
    """
    count = 0
    for a in range(n):
        oa = outbits[a]
        for b in range(a + 1, n):
            ob = outbits[b]
            # a dominates b? a->b
            if (oa >> b) & 1:
                # a can dominate sets containing a,b and two others
                # For each pair (c,d) with c,d > b, check if a->c and a->d
                # But also need to check if b,c,d dominate
                pass
            # Check if a dominates {a,b,c,d}: need a->b, a->c, a->d
            # We'll iterate over c,d
            for c in range(b + 1, n):
                oc = outbits[c]
                if (oa >> c) & 1:
                    # a->b and a->c, check d
                    for d in range(c + 1, n):
                        od = outbits[d]
                        # Check if any of a,b,c,d dominates the set
                        # a dominates: a->b (yes), a->c (yes), a->d?
                        if (oa >> d) & 1:
                            continue  # a dominates
                        # b dominates: b->a? b->c? b->d?
                        if (ob >> a) & 1 and (ob >> c) & 1 and (ob >> d) & 1:
                            continue
                        # c dominates: c->a? c->b? c->d?
                        if (oc >> a) & 1 and (oc >> b) & 1 and (oc >> d) & 1:
                            continue
                        # d dominates: d->a? d->b? d->c?
                        if (od >> a) & 1 and (od >> b) & 1 and (od >> c) & 1:
                            continue
                        count += 1
                else:
                    # a->b but not a->c
                    for d in range(c + 1, n):
                        od = outbits[d]
                        # a cannot dominate (a->c is false)
                        # b dominates: b->a? b->c? b->d?
                        if (ob >> a) & 1 and (ob >> c) & 1 and (ob >> d) & 1:
                            continue
                        # c dominates: c->a? c->b? c->d?
                        if (oc >> a) & 1 and (oc >> b) & 1 and (oc >> d) & 1:
                            continue
                        # d dominates: d->a? d->b? d->c?
                        if (od >> a) & 1 and (od >> b) & 1 and (od >> c) & 1:
                            continue
                        count += 1
            else:
                # not a->b
                for c in range(b + 1, n):
                    oc = outbits[c]
                    for d in range(c + 1, n):
                        od = outbits[d]
                        # a cannot dominate
                        # b dominates: b->a? b->c? b->d?
                        if (ob >> a) & 1 and (ob >> c) & 1 and (ob >> d) & 1:
                            continue
                        # c dominates: c->a? c->b? c->d?
                        if (oc >> a) & 1 and (oc >> b) & 1 and (oc >> d) & 1:
                            continue
                        # d dominates: d->a? d->b? d->c?
                        if (od >> a) & 1 and (od >> b) & 1 and (od >> c) & 1:
                            continue
                        count += 1
    return count

def _count_violations(outbits, n):
    """Count violating 4-sets."""
    count = 0
    for a, b, c, d in combinations(range(n), 4):
        ma = outbits[a]
        if not ((ma >> b) & 1 and (ma >> c) & 1 and (ma >> d) & 1):
            mb = outbits[b]
            if not ((mb >> a) & 1 and (mb >> c) & 1 and (mb >> d) & 1):
                mc = outbits[c]
                if not ((mc >> a) & 1 and (mc >> b) & 1 and (mc >> d) & 1):
                    md = outbits[d]
                    if not ((md >> a) & 1 and (md >> b) & 1 and (md >> c) & 1):
                        count += 1
    return count

def _delta_violations(outbits, n, u, v):
    """
    Calculate the change in the number of violations if we flip the arc between u and v.
    Only 4-sets containing both u and v are affected.
    """
    others = [i for i in range(n) if i != u and i != v]
    
    delta = 0
    u_to_v = (outbits[u] >> v) & 1
    
    for i in range(len(others)):
        x = others[i]
        for j in range(i+1, len(others)):
            y = others[j]
            
            is_viol_before = True
            if ((outbits[u] >> v) & 1) and ((outbits[u] >> x) & 1) and ((outbits[u] >> y) & 1):
                is_viol_before = False
            if ((outbits[v] >> u) & 1) and ((outbits[v] >> x) & 1) and ((outbits[v] >> y) & 1):
                is_viol_before = False
            if ((outbits[x] >> u) & 1) and ((outbits[x] >> v) & 1) and ((outbits[x] >> y) & 1):
                is_viol_before = False
            if ((outbits[y] >> u) & 1) and ((outbits[y] >> v) & 1) and ((outbits[y] >> x) & 1):
                is_viol_before = False
            
            new_u_to_v = 1 - u_to_v
            is_viol_after = True
            if new_u_to_v and ((outbits[u] >> x) & 1) and ((outbits[u] >> y) & 1):
                is_viol_after = False
            if (1 - new_u_to_v) and ((outbits[v] >> x) & 1) and ((outbits[v] >> y) & 1):
                is_viol_after = False
            if ((outbits[x] >> u) & 1) and ((outbits[x] >> v) & 1) and ((outbits[x] >> y) & 1):
                is_viol_after = False
            if ((outbits[y] >> u) & 1) and ((outbits[y] >> v) & 1) and ((outbits[y] >> x) & 1):
                is_viol_after = False
            
            if is_viol_before and not is_viol_after:
                delta -= 1
            elif not is_viol_before and is_viol_after:
                delta += 1
                
    return delta

def _flip_arc(outbits, u, v):
    """Flip the arc between u and v in-place."""
    if (outbits[u] >> v) & 1:
        outbits[u] &= ~(1 << v)
        outbits[v] |= (1 << u)
    else:
        outbits[v] &= ~(1 << u)
        outbits[u] |= (1 << v)

def _remove_vertex(outbits, n, rem):
    """Remove vertex rem, return new outbits for n-1 vertices."""
    new_n = n - 1
    new_outbits = [0] * new_n
    for i in range(n):
        if i == rem:
            continue
        i_new = i if i < rem else i - 1
        mask = outbits[i]
        new_mask = 0
        for j in range(n):
            if j == rem:
                continue
            if (mask >> j) & 1:
                j_new = j if j < rem else j - 1
                new_mask |= (1 << j_new)
        new_outbits[i_new] = new_mask
    return new_outbits

def _crossover(outbits1, outbits2, n):
    """Uniform crossover: for each pair, take from parent1 or parent2 with 50% prob."""
    child = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                child[i] |= (outbits1[i] >> j) & 1
                if (outbits1[i] >> j) & 1:
                    child[j] |= (1 << i)
                else:
                    child[i] |= (1 << j)
                    child[j] |= (1 << i)
            else:
                if (outbits2[i] >> j) & 1:
                    child[i] |= (1 << j)
                    child[j] |= (1 << i)
                else:
                    child[j] |= (1 << i)
                    child[i] |= (1 << j)
            idx += 1
    return child

def _mutate(outbits, n, rate=0.01):
    """Bit-flip mutation on random pairs."""
    new_outbits = outbits[:]
    num_pairs = n * (n - 1) // 2
    num_mutations = int(num_pairs * rate)
    for _ in range(num_mutations):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        while u == v:
            v = random.randint(0, n - 1)
        _flip_arc(new_outbits, u, v)
    return new_outbits

def _fast_local_search(outbits, n, time_limit=10.0, max_steps=100000):
    """
    Local search with delta calculation for speed.
    """
    start_time = time.time()
    cur = outbits[:]
    cur_viol = _count_violations(cur, n)
    
    if cur_viol == 0:
        return True, cur
    
    best = cur[:]
    best_viol = cur_viol
    
    T = 1.0
    T_min = 0.01
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        while u == v:
            v = random.randint(0, n - 1)
        
        delta = _delta_violations(cur, n, u, v)
        
        if delta < 0 or random.random() < (1.0 / (1.0 + delta / T)):
            _flip_arc(cur, u, v)
            cur_viol += delta
            if cur_viol < best_viol:
                best_viol = cur_viol
                best = cur[:]
            if cur_viol == 0:
                return True, cur
            T = max(T * 0.9995, T_min)
        else:
            T = max(T * 0.9995, T_min)
            
    if best_viol == 0:
        return True, best
    return False, best

def _genetic_search(n, time_limit, pop_size=20, n_generations=100):
    """
    Genetic algorithm to find a valid tournament on n vertices.
    Uses violation count as fitness.
    """
    start_time = time.time()
    
    # Initialize population: start with random tournaments
    population = []
    for _ in range(pop_size):
        outbits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    outbits[i] |= (1 << j)
                else:
                    outbits[j] |= (1 << i)
        population.append((outbits, _count_violations(outbits, n)))
    
    # Sort by fitness
    population.sort(key=lambda x: x[1])
    best_outbits = population[0][0][:]
    best_viol = population[0][1]
    
    if best_viol == 0:
        return True, best_outbits
    
    gen = 0
    while gen < n_generations:
        if time.time() - start_time > time_limit:
            break
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top 2
        new_population.append((population[0][0][:], population[0][1]))
        if len(population) > 1:
            new_population.append((population[1][0][:], population[1][1]))
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = random.randint(0, min(5, len(population) - 1))
            idx2 = random.randint(0, min(5, len(population) - 1))
            p1 = population[idx1][0]
            p2 = population[idx2][0]
            
            # Crossover
            child = _crossover(p1, p2, n)
            
            # Mutation
            child = _mutate(child, n, rate=0.005)
            
            viol = _count_violations(child, n)
            new_population.append((child, viol))
            
            if viol == 0:
                return True, child
            if viol < best_viol:
                best_viol = viol
                best_outbits = child[:]
        
        population = new_population
        population.sort(key=lambda x: x[1])
        gen += 1
    
    if best_viol == 0:
        return True, best_outbits
    return False, best_outbits

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    random.seed(42)
    
    # Step 1: Verify the known n=67 Paley tournament
    p = 67
    n67, orient67 = _paley_orientation(p)
    outbits67 = _orient_to_outbits(orient67, n67)
    
    if _count_violations(outbits67, n67) == 0:
        pass
    else:
        return {'n': n67, 'orientation': orient67}

    # Step 2: Try to find n=66 by removing each vertex
    for rem in range(67):
        if time.time() > deadline - 30:
            break
        
        new_outbits = _remove_vertex(outbits67, 67, rem)
        new_n = 66
        
        if _count_violations(new_outbits, new_n) == 0:
            new_orient = _outbits_to_orient(new_outbits, new_n)
            return {'n': new_n, 'orientation': new_orient}
        
        success, repaired = _fast_local_search(new_outbits, new_n, time_limit=1.0, max_steps=30000)
        if success:
            new_orient = _outbits_to_orient(repaired, new_n)
            return {'n': new_n, 'orientation': new_orient}
            
    # Step 3: Try to find n=65 by removing 2 vertices
    if time.time() < deadline - 15:
        pairs = list(combinations(range(67), 2))
        random.shuffle(pairs)
        
        for rem1, rem2 in pairs[:30]:
            if time.time() > deadline - 10:
                break
            
            tmp = _remove_vertex(outbits67, 67, rem1)
            rem2_adj = rem2 if rem2 < rem1 else rem2 - 1
            if rem2_adj < 0 or rem2_adj >= 66:
                continue
                
            new_outbits = _remove_vertex(tmp, 66, rem2_adj)
            new_n = 65
            
            if _count_violations(new_outbits, new_n) == 0:
                new_orient = _outbits_to_orient(new_outbits, new_n)
                return {'n': new_n, 'orientation': new_orient}
                
            success, repaired = _fast_local_search(new_outbits, new_n, time_limit=0.3, max_steps=10000)
            if success:
                new_orient = _outbits_to_orient(repaired, new_n)
                return {'n': new_n, 'orientation': new_orient}

    # Step 4: Genetic search for n=66 and n=65
    for target_n in [66, 65]:
        if time.time() > deadline - 5:
            break
        
        remaining_time = deadline - time.time() - 2
        if remaining_time < 3:
            break
            
        success, result = _genetic_search(target_n, time_limit=remaining_time, pop_size=15, n_generations=50)
        if success:
            new_orient = _outbits_to_orient(result, target_n)
            return {'n': target_n, 'orientation': new_orient}

    # Fallback: Return 67
    return {'n': n67, 'orientation': orient67}

# EVOLVE-BLOCK-END
