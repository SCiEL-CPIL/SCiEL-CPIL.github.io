"""
Strategy:
1. Use a fast bitset-based verification (O(n^3) with integer bitmasks) to validate tournaments.
2. Use a genetic algorithm with local search to explore the space of tournaments.
   - The genetic algorithm maintains a population of tournaments.
   - Selection: tournament selection based on violation count.
   - Crossover: uniform crossover between two parents.
   - Mutation: random arc flips.
   - Local search: hill-climbing on the violation count.
3. Start from the known n=67 Paley tournament (which is valid) and try to "compress" it to n=66
   by removing a vertex and re-optimizing the remaining structure, or by starting from a random
   n=66 tournament and aggressively reducing violations.
4. The key insight is that if f(4) < 67, then there exists a valid tournament on n < 67 vertices.
   We can try to find one by starting from a near-valid structure and using local search.

   A different approach: Use the fact that the Paley tournament on 67 vertices is valid.
   Try to find a valid tournament on n=66 by starting from the Paley tournament on 67,
   removing one vertex, and then checking if the remaining 66-vertex tournament is valid.
   If not, try flipping arcs to fix violations.

   Actually, a more promising approach: Start with a random tournament on n=66, 
   and use a local search that counts violations and tries to reduce them.
   The violation count can be computed efficiently using bitsets.

   Even better: Use the structure of the problem. A 4-set is bad if it has score sequence (2,2,1,1).
   We can count the number of bad 4-sets and try to minimize it.

   For n=66, the number of 4-sets is C(66,4) = 66*65*64*63/24 = ~740k.
   We can compute the violation count in O(n^3) using bitsets.
"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _paley_orientation(p):
    """Return orientation list for Paley tournament on p vertices."""
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
                # i -> j
                outbits[i] |= (1 << j)
            else:
                # j -> i
                outbits[j] |= (1 << i)
    return outbits

def _outbits_to_orient(outbits, n):
    """Convert out-degree bitmasks to orientation list."""
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            orient.append(0 if (outbits[i] >> j) & 1 else 1)
    return orient

def _count_violations(outbits, n):
    """Count violating 4-sets. A 4-set is bad if no vertex in it dominates the other 3."""
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
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

def _find_violation(outbits, n):
    """Find one violating 4-set, or None."""
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    ma = outbits[a]
                    if not ((ma >> b) & 1 and (ma >> c) & 1 and (ma >> d) & 1):
                        mb = outbits[b]
                        if not ((mb >> a) & 1 and (mb >> c) & 1 and (mb >> d) & 1):
                            mc = outbits[c]
                            if not ((mc >> a) & 1 and (mc >> b) & 1 and (mc >> d) & 1):
                                md = outbits[d]
                                if not ((md >> a) & 1 and (md >> b) & 1 and (md >> c) & 1):
                                    return (a, b, c, d)
    return None

def _flip_arc(outbits, u, v):
    """Flip the arc between u and v in-place. Returns the new outbits."""
    # Determine current direction
    if (outbits[u] >> v) & 1:
        # u -> v, flip to v -> u
        outbits[u] &= ~(1 << v)
        outbits[v] |= (1 << u)
    else:
        # v -> u, flip to u -> v
        outbits[v] &= ~(1 << u)
        outbits[u] |= (1 << v)
    return outbits

def _remove_vertex(outbits, n, rem):
    """Remove vertex rem, return new outbits for n-1 vertices."""
    new_n = n - 1
    new_outbits = [0] * new_n
    for i in range(n):
        if i == rem:
            continue
        i_new = i if i < rem else i - 1
        mask = outbits[i]
        # Build new mask: for each j != rem, map to new index
        new_mask = 0
        for j in range(n):
            if j == rem:
                continue
            if (mask >> j) & 1:
                j_new = j if j < rem else j - 1
                new_mask |= (1 << j_new)
        new_outbits[i_new] = new_mask
    return new_outbits

def _local_search(outbits, n, time_limit=10.0, max_steps=50000):
    """
    Local search with simulated annealing to minimize violations.
    Returns (success, outbits).
    """
    start_time = time.time()
    cur = outbits[:]
    cur_viol = _count_violations(cur, n)
    
    if cur_viol == 0:
        return True, cur
    
    best = cur[:]
    best_viol = cur_viol
    
    # Temperature schedule
    T = float(cur_viol)
    T_min = 0.1
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Find a violation to guide the flip
        viol = _find_violation(cur, n)
        if viol is None:
            return True, cur
        
        a, b, c, d = viol
        pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        
        # Try all 6 flips in the violating set
        best_flip = None
        best_new_viol = cur_viol
        
        for u, v in pairs:
            old_u = cur[u]
            old_v = cur[v]
            _flip_arc(cur, u, v)
            new_viol = _count_violations(cur, n)
            # Restore
            cur[u] = old_u
            cur[v] = old_v
            
            if new_viol < best_new_viol:
                best_new_viol = new_viol
                best_flip = (u, v)
        
        if best_flip is not None:
            u, v = best_flip
            _flip_arc(cur, u, v)
            cur_viol = best_new_viol
            if cur_viol < best_viol:
                best_viol = cur_viol
                best = cur[:]
            if cur_viol == 0:
                return True, cur
        else:
            # No improving move; accept a random flip with probability based on temperature
            u, v = random.choice(pairs)
            old_u = cur[u]
            old_v = cur[v]
            _flip_arc(cur, u, v)
            new_viol = _count_violations(cur, n)
            
            # Accept if better, or with probability exp(-(new_viol - cur_viol) / T)
            delta = new_viol - cur_viol
            if delta < 0 or random.random() < (T / (T + abs(delta) + 1e-10)):
                cur_viol = new_viol
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best = cur[:]
                if cur_viol == 0:
                    return True, cur
            else:
                # Reject: restore
                cur[u] = old_u
                cur[v] = old_v
            
            # Cool down
            T = max(T * 0.999, T_min)
        
        # Periodic restart: if stuck, try a few random flips
        if step % 5000 == 4999 and cur_viol > 0:
            for _ in range(5):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    _flip_arc(cur, u, v)
            cur_viol = _count_violations(cur, n)
            if cur_viol < best_viol:
                best_viol = cur_viol
                best = cur[:]
            if cur_viol == 0:
                return True, cur
    
    if best_viol == 0:
        return True, best
    return False, best

def _genetic_algorithm(n, pop_size=20, generations=100, time_limit=60.0):
    """
    Genetic algorithm with local search to find a valid tournament on n vertices.
    Returns (success, outbits).
    """
    start_time = time.time()
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        orient = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        outbits = _orient_to_outbits(orient, n)
        population.append(outbits)
    
    # Evaluate initial population
    scores = [_count_violations(ind, n) for ind in population]
    
    # Track best
    best_idx = scores.index(min(scores))
    best_outbits = population[best_idx][:]
    best_score = scores[best_idx]
    
    if best_score == 0:
        return True, best_outbits
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        # Selection: tournament selection
        selected = []
        for _ in range(pop_size):
            # Pick 3 random individuals, select the best
            candidates = random.sample(range(pop_size), 3)
            best_candidate = max(candidates, key=lambda i: -scores[i])
            selected.append(population[best_candidate][:])
        
        # Crossover and mutation
        new_population = []
        for i in range(pop_size):
            if random.random() < 0.1:
                # Keep elite
                new_population.append(selected[i][:])
            else:
                # Crossover
                parent1 = selected[i]
                parent2 = selected[(i + 1) % pop_size]
                child = []
                for j in range(n):
                    for k in range(j + 1, n):
                        if random.random() < 0.5:
                            child.append(parent1[j] >> k & 1)
                        else:
                            child.append(parent2[j] >> k & 1)
                
                # Convert to outbits
                child_outbits = [0] * n
                idx = 0
                for j in range(n):
                    for k in range(j + 1, n):
                        if child[idx]:
                            child_outbits[j] |= (1 << k)
                        else:
                            child_outbits[k] |= (1 << j)
                        idx += 1
                
                # Mutation: flip a few random arcs
                for _ in range(3):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        _flip_arc(child_outbits, u, v)
                
                new_population.append(child_outbits)
        
        # Local search on some individuals
        for i in range(pop_size):
            if time.time() - start_time > time_limit - 5:
                break
            success, improved = _local_search(new_population[i], n, time_limit=0.5, max_steps=1000)
            if success:
                return True, improved
            new_population[i] = improved
        
        # Evaluate new population
        new_scores = [_count_violations(ind, n) for ind in new_population]
        
        # Track best
        best_idx = new_scores.index(min(new_scores))
        if new_scores[best_idx] < best_score:
            best_score = new_scores[best_idx]
            best_outbits = new_population[best_idx][:]
            if best_score == 0:
                return True, best_outbits
        
        # Update population
        population = new_population
        scores = new_scores
    
    if best_score == 0:
        return True, best_outbits
    return False, best_outbits

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # First, verify the known n=67 Paley tournament
    p = 67
    n, orient = _paley_orientation(p)
    outbits = _orient_to_outbits(orient, n)
    
    # Verify 67
    viol = _find_violation(outbits, n)
    if viol is None:
        # n=67 is valid. Now try to find n < 67.
        # Strategy: Try n=66 with genetic algorithm + local search.
        for target_n in [66, 65, 64]:
            if time.time() > deadline - 30:
                break
            
            # Try genetic algorithm
            success, repaired_outbits = _genetic_algorithm(target_n, pop_size=15, generations=50, time_limit=20.0)
            if success:
                new_orient = _outbits_to_orient(repaired_outbits, target_n)
                return {'n': target_n, 'orientation': new_orient}
            
            # Also try: take the n=67 Paley tournament, remove one vertex, and local search
            for rem_vertex in range(67):
                if time.time() > deadline - 15:
                    break
                
                # Create a 66-vertex tournament by removing rem_vertex
                new_outbits = _remove_vertex(outbits, 67, rem_vertex)
                new_n = 66
                
                # Local search on this 66-vertex tournament
                success, repaired_outbits = _local_search(new_outbits, new_n, max_steps=5000, time_limit=5.0)
                if success:
                    new_orient = _outbits_to_orient(repaired_outbits, new_n)
                    return {'n': new_n, 'orientation': new_orient}
        
        # Return the verified n=67
        return {'n': n, 'orientation': orient}
    else:
        # If 67 fails (shouldn't happen), try other primes
        for p in [71, 79, 83]:
            if time.time() > deadline:
                break
            n, orient = _paley_orientation(p)
            outbits = _orient_to_outbits(orient, n)
            if _find_violation(outbits, n) is None:
                return {'n': n, 'orientation': orient}
        
        # Fallback
        n, orient = _paley_orientation(83)
        return {'n': n, 'orientation': orient}

# EVOLVE-BLOCK-END
