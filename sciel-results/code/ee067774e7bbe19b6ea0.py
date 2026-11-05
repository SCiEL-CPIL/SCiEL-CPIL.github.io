import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are all equal. Maximize cardinality.
    
    Strategy:
    1. Precompute GCD table for fast conflict checking.
    2. Build hypergraph degree: for each vertex, count bad triples it participates in.
    3. Weighted greedy: add vertices in order of lowest hypergraph degree (fewest conflicts).
    4. Local search with 2-for-1 exchanges to escape local optima.
    5. Final polish: greedy completion pass.
    """
    N = 128
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    # Precompute gcd table (1-indexed)
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g
    
    # Precompute bad triples and hypergraph degrees
    # bad_triples[a][b] = list of c such that (a,b,c) is a bad triple
    # degree[x] = number of bad triples containing x
    bad_triples = {i: set() for i in range(1, N + 1)}
    degree = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gt[a][b]
            for c in range(b + 1, N + 1):
                if gt[a][c] == g_ab and gt[b][c] == g_ab:
                    bad_triples[a].add((b, c))
                    bad_triples[b].add((a, c))
                    bad_triples[c].add((a, b))
                    degree[a] += 1
                    degree[b] += 1
                    degree[c] += 1
    
    def can_add(x, sel):
        """True if x can be added to sel without creating a bad triple."""
        for a in sel:
            gax = gt[x][a]
            for b in sel:
                if b > a and gax == gt[x][b] == gt[a][b]:
                    return False
        return True
    
    def greedy_by_degree(order):
        """Build a valid set by adding elements in the given order."""
        sel = set()
        for x in order:
            if can_add(x, sel):
                sel.add(x)
        return sel
    
    def polish(sel):
        """Repeatedly add any excluded element that fits (greedy completion)."""
        sel = set(sel)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in sel and can_add(x, sel):
                    sel.add(x)
                    changed = True
        return sel
    
    def local_search_2for1(sel, max_iters=200):
        """
        Local search with 2-for-1 exchanges: remove 1 element, try to add 2 others.
        Also try 1-for-1 and pure adds.
        """
        sel = set(sel)
        best_size = len(sel)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            excluded = [x for x in range(1, N + 1) if x not in sel]
            
            # Try pure adds first
            for x in excluded:
                if can_add(x, sel):
                    sel.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-for-1: remove x, add y1 and y2
            best_move = None
            best_gain = 0
            
            for x in list(sel):
                temp = sel - {x}
                added = []
                for y in excluded:
                    if len(added) >= 2:
                        break
                    if can_add(y, temp | set(added)):
                        added.append(y)
                
                if len(added) >= 1:
                    new_size = len(temp) + len(added)
                    gain = new_size - len(sel)
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (x, added)
            
            if best_move is not None and best_gain > 0:
                x, added = best_move
                sel.remove(x)
                sel.update(added)
                improved = True
            elif best_move is not None and best_gain == 0:
                # Same size, but might unlock future improvements
                # Only accept with some probability to avoid cycling
                if random.random() < 0.3:
                    x, added = best_move
                    sel.remove(x)
                    sel.update(added)
                    improved = True
            
            if not improved:
                break
        
        return sel
    
    def weighted_greedy_random(rng_seed):
        """
        Weighted greedy: order vertices by a combination of degree and random noise.
        Lower degree = higher priority. Add random perturbation for diversity.
        """
        rng = random.Random(rng_seed)
        scores = []
        for x in range(1, N + 1):
            # Lower degree is better; add random noise for diversity
            score = degree[x] + rng.uniform(0, 5)
            scores.append((score, x))
        scores.sort()
        order = [x for _, x in scores]
        return greedy_by_degree(order)
    
    # Strategy 1: Greedy by increasing degree (deterministic)
    order_by_degree = sorted(range(1, N + 1), key=lambda x: degree[x])
    s1 = greedy_by_degree(order_by_degree)
    s1 = polish(s1)
    s1 = local_search_2for1(s1)
    s1 = polish(s1)
    best_set = s1
    best_size = len(s1)
    
    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]
    
    # Strategy 2: Reverse degree order
    order_by_degree_desc = sorted(range(1, N + 1), key=lambda x: -degree[x])
    s2 = greedy_by_degree(order_by_degree_desc)
    s2 = polish(s2)
    s2 = local_search_2for1(s2)
    s2 = polish(s2)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2
    
    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]
    
    # Strategy 3: Multiple randomized weighted greedy trials
    random.seed(42)
    for trial in range(150):
        if time.time() > deadline:
            break
        
        s3 = weighted_greedy_random(trial)
        s3 = polish(s3)
        s3 = local_search_2for1(s3, max_iters=100)
        s3 = polish(s3)
        if len(s3) > best_size:
            best_size = len(s3)
            best_set = s3
    
    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]
    
    # Strategy 4: Aggressive local search on best_set with more iterations
    s4 = local_search_2for1(best_set, max_iters=500)
    s4 = polish(s4)
    if len(s4) > best_size:
        best_size = len(s4)
        best_set = s4
    
    # Strategy 5: Random perturbation + local search
    for _ in range(30):
        if time.time() > deadline:
            break
        
        perturbed = set(best_set)
        num_remove = random.randint(1, min(3, len(perturbed)))
        to_remove = random.sample(list(perturbed), num_remove)
        for x in to_remove:
            perturbed.discard(x)
        
        perturbed = local_search_2for1(perturbed, max_iters=150)
        perturbed = polish(perturbed)
        if len(perturbed) > best_size:
            best_size = len(perturbed)
            best_set = perturbed
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
