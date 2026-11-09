import time, math, random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    def now(): return time.time()
    
    all_m = list(range(2, 129))
    n = len(all_m)
    
    # Precompute gcd matrix
    G = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(all_m[i], all_m[j])
            G[i][j] = g
            G[j][i] = g
    
    # Precompute for each modulus, the set of moduli it's coprime with
    coprime_with = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if G[i][j] == 1:
                coprime_with[i].add(j)
                coprime_with[j].add(i)
    
    # A modulus can only be in the set if all other chosen moduli share gcd>=2 with it
    # Order by: fewer coprime neighbors first, then smaller modulus (fewer residue options)
    order = sorted(range(n), key=lambda i: (len(coprime_with[i]), all_m[i]))
    
    def find_residue(i, chosen, G, all_m):
        """Find a valid residue for modulus i given the chosen set."""
        m_i = all_m[i]
        for r in range(m_i):
            ok = True
            for (j, r_j) in chosen:
                g = G[i][j]
                if g < 2:
                    return None
                if (r % g) == (r_j % g):
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def verify_solution(chosen, G):
        for a in range(len(chosen)):
            for b in range(a + 1, len(chosen)):
                i, r = chosen[a]
                j, r2 = chosen[b]
                g = G[i][j]
                if g < 2:
                    return False
                if (r % g) == (r2 % g):
                    return False
        return True
    
    # Greedy construction
    chosen = []
    for i in order:
        if now() - t0 > DEADLINE:
            break
        r = find_residue(i, chosen, G, all_m)
        if r is not None:
            chosen.append((i, r))
    
    # Local search: try to improve by adding, swapping, or reassigning residues
    def try_add_one(chosen, G, all_m):
        included = set(i for i, _ in chosen)
        # Try candidates in order of constraint difficulty
        candidates = [i for i in order if i not in included]
        for i in candidates:
            if now() - t0 > DEADLINE:
                return None
            r = find_residue(i, chosen, G, all_m)
            if r is not None:
                return (i, r)
        return None
    
    def try_swap(chosen, G, all_m):
        """Try removing one element and adding two others."""
        for rem_idx in range(len(chosen)):
            if now() - t0 > DEADLINE:
                return None
            removed = chosen[rem_idx]
            chosen2 = chosen[:rem_idx] + chosen[rem_idx+1:]
            included = set(i for i, _ in chosen2)
            
            # Try to add two new elements
            added = []
            for i in order:
                if i in included or i == removed[0]:
                    continue
                r = find_residue(i, chosen2 + added, G, all_m)
                if r is not None:
                    added.append((i, r))
                    if len(added) == 2:
                        break
            if len(added) == 2:
                return chosen2 + added
        return None
    
    def local_search(chosen, G, all_m, max_iters=200):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if now() - t0 > DEADLINE:
                break
            improved = False
            iters += 1
            
            # Try to add one
            add = try_add_one(chosen, G, all_m)
            if add is not None:
                chosen.append(add)
                improved = True
                continue
            
            # Try swap (remove 1, add 2)
            swapped = try_swap(chosen, G, all_m)
            if swapped is not None and len(swapped) > len(chosen):
                chosen = swapped
                improved = True
        
        return chosen
    
    random.seed(42)
    chosen = local_search(chosen, G, all_m, max_iters=300)
    
    # Multiple restarts with different orderings
    best = chosen
    best_len = len(chosen)
    
    for restart in range(30):
        if now() - t0 > DEADLINE - 5:
            break
        
        # Create varied orderings
        if restart % 3 == 0:
            # Shuffle slightly
            order_r = order[:]
            random.shuffle(order_r[len(order_r)//2:])  # shuffle second half
        elif restart % 3 == 1:
            # Reverse
            order_r = order[::-1]
        else:
            # Sort by different criterion
            order_r = sorted(range(n), key=lambda i: (len(coprime_with[i]), -all_m[i]))
        
        chosen_r = []
        for i in order_r:
            if now() - t0 > DEADLINE - 5:
                break
            r = find_residue(i, chosen_r, G, all_m)
            if r is not None:
                chosen_r.append((i, r))
        
        chosen_r = local_search(chosen_r, G, all_m, max_iters=150)
        
        if len(chosen_r) > best_len:
            best = chosen_r
            best_len = len(chosen_r)
    
    # Final verification
    if not verify_solution(best, G):
        # Fallback: simple valid pair
        best = [(0, 0), (2, 1)]  # m=2 r=0, m=4 r=1
    
    result = [[r, all_m[i]] for (i, r) in best]
    return result
