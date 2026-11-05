import time
import random
import math
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are all equal. Maximize cardinality.
    
    Strategy: Simulated Annealing over the full [1..128] set.
    1. Precompute GCD table and "conflict neighbors" (pairs that share a gcd with x).
    2. Objective: Minimize (number of bad triples). Ties broken by maximizing size.
    3. Start from a random valid set (greedy).
    4. Anneal: Random flips (add/remove). Accept moves that reduce conflicts, or increase
       them with probability exp(-delta / T).
    5. This allows escaping local optima by temporarily entering invalid states.
    """
    N = 128
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    # 1. Precompute GCD table
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g
    
    # 2. Precompute conflict structure for fast delta calculation
    # For a fixed x, a bad triple (a,b,x) requires:
    #   gcd(a,x) == gcd(b,x) == gcd(a,b)
    # Let g = gcd(a,x). Then b must satisfy:
    #   gcd(b,x) == g  AND  gcd(a,b) == g
    # We can precompute for each x:
    #   groups_by_gcd_g[x][g] = list of y in [1..N] such that gcd(y,x) == g
    #   Then for a candidate a in sel, we look for b in sel such that:
    #     gcd(b,x) == gcd(a,x) AND gcd(a,b) == gcd(a,x)
    
    # To make it even faster, for each pair (a,b) we can precompute the gcd,
    # but that's 128^2/2 ~ 8k pairs, which is fine.
    # Actually, let's precompute a "conflict matrix" for each x? No, that's 128 * 128 * 128.
    
    # Let's use a different fast check:
    # conflicts(x, sel) = number of pairs (a,b) in sel such that (a,b,x) is bad.
    # We can iterate over a in sel, compute g_ax = gcd(a,x).
    # Then iterate over b in sel where b > a, check if gcd(b,x) == g_ax and gcd(a,b) == g_ax.
    # This is O(|sel|^2) per x. With |sel| ~ 80, that's 6400 ops per x.
    # For a delta calculation (adding x), we need conflicts(x, sel).
    # For removing x, we need conflicts(x, sel_without_x).
    
    # Let's try to optimize the delta calculation.
    # When adding x to S:
    #   new_conflicts = conflicts(x, S)
    #   delta_conflicts = new_conflicts
    #   delta_size = 1
    # When removing x from S:
    #   old_conflicts = conflicts(x, S \ {x})
    #   delta_conflicts = -old_conflicts
    #   delta_size = -1
    
    # To speed up conflicts(x, S), we can precompute for each x:
    #   for each g, list of y in [1..N] with gcd(y,x)=g.
    #   Then for a in S, g = gcd(a,x). We need to count b in S such that b in list[g] and gcd(a,b)=g.
    #   We can precompute for each pair (a,b) their gcd.
    
    # Let's precompute gcd_pair[a][b] for all a,b.
    gcd_pair = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(1, N+1):
            gcd_pair[i][j] = gcd(i,j)
            
    # Precompute for each x and each g, the bitmask (or list) of y with gcd(y,x)=g?
    # Bitmasks might be slow in Python for 128 bits. Lists are fine.
    # neighbors[x][g] = list of y such that gcd(x,y) == g
    neighbors = {}
    for x in range(1, N+1):
        neighbors[x] = {}
        for y in range(1, N+1):
            g = gcd(x,y)
            if g not in neighbors[x]:
                neighbors[x][g] = []
            neighbors[x][g].append(y)
            
    def count_conflicts(x, S):
        """Count number of pairs (a,b) in S such that (a,b,x) is a bad triple."""
        count = 0
        # We need to count pairs (a,b) in S with a < b such that:
        # gcd(a,x) == gcd(b,x) == gcd(a,b)
        # Let's iterate a in S.
        # For each a, let g = gcd(a,x).
        # We need to count b in S such that b != a, gcd(b,x) == g, and gcd(a,b) == g.
        # We can use the neighbors[x][g] list to check if b is in it.
        # But checking membership in a list is O(n).
        # Let's use a set for S for O(1) lookup?
        # Or, since N is small, just iterate.
        
        # Optimization: Precompute for each x, a mapping from g to a set of y?
        # neighbors_set[x][g] = set(neighbors[x][g])
        # Then check if b in neighbors_set[x][g] and gcd_pair[a][b] == g.
        
        # Let's build neighbors_set on the fly or precompute it.
        # Precomputing neighbors_set is better.
        pass # Placeholder, will use precomputed sets
        
    # Precompute neighbors_set
    neighbors_set = {}
    for x in range(1, N+1):
        neighbors_set[x] = {}
        for g, lst in neighbors[x].items():
            neighbors_set[x][g] = set(lst)
            
    def count_conflicts_fast(x, S):
        """Count bad triples involving x and two elements from S."""
        count = 0
        # Iterate over a in S
        for a in S:
            g_ax = gcd_pair[x][a]
            # We need b in S such that:
            # 1. gcd(b,x) == g_ax  => b in neighbors_set[x][g_ax]
            # 2. gcd(a,b) == g_ax
            # 3. b != a
            # 4. To avoid double counting, we can assume a < b?
            #    But S is a set, order doesn't matter.
            #    If we count all pairs (a,b), we count each bad triple twice (once for a, once for b)?
            #    No, the triple is (a,b,x). The pairs are (a,b).
            #    If we iterate a in S and b in S, we count (a,b) and (b,a).
            #    We only want distinct pairs.
            #    So we can require b > a? But S is a set, no natural order unless we sort.
            #    Or we can just count all and divide by 2?
            #    Or iterate a, and for b, only consider b > a?
            #    Let's just iterate all b in S and check b != a, then divide by 2 at the end?
            #    Or better: iterate a in S, and for b in S, if b > a.
            #    This requires S to be sorted or we check index.
            #    Since N=128, sorting S each time is O(|S| log |S|).
            #    Alternatively, just check b != a and count, then divide by 2.
            
            # Let's do the "count all and divide by 2" approach for simplicity,
            # but be careful with integer division.
            # Actually, let's just iterate b in S, if b != a:
            #   if b in neighbors_set[x][g_ax] and gcd_pair[a][b] == g_ax:
            #       count += 1
            # Then return count // 2.
            
            target_set = neighbors_set[x].get(g_ax, set())
            for b in S:
                if b == a:
                    continue
                if b in target_set and gcd_pair[a][b] == g_ax:
                    count += 1
        return count // 2

    # Greedy initialization
    def greedy_init():
        S = set()
        # Sort by degree (number of conflicts in full set) to start with "safe" nodes?
        # Or just random order?
        # Let's use the degree from the current program's logic as a hint.
        # Degree of x = number of bad triples containing x in [1..128].
        # We can compute this once.
        degrees = [0]*(N+1)
        for x in range(1, N+1):
            # Count bad triples (a,b,x) in full set
            # This is count_conflicts_fast(x, set(range(1, N+1)))
            # But that's slow. Let's precompute degrees properly.
            # A bad triple (a,b,c) has gcd(a,b)=gcd(a,c)=gcd(b,c)=g.
            # For a fixed x, how many pairs (a,b) form a bad triple with x?
            # This is exactly count_conflicts_fast(x, All).
            # Let's just compute it once.
            pass
            
        # Simpler greedy: random order, add if valid.
        # To make it deterministic and good, sort by something.
        # Let's sort by number of coprime neighbors? Or just random.
        # Let's use the "low degree" heuristic from before.
        # Compute degree[x] = number of bad triples containing x.
        # We can compute this by summing count_conflicts_fast(x, S) for S=All?
        # No, count_conflicts_fast(x, S) counts pairs in S.
        # If S=All, it counts all pairs (a,b) in All that form a bad triple with x.
        # This is exactly the degree.
        
        # Let's compute degrees once.
        All = set(range(1, N+1))
        for x in range(1, N+1):
            degrees[x] = count_conflicts_fast(x, All)
            
        order = sorted(range(1, N+1), key=lambda x: degrees[x])
        for x in order:
            # Check if x can be added to S
            # This is equivalent to count_conflicts_fast(x, S) == 0
            if count_conflicts_fast(x, S) == 0:
                S.add(x)
        return S

    # Simulated Annealing
    def simulated_annealing(S_init, time_limit):
        S = set(S_init)
        current_conflicts = 0
        # Compute initial conflicts
        # We can sum count_conflicts_fast(x, S-{x}) for x in S?
        # Or just compute total bad triples.
        # Total bad triples = sum_{x in S} count_conflicts_fast(x, S-{x}) / 3?
        # No, each bad triple (a,b,c) is counted in count_conflicts_fast(a, {b,c}), etc.
        # So sum_{x in S} count_conflicts_fast(x, S-{x}) = 3 * (number of bad triples).
        # Let's just maintain a running count.
        
        # Recompute current_conflicts from scratch for safety
        # This is O(|S|^3) if we do it naively, but we have the fast function.
        # Sum over x in S of count_conflicts_fast(x, S-{x})
        # Note: count_conflicts_fast(x, S) includes pairs in S.
        # If we pass S-{x}, it counts pairs (a,b) in S-{x} that form bad triple with x.
        # This is exactly the contribution of x to the total bad triples.
        # So total_conflicts = sum_{x in S} count_conflicts_fast(x, S-{x})
        
        total_conflicts = 0
        for x in S:
            total_conflicts += count_conflicts_fast(x, S - {x})
            
        best_S = set(S)
        best_conflicts = total_conflicts
        best_size = len(S)
        
        T = 10.0
        cooling = 0.999
        rng = random.Random(42)
        
        start_time = time.time()
        
        while time.time() < time_limit:
            # Pick a random x to flip
            # Bias towards adding elements if we are small?
            # Or just pick random x in [1..N]
            x = rng.randint(1, N)
            
            if x in S:
                # Remove x
                # delta_conflicts = - count_conflicts_fast(x, S-{x})
                # delta_size = -1
                c = count_conflicts_fast(x, S - {x})
                delta_conflicts = -c
                delta_size = -1
            else:
                # Add x
                # delta_conflicts = + count_conflicts_fast(x, S)
                # delta_size = +1
                c = count_conflicts_fast(x, S)
                delta_conflicts = c
                delta_size = 1
                
            # Objective: Minimize conflicts, then Maximize size.
            # Let's use a lexicographic or weighted objective.
            # Score = conflicts * 1000 - size
            # Delta Score = delta_conflicts * 1000 - delta_size
            delta_score = delta_conflicts * 1000 - delta_size
            
            if delta_score <= 0:
                # Accept (improvement or tie)
                if x in S:
                    S.remove(x)
                else:
                    S.add(x)
                total_conflicts += delta_conflicts
            else:
                # Accept with probability exp(-delta_score / T)
                if rng.random() < math.exp(-delta_score / T):
                    if x in S:
                        S.remove(x)
                    else:
                        S.add(x)
                    total_conflicts += delta_conflicts
                # Else reject
                
            # Update best
            current_size = len(S)
            if total_conflicts < best_conflicts or (total_conflicts == best_conflicts and current_size > best_size):
                best_conflicts = total_conflicts
                best_size = current_size
                best_S = set(S)
                
            # Cool down
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return best_S, best_conflicts, best_size

    # 1. Greedy Init
    S_init = greedy_init()
    
    # 2. Simulated Annealing
    best_S, best_conf, best_size = simulated_annealing(S_init, deadline)
    
    # 3. Local Search Polish on the best found
    # Greedy add any element that doesn't increase conflicts (i.e., conflicts=0)
    changed = True
    while changed:
        changed = False
        for x in range(1, N+1):
            if x not in best_S:
                if count_conflicts_fast(x, best_S) == 0:
                    best_S.add(x)
                    changed = True
                    
    # 4. Try to remove elements that don't reduce conflicts?
    # If an element has 0 conflicts, removing it reduces size but keeps conflicts 0. Bad.
    # If an element has >0 conflicts, removing it might reduce total conflicts.
    # We want to minimize conflicts, then maximize size.
    # If we have 0 conflicts, we just want to maximize size.
    # If we have >0 conflicts, we want to reduce them.
    
    # If best_conf == 0, we are done (max size valid set).
    # If best_conf > 0, we might be able to trade size for validity?
    # The task requires NO bad triples. So we MUST have 0 conflicts.
    # The SA might end up with >0 conflicts if it couldn't escape.
    # We need to ensure the final set is VALID (0 conflicts).
    
    # If best_S has conflicts, we need to fix it.
    # We can do a "repair" phase: remove elements until conflicts are 0.
    # To maximize size, we should remove elements that cause the most conflicts.
    
    current_S = set(best_S)
    cur_conf = 0
    for x in current_S:
        cur_conf += count_conflicts_fast(x, current_S - {x})
        
    while cur_conf > 0 and time.time() < deadline:
        # Find x in current_S that contributes most to conflicts
        max_contrib = -1
        worst_x = None
        for x in current_S:
            c = count_conflicts_fast(x, current_S - {x})
            if c > max_contrib:
                max_contrib = c
                worst_x = x
                
        if worst_x is None:
            break
            
        # Remove worst_x
        # Update cur_conf
        # New conf = cur_conf - contrib(worst_x) + new_contribs_of_others?
        # This is tricky. Let's just recompute cur_conf after removal.
        current_S.remove(worst_x)
        cur_conf = 0
        for x in current_S:
            cur_conf += count_conflicts_fast(x, current_S - {x})
            
    # Final polish: add any valid element
    changed = True
    while changed:
        changed = False
        for x in range(1, N+1):
            if x not in current_S:
                if count_conflicts_fast(x, current_S) == 0:
                    current_S.add(x)
                    changed = True

    # Verify validity
    final_conf = 0
    for x in current_S:
        final_conf += count_conflicts_fast(x, current_S - {x})
        
    if final_conf != 0:
        # Fallback: use the greedy init which is guaranteed valid
        current_S = S_init
        
    if not current_S:
        current_S = {1}
        
    return sorted(current_S)

# EVOLVE-BLOCK-END
