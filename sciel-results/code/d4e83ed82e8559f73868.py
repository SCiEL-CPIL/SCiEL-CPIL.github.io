# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    n = 16
    k = 4
    max_target = 250

    # Generate all 4-subsets as bitmasks (0-indexed internally, 1-indexed in output)
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)
    best_family = []
    best_size = 0
    rng = random.Random(12345)

    def greedy_build(order, current_best_size):
        """
        Greedily build a family from the given order of set indices.
        Returns the family as a list of masks.
        Uses an efficient data structure to detect sunflowers:
        For each set i in F, inter_maps[i] maps intersection_mask -> list of indices j
        such that F[i] & F[j] == intersection_mask.
        
        A new set m conflicts with existing F[i], F[j] if there exists i, j in F
        such that m & F[i] == m & F[j] == F[i] & F[j].
        This is equivalent to: for some i, let s = m & F[i]. If there is a j in F
        such that m & F[j] == s and F[i] & F[j] == s, then {m, F[i], F[j]} is a sunflower.
        Actually, the condition is that the three pairwise intersections are equal.
        Let A=m, B=F[i], C=F[j].
        A&B = s, A&C = s. We need B&C = s.
        So for a fixed i, s = A&B. We check if there exists j such that A&C = s AND B&C = s.
        The data structure inter_maps[i][s] gives all j such that B&C = s (i.e., F[i]&F[j]==s).
        For each such j, we check if A&C == s (i.e., m & F[j] == s).
        If yes, conflict.
        """
        F = []
        inter_maps = []
        
        for idx in order:
            m = all_sets[idx]
            conflict = False
            
            for i in range(len(F)):
                s = m & F[i]
                lst = inter_maps[i].get(s)
                if lst:
                    for j in lst:
                        if (m & F[j]) == s:
                            conflict = True
                            break
                if conflict:
                    break
            
            if not conflict:
                idx_new = len(F)
                F.append(m)
                inter_maps.append({})
                
                for i in range(idx_new):
                    s = m & F[i]
                    if s not in inter_maps[i]:
                        inter_maps[i][s] = []
                    inter_maps[i][s].append(idx_new)
                    if s not in inter_maps[idx_new]:
                        inter_maps[idx_new][s] = []
                    inter_maps[idx_new][s].append(i)
                
                if len(F) >= max_target:
                    break
                # Early termination if we can't beat current best
                if len(F) + (total - order.index(idx) if idx in order else 0) <= current_best_size:
                    # This check is expensive due to order.index, so we skip it or do it differently
                    # Actually, just checking len(F) + remaining < current_best_size is better
                    # But we don't track remaining easily in the loop without extra state.
                    # Let's just rely on the time check and the final comparison.
                    pass

        return F

    # 1. Deterministic pass (natural order)
    if time.time() < deadline:
        order = list(range(total))
        F = greedy_build(order, best_size)
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
    
    # 2. Random restarts with different seeds/orders
    # We will run multiple random restarts.
    # To make it more effective, we can try different random seeds.
    # Also, we can try a "reverse" deterministic pass or other simple heuristics if time permits.
    
    # Let's run random restarts until time runs out.
    # Each restart is O(N^3) roughly in the worst case, but usually much faster.
    # N=1820.
    # The greedy build is the bottleneck.
    
    # Optimization: Pre-compute intersections? No, memory is too high (1820^2 * 4 bytes ~ 13MB, which is fine, but logic is complex).
    # The current inter_maps approach is O(N^2) per build, which is ~3.3M ops. In Python, this might take ~0.1-0.5s per build.
    # So we can do roughly 100-500 restarts in 55 seconds.
    
    # Let's increase the number of restarts significantly if the first few are fast.
    # We will loop until deadline.
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Use a different seed for each restart to diversify
        seed = 12345 + restart_count
        rng = random.Random(seed)
        
        order = list(range(total))
        rng.shuffle(order)
        
        F = greedy_build(order, best_size)
        
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
            if best_size >= max_target:
                break
        
        # If we've done a lot of restarts without improvement, maybe try a different strategy?
        # For now, just keep going.
        
        # Safety check: if we are getting very few restarts, we might be too slow.
        # If restart_count is 1 and time used is > 10s, we are in trouble.
        # But we can't change the algorithm mid-loop easily without restructuring.
        # Let's just monitor.
        
    # Convert best_family masks to sorted 1-indexed lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
