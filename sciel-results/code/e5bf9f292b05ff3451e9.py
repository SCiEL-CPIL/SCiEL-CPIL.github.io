import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 2s safety margin
    
    # We want the smallest inclusion-maximal Sidon set.
    # Lower bound on size: A Sidon set in [1, N] has size at most ~sqrt(N).
    # But we want SMALL maximal sets.
    # A set S is maximal if every x not in S conflicts with S.
    # Conflict for x means: exists a,b in S such that x+a = b+c (with a<=b, c<=d? No, just sums).
    # Actually, x conflicts if {x+s for s in S} U {2x} intersects Sums(S).
    
    # Strategy:
    # 1. Try to find a valid maximal Sidon set of size k, starting from small k.
    # 2. Use randomized backtracking / local search to find one.
    
    # Heuristic: Small maximal sets often consist of large numbers?
    # If we pick large numbers, their sums are large and distinct more easily?
    # But to be maximal, we need to block all small numbers too.
    # If S contains large numbers, x (small) + s (large) might be small+large.
    # Does it hit a sum in S?
    # If S = {N, N-1, ...}, sums are ~2N.
    # x + N ~ N+x. Is N+x a sum in S?
    # Sums of {N, N-1} are 2N, 2N-1, 2N-2.
    # N+x is not in {2N, ...} for small x.
    # So large numbers might not block small numbers.
    
    # Conversely, if S contains small numbers, x (small) + s (small) is small.
    # Sums are small.
    # If S = {1, 3}, sums: 2, 4, 6.
    # x=2: 2+1=3 (in S? No, sum must be in Sums(S)). Wait.
    # Conflict definition: x can be added if (x+a) not in Sums(S) for all a in S, and 2x not in Sums(S).
    # If S={1,3}, Sums={2,4,6}.
    # x=2: 2+1=3 (not in Sums), 2+3=5 (not in Sums), 2*2=4 (IN Sums). Conflict!
    # So x=2 is blocked.
    # x=4: 4+1=5 (no), 4+3=7 (no), 8 (no). Can be added?
    # If we add 4, S'={1,3,4}. Sums: 2,4,6, 5,7, 8. Distinct? Yes.
    # So {1,3} is NOT maximal.
    
    # To be maximal, S must "cover" all x in [1,N]\S.
    # This implies S must be fairly dense or strategically placed.
    
    # Let's try a different angle:
    # The problem is equivalent to finding a minimal dominating set in a graph?
    # Graph G where vertices are [1,N]. Edge (x,y) if x and y can coexist in a Sidon set?
    # No, Sidon is a global property (all pair sums distinct).
    
    # Let's use a randomized local search that starts from a random set of size k
    # and tries to satisfy:
    # 1. Sidon property.
    # 2. Maximality.
    #
    # Objective: Minimize k.
    
    best_set = None
    best_size = N + 1
    
    rng = random.Random(123)
    
    # We know from literature/experiments that minimal maximal Sidon sets in [1,256]
    # are likely around size 10-20?
    # Let's try sizes from 8 upwards.
    
    # Helper: Check if a set is valid (Sidon + Maximal)
    def is_valid(arr):
        if not arr:
            return False
        arr_set = set(arr)
        if len(arr_set) != len(arr):
            return False
        for x in arr:
            if x < 1 or x > N:
                return False
        
        sums = set()
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i, len(arr)):
                b = arr[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        
        # Maximality
        for x in range(1, N + 1):
            if x in arr_set:
                continue
            conflict = False
            for a in arr:
                if (x + a) in sums:
                    conflict = True
                    break
            if not conflict:
                if (2 * x) in sums:
                    conflict = True
            if not conflict:
                return False
        return True

    # Helper: Count violations for a set (for local search)
    # Violation 1: Sidon (duplicate sums)
    # Violation 2: Maximality (x that can be added)
    
    def get_violations(arr):
        arr_set = set(arr)
        if len(arr_set) != len(arr):
            return float('inf'), 0
        for x in arr:
            if x < 1 or x > N:
                return float('inf'), 0
        
        sums = set()
        sidon_viol = 0
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i, len(arr)):
                b = arr[j]
                s = a + b
                if s in sums:
                    sidon_viol += 1
                else:
                    sums.add(s)
        
        max_viol = 0
        for x in range(1, N + 1):
            if x in arr_set:
                continue
            conflict = False
            for a in arr:
                if (x + a) in sums:
                    conflict = True
                    break
            if not conflict:
                if (2 * x) in sums:
                    conflict = True
            if not conflict:
                max_viol += 1
                
        return sidon_viol, max_viol

    # Local Search:
    # Start with a random set of size k.
    # Move: Add, Remove, or Swap an element.
    # Accept move if it reduces total violations (Sidon + Maximal).
    
    def local_search(k, seed):
        rng_local = random.Random(seed)
        # Initialize: Random k elements
        arr = rng_local.sample(range(1, N + 1), k)
        
        cur_sidon, cur_max = get_violations(arr)
        cur_cost = cur_sidon * 10 + cur_max  # Prioritize fixing Sidon? Or Max?
        # Actually, if Sidon is broken, it's invalid. If Max is broken, it's not maximal.
        # We want both 0.
        
        best_arr = arr[:]
        best_cost = cur_cost
        
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random move
            move_type = rng_local.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add' and len(arr) < N:
                candidates = [x for x in range(1, N + 1) if x not in arr]
                if not candidates: continue
                x = rng_local.choice(candidates)
                new_arr = arr + [x]
            elif move_type == 'remove' and len(arr) > 1:
                x = rng_local.choice(arr)
                new_arr = [a for a in arr if a != x]
            elif move_type == 'swap' and len(arr) > 0:
                x_in = rng_local.choice(arr)
                candidates = [x for x in range(1, N + 1) if x not in arr]
                if not candidates: continue
                x_out = rng_local.choice(candidates)
                new_arr = [x_out if a == x_in else a for a in arr]
            else:
                continue
            
            new_sidon, new_max = get_violations(new_arr)
            new_cost = new_sidon * 10 + new_max
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = new_arr[:]
                if new_cost == 0:
                    return new_arr
            if new_cost <= cur_cost: # Accept if not worse (or equal) to allow exploration
                cur_cost = new_cost
                arr = new_arr
            else:
                # Simulated annealing style acceptance?
                # prob = exp(-(new_cost - cur_cost) / T)
                # For now, simple hill climbing with restarts is safer.
                pass
                
        return best_arr if best_cost == 0 else None

    # Try increasing sizes
    # Based on similar problems, minimal maximal Sidon sets for N=256 might be small.
    # Let's try k from 6 to 20.
    
    for k in range(6, 25):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            seed = attempts * 1000 + k
            res = local_search(k, seed)
            if res:
                # Verify
                if is_valid(res):
                    best_set = res
                    best_size = k
                    found = True
                    break
        
        if found:
            # Try to find even smaller? No, we went up from 6.
            # So k is the minimum found.
            break
            
    if best_set is None:
        # Fallback: Greedy construction (which is guaranteed to be maximal if we keep adding? No, greedy adds if possible. It stops when no more can be added. So it is maximal.)
        # Wait, standard greedy: "Add x if it doesn't conflict".
        # If we iterate through all x in [1,N], and add if possible, the resulting set is maximal?
        # Yes, because we considered every x. If x wasn't added, it conflicted with the set at that time.
        # But adding later elements might change the conflict status?
        # No, Sidon property is monotonic? No.
        # If x conflicted with S, and we add y to S to get S', does x still conflict with S'?
        # Conflict for x: exists a in S' such that x+a in Sums(S').
        # If x conflicted with S, there was a in S such that x+a in Sums(S).
        # Sums(S) is a subset of Sums(S')? Yes.
        # a is in S, so a is in S'.
        # So x+a is in Sums(S') and a is in S'.
        # So x still conflicts.
        # Thus, greedy saturation produces a maximal Sidon set.
        
        # Let's just run a few greedy restarts to get a valid set if search fails.
        for _ in range(50):
            if time.time() > deadline:
                break
            order = list(range(1, N + 1))
            rng.shuffle(order)
            selected = []
            sums = set()
            for x in order:
                conflict = False
                for a in selected:
                    if (x + a) in sums:
                        conflict = True
                        break
                if conflict:
                    continue
                if (2 * x) in sums:
                    continue
                selected.append(x)
                for a in selected:
                    sums.add(x + a)
                sums.add(2 * x)
            
            if is_valid(selected):
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected
                break # Found one, good enough for fallback?
                # Actually, let's keep looking for smaller if we have time.
                # But if we found one via greedy, it's likely large.
                # The local search above should have found smaller if they existed.
                # So we can break.
    
    if best_set is None:
        # Absolute fallback: return a hardcoded valid set or construct one deterministically
        # Greedy descending is usually valid and fast.
        order = list(range(N, 0, -1))
        selected = []
        sums = set()
        for x in order:
            conflict = False
            for a in selected:
                if (x + a) in sums:
                    conflict = True
                    break
            if conflict:
                continue
            if (2 * x) in sums:
                continue
            selected.append(x)
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
        best_set = selected
        
    return best_set

# EVOLVE-BLOCK-END
