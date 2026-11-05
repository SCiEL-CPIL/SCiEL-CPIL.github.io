import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 27.0  # safety margin before 30s limit

    def is_sidon(A):
        """Check if A is a Sidon set (all unordered sums a+b with a<=b are distinct)."""
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(A):
        """Check if A is inclusion-maximal: adding any element from [1,N]\A destroys Sidon property."""
        A_set = set(A)
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        for x in range(1, N + 1):
            if x in A_set:
                continue
            if 2 * x in sums:
                continue
            for y in A:
                if x + y in sums:
                    break
            else:
                return False  # Can add x, so not maximal
        return True

    def greedy_max_coverage(init_A=None, randomize_ties=False, bias=0.0, seed=None):
        """Greedy: at each step pick the element that blocks the most remaining unblocked elements."""
        if seed is not None:
            random.seed(seed)
        
        A = list(init_A) if init_A else []
        A_set = set(A)
        
        used_sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                used_sums.add(A[i] + A[j])
        
        unblocked = set(range(1, N + 1)) - A_set
        
        # Precompute compatibility for each element
        def get_compatible(A_list, sums):
            comp = [False] * (N + 1)
            for z in range(1, N + 1):
                if z in A_set:
                    continue
                if 2 * z in sums:
                    continue
                ok = True
                for y in A_list:
                    if z + y in sums:
                        ok = False
                        break
                if ok:
                    comp[z] = True
            return comp
        
        comp = get_compatible(A, used_sums)
        
        while unblocked:
            best_x = None
            best_score = -1
            
            for x in range(1, N + 1):
                if x in A_set or not comp[x]:
                    continue
                
                # Algebraic blocking: z is newly blocked by adding x iff
                # 2z == 2x OR 2z == x+y (y in A) OR z+y == 2x (y in A) OR z+y == x+w (y,w in A)
                # Since A is Sidon, all these sums are distinct from used_sums.
                newly_blocked = 0
                for z in unblocked:
                    if z == x or not comp[z]:
                        continue
                    blocked = False
                    # 2z == x + y => y = 2z - x, check if y in A
                    y_cand = 2 * z - x
                    if y_cand in A_set:
                        blocked = True
                    if not blocked:
                        # z + y == 2x => y = 2x - z, check if y in A
                        y_cand = 2 * x - z
                        if y_cand in A_set:
                            blocked = True
                    if not blocked:
                        # z + y == x + w => w = z + y - x, check if w in A for some y in A
                        for y in A:
                            if z + y - x in A_set:
                                blocked = True
                                break
                    if blocked:
                        newly_blocked += 1
                
                score = newly_blocked
                if bias != 0.0:
                    if bias > 0:
                        score -= x * bias
                    else:
                        score += x * abs(bias)
                if randomize_ties:
                    score += random.uniform(0, 0.5)
                
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            A.append(best_x)
            A_set.add(best_x)
            used_sums.add(2 * best_x)
            for y in A[:-1]:
                used_sums.add(best_x + y)
            
            unblocked.discard(best_x)
            comp = get_compatible(A, used_sums)
        
        return A

    def local_search_improve(A):
        """Try to reduce size via swap/remove operations while maintaining maximality."""
        A = list(A)
        improved = True
        while improved:
            improved = False
            for i in range(len(A)):
                candidate = A[:i] + A[i+1:]
                if len(candidate) == 0:
                    continue
                if not is_sidon(candidate):
                    continue
                saturated = greedy_max_coverage(init_A=candidate)
                if len(saturated) < len(A) and is_maximal(saturated):
                    A = saturated
                    improved = True
                    break
        return A

    # Initial deterministic run
    best_set = greedy_max_coverage()
    best_size = len(best_set)
    
    improved = local_search_improve(best_set)
    if len(improved) < best_size:
        best_size = len(improved)
        best_set = improved

    # Randomized restarts with various strategies
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        r = random.random()
        
        if r < 0.25:
            current_set = greedy_max_coverage(randomize_ties=True, seed=attempt)
        elif r < 0.45:
            current_set = greedy_max_coverage(randomize_ties=True, bias=0.005, seed=attempt)
        elif r < 0.65:
            current_set = greedy_max_coverage(randomize_ties=True, bias=-0.005, seed=attempt)
        elif r < 0.85:
            init_x = random.randint(1, N)
            current_set = greedy_max_coverage(init_A=[init_x], randomize_ties=True, seed=attempt)
        else:
            x1 = random.randint(1, N)
            x2 = random.randint(1, N)
            while x2 == x1 or x1 + x2 == 2*x1 or x1 + x2 == 2*x2:
                x2 = random.randint(1, N)
            init = [x1, x2]
            if is_sidon(init):
                current_set = greedy_max_coverage(init_A=init, randomize_ties=True, seed=attempt)
            else:
                current_set = greedy_max_coverage(randomize_ties=True, seed=attempt)
        
        improved = local_search_improve(current_set)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved
    
    return best_set
