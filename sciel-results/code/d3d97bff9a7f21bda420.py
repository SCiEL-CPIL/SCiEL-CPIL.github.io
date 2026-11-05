import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0

    def is_sidon_compatible(x, A, used_sums):
        if 2 * x in used_sums:
            return False
        for y in A:
            if x + y in used_sums:
                return False
        return True

    def get_newly_blocked(x, A, used_sums, unblocked):
        new_used_sums = set(used_sums)
        new_used_sums.add(2 * x)
        for y in A:
            new_used_sums.add(x + y)

        newly_blocked = []
        for z in unblocked:
            if z == x:
                continue
            if 2 * z in new_used_sums:
                newly_blocked.append(z)
                continue
            blocked = False
            for y in A:
                if z + y in new_used_sums:
                    blocked = True
                    break
            if not blocked and z + x in new_used_sums:
                blocked = True
            if blocked:
                newly_blocked.append(z)
        return newly_blocked

    def greedy_max_coverage(init_A=None, randomize_ties=False, bias=0.0):
        if init_A is None:
            A = []
        else:
            A = list(init_A)
            
        # Validate init_A is Sidon
        # (Assuming caller provides valid Sidon set)
        
        used_sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                used_sums.add(A[i] + A[j])

        blocked = set(A)
        unblocked = set(range(1, N + 1)) - set(A)

        while unblocked:
            best_x = None
            best_score = -1
            best_newly_blocked = []
            
            # Pre-compute candidates?
            # Just iterate
            for x in range(1, N + 1):
                if x in A:
                    continue
                if not is_sidon_compatible(x, A, used_sums):
                    continue
                
                newly_blocked = get_newly_blocked(x, A, used_sums, unblocked)
                score = len(newly_blocked)
                
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
                    best_newly_blocked = newly_blocked
            
            if best_x is None:
                break

            A.append(best_x)
            used_sums.add(2 * best_x)
            for y in A:
                if y != best_x: # Avoid double adding 2*best_x? No, y is from A before append?
                    # Wait, A was appended.
                    # used_sums.add(best_x + y)
                    pass
            # Correct update of used_sums
            # We added best_x to A.
            # We need to add sums best_x + y for all y in A (including best_x)
            # But we already added 2*best_x.
            # So add best_x + y for y in A where y != best_x?
            # Actually, simpler:
            # for y in A:
            #     used_sums.add(best_x + y)
            # This adds 2*best_x again (harmless) and best_x+y for others.
            for y in A:
                used_sums.add(best_x + y)
            
            for z in best_newly_blocked:
                if z in unblocked:
                    unblocked.remove(z)
                    blocked.add(z)
            
            if best_x in unblocked:
                unblocked.remove(best_x)
                blocked.add(best_x)

        return A

    # Initial deterministic run
    best_set = greedy_max_coverage()
    best_size = len(best_set)
    
    # Iterated Local Search
    while time.time() < deadline:
        # Perturbation strategy: Remove k elements from best_set and re-saturate
        # Try different k
        k = random.randint(1, 3)
        if len(best_set) <= k:
            # If set is too small, just do a fresh restart or random init
            init_A = [random.randint(1, N)]
        else:
            # Remove k random elements
            A_perturbed = list(best_set)
            indices_to_remove = random.sample(range(len(A_perturbed)), k)
            # Remove in reverse order to preserve indices
            for idx in sorted(indices_to_remove, reverse=True):
                A_perturbed.pop(idx)
            init_A = A_perturbed

        # Re-saturate with randomization
        current_set = greedy_max_coverage(init_A=init_A, randomize_ties=True)
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set

    return best_set
