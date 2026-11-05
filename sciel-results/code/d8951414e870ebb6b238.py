import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    N = 511
    
    # Precompute coprime table
    coprime = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if math.gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True
    
    def get_score(s):
        return sum(1.0 / (512 - a) for a in s)
    
    # Greedy initialization: pick largest a first
    def greedy_init():
        chosen = []
        for a in range(N, 0, -1):
            if all(coprime[a][b] for b in chosen):
                chosen.append(a)
        return chosen
    
    best_set = greedy_init()
    best_score = get_score(best_set)
    best_set_list = list(best_set)
    best_set_set = set(best_set)
    
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    def try_add_elements(chosen_list, chosen_set, current_score):
        """Try to add new elements that are coprime to all existing ones."""
        added = False
        # Try adding from largest to smallest (highest value first)
        for a in range(N, 0, -1):
            if a in chosen_set:
                continue
            if all(coprime[a][b] for b in chosen_list):
                chosen_list.append(a)
                chosen_set.add(a)
                current_score += 1.0 / (512 - a)
                added = True
        return chosen_list, chosen_set, current_score, added
    
    def local_search_swap(chosen_list, chosen_set, current_score, max_passes=5):
        """Systematic local search: try to swap each element for a better one."""
        improved = True
        passes = 0
        while improved and passes < max_passes and (time.time() - start_time) < deadline:
            improved = False
            passes += 1
            for i in range(len(chosen_list)):
                a = chosen_list[i]
                best_candidate = None
                best_gain = 0.0
                # Try candidates from largest to smallest (highest 1/(512-a) first)
                for b in range(N, 0, -1):
                    if b in chosen_set:
                        continue
                    # Check if b is coprime to all other chosen elements
                    ok = True
                    for j in range(len(chosen_list)):
                        if j == i:
                            continue
                        if not coprime[b][chosen_list[j]]:
                            ok = False
                            break
                    if not ok:
                        continue
                    gain = 1.0 / (512 - b) - 1.0 / (512 - a)
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = b
                if best_candidate is not None and best_gain > 0:
                    chosen_set.remove(a)
                    chosen_set.add(best_candidate)
                    chosen_list[i] = best_candidate
                    current_score += best_gain
                    improved = True
                    if (time.time() - start_time) > deadline:
                        break
        return chosen_list, chosen_set, current_score
    
    # Phase 1: Try to add elements
    best_set_list, best_set_set, best_score, _ = try_add_elements(
        best_set_list, best_set_set, best_score
    )
    
    # Phase 2: Local search with swaps
    best_set_list, best_set_set, best_score = local_search_swap(
        best_set_list, best_set_set, best_score
    )
    
    # Phase 3: Try adding again after swaps (swaps may have freed up coprime slots)
    best_set_list, best_set_set, best_score, _ = try_add_elements(
        best_set_list, best_set_set, best_score
    )
    
    # Phase 4: Another round of local search
    best_set_list, best_set_set, best_score = local_search_swap(
        best_set_list, best_set_set, best_score
    )
    
    # Phase 5: Random hill-climbing with restarts for further improvement
    all_candidates = list(range(1, N + 1))
    
    while time.time() < deadline:
        # Perform a series of random swaps on the current best
        current_set = list(best_set_list)
        current_set_set = set(best_set_list)
        current_score = best_score
        
        for _ in range(300):
            if time.time() > deadline:
                break
            
            # Pick a random element to remove
            idx = random.randint(0, len(current_set) - 1)
            a_old = current_set[idx]
            
            # Pick a random candidate to add
            a_new = random.choice(all_candidates)
            
            if a_new in current_set_set:
                continue
            
            # Check if replacing a_old with a_new is valid
            valid = True
            for i, a in enumerate(current_set):
                if i == idx:
                    continue
                if not coprime[a_new][a]:
                    valid = False
                    break
            
            if valid:
                new_score = current_score - (1.0 / (512 - a_old)) + (1.0 / (512 - a_new))
                
                if new_score > current_score:
                    # Accept
                    current_set[idx] = a_new
                    current_set_set.remove(a_old)
                    current_set_set.add(a_new)
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set_list = list(current_set)
                        best_set_set = set(current_set)
                else:
                    # Revert (already not changed)
                    pass
        
        # After each chain, try to add elements
        best_set_list, best_set_set, best_score, _ = try_add_elements(
            best_set_list, best_set_set, best_score
        )
        
        # Quick polish
        best_set_list, best_set_set, best_score = local_search_swap(
            best_set_list, best_set_set, best_score, max_passes=2
        )
    
    return best_set_list
# EVOLVE-BLOCK-END
