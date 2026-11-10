import time
import random

N = 256
DEADLINE = 55.0

def _precompute():
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_partners, ap4_partners

def _count_3aps(in_set, ap3_partners):
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            for j, k in ap3_partners[i]:
                if in_set[j] and in_set[k]:
                    total += 1
    return total // 3

def _is_valid_addition(y, in_set, ap4_partners):
    for j, k, l in ap4_partners[y]:
        if in_set[j] and in_set[k] and in_set[l]:
            return False
    return True

def _is_valid_swap(x, y, in_set, ap4_partners):
    # Check if y can be added to S \ {x}
    for j, k, l in ap4_partners[y]:
        if (in_set[j] and j != x) and (in_set[k] and k != x) and (in_set[l] and l != x):
            return False
    return True

def _calculate_gain(y, in_set, ap3_partners, exclude_x=None):
    """Calculate the number of 3-APs formed by adding y to in_set (with optional exclusion of x)."""
    count = 0
    for j, k in ap3_partners[y]:
        j_ok = in_set[j] and (j != exclude_x)
        k_ok = in_set[k] and (k != exclude_x)
        if j_ok and k_ok:
            count += 1
    return count

def _calculate_loss(x, in_set, ap3_partners):
    """Calculate the number of 3-APs lost by removing x from in_set."""
    count = 0
    for j, k in ap3_partners[x]:
        if in_set[j] and in_set[k]:
            count += 1
    return count

def construct():
    start_time = time.time()
    ap3_partners, ap4_partners = _precompute()
    
    best_score = -1
    best_set = [False] * (N + 1)
    
    num_iterations = 0
    while time.time() - start_time < DEADLINE - 2.0:
        num_iterations += 1
        
        # --- Phase 1: GRASP Initialization ---
        in_set = [False] * (N + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # We will iteratively pick from a restricted list
        # To make it faster, we just do a greedy pass with randomness
        # Actually, let's do a standard greedy with random tie-breaking or random selection from top K
        # Since N is small, we can compute gains for all candidates each step
        
        current_candidates = list(range(1, N + 1))
        
        while current_candidates:
            valid_candidates = []
            gains = {}
            
            for y in current_candidates:
                if _is_valid_addition(y, in_set, ap4_partners):
                    gain = _calculate_gain(y, in_set, ap3_partners)
                    valid_candidates.append(y)
                    gains[y] = gain
            
            if not valid_candidates:
                break
            
            # Select from top K
            K = min(5, len(valid_candidates))
            # Sort by gain descending
            valid_candidates.sort(key=lambda x: gains[x], reverse=True)
            top_k = valid_candidates[:K]
            
            # Random selection from top K, weighted by gain or just random
            # Let's use a softmax-like selection or just random
            y = random.choice(top_k)
            
            in_set[y] = True
            current_candidates.remove(y)
            
        # Calculate initial score
        score = _count_3aps(in_set, ap3_partners)
        
        if score > best_score:
            best_score = score
            best_set = in_set[:]
            
        # --- Phase 2: Tabu Search Local Search ---
        # Tabu list: list of (element, type) or just element?
        # Let's use a simple list of elements that are tabu to be added/removed/swapped in a specific way.
        # Simpler: Tabu list of (x, y) for swaps, and x for adds/removes?
        # Let's just use a list of recent moves.
        # Move: (type, args)
        # Type 0: Add y
        # Type 1: Remove x
        # Type 2: Swap x, y
        
        tabu_list = []
        tabu_tenure = 10
        no_improve_steps = 0
        max_no_improve = 50
        steps = 0
        max_steps = 2000
        
        while steps < max_steps and no_improve_steps < max_no_improve:
            if time.time() - start_time > DEADLINE - 1.0:
                break
                
            best_delta = -10**9
            best_move = None
            
            # Generate all possible moves
            # 1. Add y
            for y in range(1, N + 1):
                if not in_set[y]:
                    if _is_valid_addition(y, in_set, ap4_partners):
                        # Check tabu: is (0, y) in tabu_list?
                        is_tabu = any(move[0] == 0 and move[1] == y for move in tabu_list)
                        if not is_tabu:
                            delta = _calculate_gain(y, in_set, ap3_partners)
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (0, y)
            
            # 2. Remove x
            for x in range(1, N + 1):
                if in_set[x]:
                    is_tabu = any(move[0] == 1 and move[1] == x for move in tabu_list)
                    if not is_tabu:
                        delta = -_calculate_loss(x, in_set, ap3_partners)
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (1, x)
            
            # 3. Swap x, y
            # This is O(N^2) which might be slow.
            # Optimization: Only consider swaps where x is in S and y is not in S.
            # We can limit this to a random subset or just do it.
            # N=256, N^2 = 65k. 2000 steps * 65k = 130M operations. Might be too slow in Python.
            # Let's limit swap candidates.
            # Randomly select 10 x in S and 10 y not in S.
            present = [i for i in range(1, N + 1) if in_set[i]]
            absent = [i for i in range(1, N + 1) if not in_set[i]]
            
            if present and absent:
                sample_present = random.sample(present, min(10, len(present)))
                sample_absent = random.sample(absent, min(10, len(absent)))
                
                for x in sample_present:
                    for y in sample_absent:
                        if _is_valid_swap(x, y, in_set, ap4_partners):
                            is_tabu = any((move[0] == 2 and move[1] == x and move[2] == y) or 
                                          (move[0] == 2 and move[1] == y and move[2] == x) 
                                          for move in tabu_list)
                            if not is_tabu:
                                # Delta = Gain(y, S\{x}) - Loss(x, S)
                                gain_y = _calculate_gain(y, in_set, ap3_partners, exclude_x=x)
                                loss_x = _calculate_loss(x, in_set, ap3_partners)
                                delta = gain_y - loss_x
                                if delta > best_delta:
                                    best_delta = delta
                                    best_move = (2, x, y)
            
            if best_move is None:
                break
            
            # Apply best move
            if best_move[0] == 0: # Add
                y = best_move[1]
                in_set[y] = True
                score += best_delta
                tabu_list.append((0, y))
            elif best_move[0] == 1: # Remove
                x = best_move[1]
                in_set[x] = False
                score += best_delta
                tabu_list.append((1, x))
            else: # Swap
                x, y = best_move[1], best_move[2]
                in_set[x] = False
                in_set[y] = True
                score += best_delta
                tabu_list.append((2, x, y))
            
            # Update best
            if score > best_score:
                best_score = score
                best_set = in_set[:]
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            # Update tabu list
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
                
            steps += 1
            
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
