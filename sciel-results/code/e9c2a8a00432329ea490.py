"""
Optimization program for maximizing 3-term APs in a subset of [1,256] with no 4-term AP.
Combines fast precomputed AP checking with multi-phase search:
1. Greedy construction using incremental delta gains
2. Local search with 1-move (add/remove) and 2-swap improvements
3. Simulated annealing with restarts to escape local optima
"""
import time
import random
import math

N = 256
MAX_TIME = 82  # seconds, safety margin before 90s kill

# Precompute 3-APs and 4-APs
_three_aps = []
for d in range(1, N // 2 + 1):
    for a in range(1, N - 2 * d + 1):
        _three_aps.append((a, a + d, a + 2 * d))

_four_aps = []
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        _four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

# Map elements to APs containing them
_elem_to_3aps = [[] for _ in range(N + 1)]
for i, ap in enumerate(_three_aps):
    for x in ap:
        _elem_to_3aps[x].append(i)

_elem_to_4aps = [[] for _ in range(N + 1)]
for i, ap in enumerate(_four_aps):
    for x in ap:
        _elem_to_4aps[x].append(i)


def count_3aps(S):
    """Count distinct 3-term arithmetic progressions in set S."""
    S = set(S)
    count = 0
    for ap in _three_aps:
        if ap[0] in S and ap[1] in S and ap[2] in S:
            count += 1
    return count


def has_4ap(S):
    """Check if set S contains any nonconstant 4-term arithmetic progression."""
    S = set(S)
    for ap in _four_aps:
        if ap[0] in S and ap[1] in S and ap[2] in S and ap[3] in S:
            return True
    return False


def is_valid_add(S, x):
    """Check if adding x to S creates a 4-AP."""
    for ap_idx in _elem_to_4aps[x]:
        a, b, c, d = _four_aps[ap_idx]
        if x == a:
            if b in S and c in S and d in S:
                return False
        elif x == b:
            if a in S and c in S and d in S:
                return False
        elif x == c:
            if a in S and b in S and d in S:
                return False
        else:
            if a in S and b in S and c in S:
                return False
    return True


def delta_add(S, x):
    """Change in 3-AP count if we add x to S."""
    delta = 0
    for ap_idx in _elem_to_3aps[x]:
        a, b, c = _three_aps[ap_idx]
        if x == a:
            if b in S and c in S:
                delta += 1
        elif x == b:
            if a in S and c in S:
                delta += 1
        else:
            if a in S and b in S:
                delta += 1
    return delta


def delta_remove(S, x):
    """Change in 3-AP count if we remove x from S (negative = loss)."""
    delta = 0
    for ap_idx in _elem_to_3aps[x]:
        a, b, c = _three_aps[ap_idx]
        if x == a:
            if b in S and c in S:
                delta += 1
        elif x == b:
            if a in S and c in S:
                delta += 1
        else:
            if a in S and b in S:
                delta += 1
    return -delta


def is_valid_swap(S, x_out, x_in):
    """Check if swapping x_out for x_in is valid (no 4-APs in new set)."""
    # New set is S' = S \ {x_out} U {x_in}
    # A 4-AP is formed if it contains x_in and 3 other elements from S'
    for ap_idx in _elem_to_4aps[x_in]:
        a, b, c, d = _four_aps[ap_idx]
        if x_in == a:
            others = (b, c, d)
        elif x_in == b:
            others = (a, c, d)
        elif x_in == c:
            others = (a, b, d)
        else:
            others = (a, b, c)
        
        # All others must be in S' = S \ {x_out}
        # i.e., all in S and none equal to x_out
        all_in = True
        for y in others:
            if y not in S or y == x_out:
                all_in = False
                break
        if all_in:
            return False
    return True


def swap_gain(S, x_out, x_in):
    """Net change in 3-AP count for swapping x_out out and x_in in."""
    # Loss from removing x_out
    loss = 0
    for ap_idx in _elem_to_3aps[x_out]:
        a, b, c = _three_aps[ap_idx]
        if x_out == a:
            if b in S and c in S:
                loss += 1
        elif x_out == b:
            if a in S and c in S:
                loss += 1
        else:
            if a in S and b in S:
                loss += 1
    
    # Gain from adding x_in to S \ {x_out}
    gain = 0
    for ap_idx in _elem_to_3aps[x_in]:
        a, b, c = _three_aps[ap_idx]
        if x_in == a:
            if b in S and c in S and b != x_out and c != x_out:
                gain += 1
        elif x_in == b:
            if a in S and c in S and a != x_out and c != x_out:
                gain += 1
        else:
            if a in S and b in S and a != x_out and b != x_out:
                gain += 1
    
    return gain - loss


def greedy_construct(seed=0):
    """
    Greedy construction: add elements in order of highest marginal 3-AP gain
    while maintaining no-4-AP constraint.
    """
    random.seed(seed)
    S = set()
    available = list(range(1, N + 1))
    
    while available:
        best_gain = -1
        best_elem = None
        candidates = []
        for x in available:
            if not is_valid_add(S, x):
                continue
            gain = delta_add(S, x)
            candidates.append((gain, x))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        # Pick best or from top-K for diversity
        top_k = min(3, len(candidates))
        if random.random() < 0.7:
            best_elem = candidates[0][1]
        else:
            best_elem = random.choice([c[1] for c in candidates[:top_k]])
        
        S.add(best_elem)
        available.remove(best_elem)
    
    return S


def local_search_1move(S, max_iters=5000):
    """
    Local search with single-add and single-remove moves.
    Returns improved set.
    """
    S = set(S)
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding elements
        for x in range(1, N + 1):
            if x not in S:
                if not is_valid_add(S, x):
                    continue
                gain = delta_add(S, x)
                if gain > 0:
                    S.add(x)
                    current_score += gain
                    improved = True
                    break
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
            continue
        
        # Try removing elements
        for x in list(S):
            loss = -delta_remove(S, x)  # loss is positive
            if loss < 0:  # removing increases score (unlikely but possible)
                S.discard(x)
                current_score += loss
                improved = True
                break
        
        if not improved:
            break
    
    return best_S


def local_search_swap(S, max_iters=3000):
    """
    Local search with swap moves (remove one, add another).
    Returns improved set.
    """
    S = set(S)
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    not_in_S = set(range(1, N + 1)) - S
    
    for _ in range(max_iters):
        improved = False
        best_swap = None
        best_gain = 0
        
        for x_out in list(S):
            for x_in in not_in_S:
                if not is_valid_swap(S, x_out, x_in):
                    continue
                g = swap_gain(S, x_out, x_in)
                if g > best_gain:
                    best_gain = g
                    best_swap = (x_out, x_in)
        
        if best_swap is not None and best_gain > 0:
            x_out, x_in = best_swap
            S.discard(x_out)
            S.add(x_in)
            not_in_S.discard(x_in)
            not_in_S.add(x_out)
            current_score += best_gain
            improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
        
        if not improved:
            break
    
    return best_S


def simulated_annealing(S, initial_temp=10.0, cooling=0.9975, max_iters=15000):
    """
    Simulated annealing with add, remove, and swap moves.
    Uses weighted move selection and better acceptance criteria.
    """
    S = set(S)
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    T = initial_temp
    not_in_S = set(range(1, N + 1)) - S
    no_improve_count = 0
    
    for it in range(max_iters):
        # Adaptive move probabilities: more swaps as we cool
        swap_prob = 0.3 + 0.3 * (1.0 - T / initial_temp)
        add_prob = 0.35
        move_type = random.random()
        moved = False
        
        if move_type < add_prob:
            # Add move
            candidates = [x for x in not_in_S if is_valid_add(S, x)]
            if candidates:
                # Weight by gain for better exploration
                gains = [delta_add(S, x) for x in candidates]
                total_gain = sum(max(g, 0.1) for g in gains)
                if total_gain > 0:
                    weights = [max(g, 0.1) / total_gain for g in gains]
                    x = random.choices(candidates, weights=weights, k=1)[0]
                else:
                    x = random.choice(candidates)
                gain = delta_add(S, x)
                if gain > 0 or random.random() < _exp(gain / max(T, 0.1)):
                    S.add(x)
                    not_in_S.discard(x)
                    current_score += gain
                    moved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
        elif move_type < add_prob + (1 - add_prob - swap_prob):
            # Remove move
            if S:
                x = random.choice(list(S))
                loss = -delta_remove(S, x)  # loss is negative when removing helps
                delta = -loss  # change in score = -loss
                if delta > 0 or random.random() < _exp(delta / max(T, 0.1)):
                    S.discard(x)
                    not_in_S.add(x)
                    current_score += delta
                    moved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
        else:
            # Swap move
            in_S = list(S)
            if in_S and not_in_S:
                x_out = random.choice(in_S)
                x_in = random.choice(list(not_in_S))
                if is_valid_swap(S, x_out, x_in):
                    g = swap_gain(S, x_out, x_in)
                    if g > 0 or random.random() < _exp(g / max(T, 0.1)):
                        S.discard(x_out)
                        S.add(x_in)
                        not_in_S.discard(x_in)
                        not_in_S.add(x_out)
                        current_score += g
                        moved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
        
        if not moved:
            no_improve_count += 1
        
        # Adaptive cooling: cool faster if stuck
        if no_improve_count > 200:
            T *= 0.98
            no_improve_count = 0
        else:
            T *= cooling
        
        if T < 0.05:
            break
    
    return best_S


def _exp(x):
    """Safe exponent for acceptance probability."""
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    return math.exp(x)


def construct():
    start_time = time.time()
    
    # Phase 1: Multiple greedy constructions with different seeds
    best_S = set()
    best_score = -1
    
    for seed in range(5):
        if time.time() - start_time > MAX_TIME * 0.15:
            break
        S = greedy_construct(seed)
        score = count_3aps(S)
        if score > best_score:
            best_score = score
            best_S = set(S)
    
    # Phase 2: Local search refinement (1-move then swap)
    S = local_search_1move(best_S)
    score = count_3aps(S)
    if score > best_score:
        best_score = score
        best_S = set(S)
    
    S = local_search_swap(best_S)
    score = count_3aps(S)
    if score > best_score:
        best_score = score
        best_S = set(S)
    
    # Another 1-move pass after swaps
    S = local_search_1move(best_S)
    score = count_3aps(S)
    if score > best_score:
        best_score = score
        best_S = set(S)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME * 0.4:
        if has_4ap(best_S):
            best_S = greedy_construct(0)
        return sorted(best_S)
    
    # Phase 3: Simulated annealing with restarts
    remaining_time = MAX_TIME - elapsed
    n_restarts = max(1, int(remaining_time / 6))
    
    for r in range(n_restarts):
        if time.time() - start_time > MAX_TIME - 5:
            break
        
        # Structured perturbation: remove k elements and let SA refill
        perturbed = set(best_S)
        k_remove = random.randint(4, 10)
        elems = list(perturbed)
        random.shuffle(elems)
        for x in elems[:k_remove]:
            perturbed.discard(x)
        
        # Check validity and fix if needed
        if has_4ap(perturbed):
            for x in list(perturbed):
                if not has_4ap(perturbed - {x}):
                    perturbed.discard(x)
                    break
        
        # Run SA from perturbed state with higher initial temp for exploration
        sa_temp = 6.0 + random.random() * 4.0
        sa_S = simulated_annealing(perturbed, initial_temp=sa_temp, cooling=0.9975, max_iters=10000)
        sa_score = count_3aps(sa_S)
        
        if sa_score > best_score:
            best_score = sa_score
            best_S = set(sa_S)
        
        # Local search on SA result
        ls_S = local_search_1move(sa_S, max_iters=1500)
        ls_score = count_3aps(ls_S)
        if ls_score > best_score:
            best_score = ls_score
            best_S = set(ls_S)
        
        ls_S = local_search_swap(best_S, max_iters=800)
        ls_score = count_3aps(ls_S)
        if ls_score > best_score:
            best_score = ls_score
            best_S = set(ls_S)
    
    # Final polish: alternating local search passes
    for _ in range(2):
        if time.time() - start_time > MAX_TIME - 3:
            break
        final_S = local_search_1move(best_S, max_iters=1500)
        final_score = count_3aps(final_S)
        if final_score > best_score:
            best_score = final_score
            best_S = set(final_S)
        
        final_S = local_search_swap(best_S, max_iters=800)
        final_score = count_3aps(final_S)
        if final_score > best_score:
            best_score = final_score
            best_S = set(final_S)
    
    # Final verification
    if has_4ap(best_S):
        # Fallback to greedy
        best_S = greedy_construct(0)
        best_S = local_search_1move(best_S)
    
    return sorted(best_S)


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}, 3-APs: {count_3aps(result)}")
    print(f"Has 4-AP: {has_4ap(result)}")
    print(f"First 20: {result[:20]}")
