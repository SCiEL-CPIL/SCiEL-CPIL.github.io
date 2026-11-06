"""
Optimized search for 4-AP-free subsets of [1,256] maximizing 3-AP count.
Uses bitset representation for speed, parallel tempering, and multi-phase search.
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
        _three_aps.append((a - 1, a + d - 1, a + 2 * d - 1))  # 0-indexed

_four_aps = []
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        _four_aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))  # 0-indexed

# Bit masks for each element in 3-APs and 4-APs
_elem_3ap_mask = [0] * N
_elem_3ap_count = [0] * N  # number of 3-APs containing element i
for i, (a, b, c) in enumerate(_three_aps):
    for x in (a, b, c):
        _elem_3ap_mask[x] |= (1 << i)
        _elem_3ap_count[x] += 1

_elem_4ap_mask = [0] * N
_elem_4ap_count = [0] * N
for i, (a, b, c, d) in enumerate(_four_aps):
    for x in (a, b, c, d):
        _elem_4ap_mask[x] |= (1 << i)
        _elem_4ap_count[x] += 1

# Precompute for each element, the 3-APs where it's the middle
# and the other two elements
_elem_middle_3aps = [[] for _ in range(N)]
for a, b, c in _three_aps:
    _elem_middle_3aps[b].append((a, c))

# For each element, the 3-APs where it's the first or last
_elem_end_3aps = [[] for _ in range(N)]
for a, b, c in _three_aps:
    _elem_end_3aps[a].append((b, c, 'first'))
    _elem_end_3aps[c].append((a, b, 'last'))


def count_3aps_bitset(bits):
    """Count 3-APs in a bitset (Python int)."""
    count = 0
    for a, b, c in _three_aps:
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1:
            count += 1
    return count


def has_4ap_bitset(bits):
    """Check if bitset contains any 4-AP."""
    for a, b, c, d in _four_aps:
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1 and (bits >> d) & 1:
            return True
    return False


def is_valid_add_bitset(bits, x):
    """Check if adding element x to bitset creates a 4-AP."""
    mask = _elem_4ap_mask[x]
    # For each 4-AP containing x, check if the other 3 elements are all in bits
    # We need to check each 4-AP individually since we can't easily do this with bit ops
    for idx in range(len(_four_aps)):
        if (mask >> idx) & 1:
            a, b, c, d = _four_aps[idx]
            # Check the other three elements
            others = [(a, b, c, d, 0), (a, b, c, d, 1), (a, b, c, d, 2), (a, b, c, d, 3)]
            others = [o for o in others if o[4] != x]
            if len(others) == 3:
                all_present = True
                for _, _, _, _, pos in others:
                    elem = _four_aps[idx][pos]
                    if elem != x and not ((bits >> elem) & 1):
                        all_present = False
                        break
                if all_present:
                    return False
    return True


def is_valid_add_fast(bits, x):
    """Faster validity check using precomputed data."""
    # For each 4-AP containing x, check if the other 3 are in bits
    for idx in range(_elem_4ap_count[x]):
        # We need to iterate through 4-APs containing x
        pass
    # Simpler approach: iterate through 4-APs containing x
    for a, b, c, d in _four_aps:
        if a == x or b == x or c == x or d == x:
            others = [y for y in (a, b, c, d) if y != x]
            if all((bits >> y) & 1 for y in others):
                return False
    return True


def delta_add_bitset(bits, x):
    """Change in 3-AP count when adding x to bitset."""
    delta = 0
    for a, b, c in _three_aps:
        if a == x or b == x or c == x:
            others = [y for y in (a, b, c) if y != x]
            if all((bits >> y) & 1 for y in others):
                delta += 1
    return delta


def delta_remove_bitset(bits, x):
    """Change in 3-AP count when removing x from bitset (negative = loss)."""
    delta = 0
    for a, b, c in _three_aps:
        if a == x or b == x or c == x:
            others = [y for y in (a, b, c) if y != x]
            if all((bits >> y) & 1 for y in others):
                delta += 1
    return -delta


def swap_gain_bitset(bits, x_out, x_in):
    """Net change in 3-AP count for swapping x_out out and x_in in."""
    # Loss from removing x_out
    loss = 0
    for a, b, c in _three_aps:
        if a == x_out or b == x_out or c == x_out:
            others = [y for y in (a, b, c) if y != x_out]
            if all((bits >> y) & 1 for y in others):
                loss += 1
    
    # Gain from adding x_in to bits without x_out
    bits_no_out = bits & ~(1 << x_out)
    gain = 0
    for a, b, c in _three_aps:
        if a == x_in or b == x_in or c == x_in:
            others = [y for y in (a, b, c) if y != x_in]
            if all((bits_no_out >> y) & 1 for y in others):
                gain += 1
    
    return gain - loss


def is_valid_swap_bitset(bits, x_out, x_in):
    """Check if swapping x_out for x_in is valid."""
    bits_no_out = bits & ~(1 << x_out)
    # Check if adding x_in to bits_no_out creates a 4-AP
    for a, b, c, d in _four_aps:
        if a == x_in or b == x_in or c == x_in or d == x_in:
            others = [y for y in (a, b, c, d) if y != x_in]
            if all((bits_no_out >> y) & 1 for y in others):
                return False
    return True


def greedy_construct_bitset(seed=0):
    """Greedy construction using bitset."""
    random.seed(seed)
    bits = 0
    available = list(range(N))
    
    while available:
        best_gain = -1
        best_elem = None
        candidates = []
        for x in available:
            if not is_valid_add_fast(bits, x):
                continue
            gain = delta_add_bitset(bits, x)
            candidates.append((gain, x))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_k = min(3, len(candidates))
        if random.random() < 0.7:
            best_elem = candidates[0][1]
        else:
            best_elem = random.choice([c[1] for c in candidates[:top_k]])
        
        bits |= (1 << best_elem)
        available.remove(best_elem)
    
    return bits


def local_search_1move_bitset(bits, max_iters=3000):
    """Local search with add/remove moves on bitset."""
    current_score = count_3aps_bitset(bits)
    best_bits = bits
    best_score = current_score
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding elements
        for x in range(N):
            if not (bits >> x) & 1:
                if not is_valid_add_fast(bits, x):
                    continue
                gain = delta_add_bitset(bits, x)
                if gain > 0:
                    bits |= (1 << x)
                    current_score += gain
                    improved = True
                    break
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_bits = bits
            continue
        
        # Try removing elements
        for x in range(N):
            if (bits >> x) & 1:
                loss = -delta_remove_bitset(bits, x)
                if loss < 0:
                    bits &= ~(1 << x)
                    current_score += loss
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_bits


def local_search_swap_bitset(bits, max_iters=2000):
    """Local search with swap moves on bitset."""
    current_score = count_3aps_bitset(bits)
    best_bits = bits
    best_score = current_score
    
    for _ in range(max_iters):
        improved = False
        best_swap = None
        best_gain = 0
        
        in_set = [x for x in range(N) if (bits >> x) & 1]
        not_in_set = [x for x in range(N) if not (bits >> x) & 1]
        
        for x_out in in_set:
            for x_in in not_in_set:
                if not is_valid_swap_bitset(bits, x_out, x_in):
                    continue
                g = swap_gain_bitset(bits, x_out, x_in)
                if g > best_gain:
                    best_gain = g
                    best_swap = (x_out, x_in)
        
        if best_swap is not None and best_gain > 0:
            x_out, x_in = best_swap
            bits &= ~(1 << x_out)
            bits |= (1 << x_in)
            current_score += best_gain
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_bits = bits
        
        if not improved:
            break
    
    return best_bits


def parallel_tempering(bits_list, temps, n_sweeps=1000):
    """Parallel tempering (replica exchange) with swap and add/remove moves."""
    n_chains = len(bits_list)
    best_bits = max(bits_list, key=count_3aps_bitset)
    best_score = count_3aps_bitset(best_bits)
    
    for sweep in range(n_sweeps):
        # Update each chain
        for i in range(n_chains):
            bits = bits_list[i]
            T = temps[i]
            score = count_3aps_bitset(bits)
            
            # Random move
            move_type = random.random()
            new_bits = bits
            
            if move_type < 0.4:
                # Swap
                in_set = [x for x in range(N) if (bits >> x) & 1]
                not_in_set = [x for x in range(N) if not (bits >> x) & 1]
                if in_set and not_in_set:
                    x_out = random.choice(in_set)
                    x_in = random.choice(not_in_set)
                    if is_valid_swap_bitset(bits, x_out, x_in):
                        g = swap_gain_bitset(bits, x_out, x_in)
                        new_bits = (bits & ~(1 << x_out)) | (1 << x_in)
                        if g > 0 or random.random() < math.exp(g / max(T, 0.1)):
                            bits = new_bits
                            score += g
            elif move_type < 0.7:
                # Add
                not_in_set = [x for x in range(N) if not (bits >> x) & 1]
                if not_in_set:
                    x = random.choice(not_in_set)
                    if is_valid_add_fast(bits, x):
                        gain = delta_add_bitset(bits, x)
                        new_bits = bits | (1 << x)
                        if gain > 0 or random.random() < math.exp(gain / max(T, 0.1)):
                            bits = new_bits
                            score += gain
            else:
                # Remove
                in_set = [x for x in range(N) if (bits >> x) & 1]
                if in_set:
                    x = random.choice(in_set)
                    loss = -delta_remove_bitset(bits, x)
                    new_bits = bits & ~(1 << x)
                    if loss > 0 or random.random() < math.exp(loss / max(T, 0.1)):
                        bits = new_bits
                        score += loss
            
            bits_list[i] = bits
            if score > best_score:
                best_score = score
                best_bits = bits
        
        # Exchange adjacent chains with probability based on temperature difference
        for i in range(n_chains - 1):
            if random.random() < 0.3:
                s1 = count_3aps_bitset(bits_list[i])
                s2 = count_3aps_bitset(bits_list[i + 1])
                T1 = temps[i]
                T2 = temps[i + 1]
                # Accept exchange if it doesn't increase "free energy"
                # P(accept) = exp(-(s1/T2 + s2/T1 - s1/T1 - s2/T2))
                delta_fe = (s1 / T2 + s2 / T1) - (s1 / T1 + s2 / T2)
                if delta_fe <= 0 or random.random() < math.exp(-delta_fe):
                    bits_list[i], bits_list[i + 1] = bits_list[i + 1], bits_list[i]
    
    return best_bits, best_score


def construct():
    start_time = time.time()
    
    # Phase 1: Multiple greedy constructions
    best_bits = 0
    best_score = -1
    
    for seed in range(8):
        if time.time() - start_time > MAX_TIME * 0.1:
            break
        bits = greedy_construct_bitset(seed)
        score = count_3aps_bitset(bits)
        if score > best_score:
            best_score = score
            best_bits = bits
    
    # Phase 2: Local search refinement
    bits = local_search_1move_bitset(best_bits, max_iters=2000)
    score = count_3aps_bitset(bits)
    if score > best_score:
        best_score = score
        best_bits = bits
    
    bits = local_search_swap_bitset(best_bits, max_iters=1500)
    score = count_3aps_bitset(bits)
    if score > best_score:
        best_score = score
        best_bits = bits
    
    bits = local_search_1move_bitset(best_bits, max_iters=1500)
    score = count_3aps_bitset(bits)
    if score > best_score:
        best_score = score
        best_bits = bits
    
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME * 0.35:
        if has_4ap_bitset(best_bits):
            best_bits = greedy_construct_bitset(0)
            best_bits = local_search_1move_bitset(best_bits)
        return sorted([i + 1 for i in range(N) if (best_bits >> i) & 1])
    
    # Phase 3: Parallel tempering
    remaining_time = MAX_TIME - elapsed
    n_chains = 6
    base_temp = 5.0
    temps = [base_temp * (1.5 ** i) for i in range(n_chains)]
    
    # Initialize chains with perturbations of best solution
    chains = [best_bits]
    for _ in range(n_chains - 1):
        perturbed = best_bits
        k_remove = random.randint(3, 8)
        elems = [i for i in range(N) if (perturbed >> i) & 1]
        random.shuffle(elems)
        for x in elems[:k_remove]:
            perturbed &= ~(1 << x)
        chains.append(perturbed)
    
    n_sweeps = max(200, int(remaining_time / 0.05))
    best_bits_pt, best_score_pt = parallel_tempering(chains, temps, n_sweeps=n_sweeps)
    
    if best_score_pt > best_score:
        best_score = best_score_pt
        best_bits = best_bits_pt
    
    # Phase 4: Final local search polish
    for _ in range(3):
        if time.time() - start_time > MAX_TIME - 5:
            break
        bits = local_search_1move_bitset(best_bits, max_iters=1000)
        score = count_3aps_bitset(bits)
        if score > best_score:
            best_score = score
            best_bits = bits
        
        bits = local_search_swap_bitset(best_bits, max_iters=800)
        score = count_3aps_bitset(bits)
        if score > best_score:
            best_score = score
            best_bits = bits
    
    # Final verification
    if has_4ap_bitset(best_bits):
        best_bits = greedy_construct_bitset(0)
        best_bits = local_search_1move_bitset(best_bits)
    
    return sorted([i + 1 for i in range(N) if (best_bits >> i) & 1])


if __name__ == "__main__":
    result = construct()
    bits = 0
    for x in result:
        bits |= (1 << (x - 1))
    print(f"Size: {len(result)}, 3-APs: {count_3aps_bitset(bits)}")
    print(f"Has 4-AP: {has_4ap_bitset(bits)}")
    print(f"First 20: {result[:20]}")
