import time
import random
import math
from fractions import Fraction

N = 256
TIME_LIMIT = 55.0  # Safety margin before 60s kill
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Precompute AP conflict pairs
ap_pairs = [[] for _ in range(N + 1)]

def precompute_ap_pairs():
    for x in range(1, N + 1):
        pairs = set()
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if y <= N and z <= N:
                pairs.add((y, z))
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        ap_pairs[x] = list(pairs)

precompute_ap_pairs()

# Precompute reciprocal values as floats
recip_float = [0.0] * (N + 1)
for i in range(1, N + 1):
    recip_float[i] = 1.0 / i

def can_add(bitset, x):
    """Check if adding x creates a 3-AP."""
    for a, b in ap_pairs[x]:
        if (bitset & (1 << (a - 1))) and (bitset & (1 << (b - 1))):
            return False
    return True

def greedy_construct(order):
    """Greedy construction: add elements in order if no 3-AP."""
    bitset = 0
    score_float = 0.0
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
            score_float += recip_float[x]
    return bitset, score_float

def greedy_base11():
    """
    Greedy using base-11 digit-free numbers as initial seed.
    Numbers with no zero digit in base 11 are 3-AP-free (Cantor-like construction).
    Then greedily add remaining numbers in ascending order.
    """
    # Find numbers 1..256 with no zero digit in base 11
    base11_free = []
    for x in range(1, N + 1):
        n = x
        has_zero = False
        while n > 0:
            if n % 11 == 0:
                has_zero = True
                break
            n //= 11
        if not has_zero:
            base11_free.append(x)
    
    # Start with base-11-free set, then greedily add others in ascending order
    bitset = 0
    score_float = 0.0
    for x in base11_free:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
            score_float += recip_float[x]
    
    # Now greedily add remaining numbers in ascending order (small = high reciprocal)
    for x in range(1, N + 1):
        if not (bitset & (1 << (x - 1))):
            if can_add(bitset, x):
                bitset |= (1 << (x - 1))
                score_float += recip_float[x]
    
    return bitset, score_float

def local_search_systematic(bitset, score_float, deadline):
    """
    Systematic local search: repeatedly try remove-and-best-add moves,
    2-swap moves, and pure additions.
    """
    best_bitset = bitset
    best_score = score_float
    current_bitset = bitset
    current_score = score_float
    
    # Precompute sorted nonmembers by reciprocal (descending)
    all_nonmembers_sorted = list(range(1, N + 1))
    all_nonmembers_sorted.sort(key=lambda x: -recip_float[x])
    
    def get_members(bs):
        return [i for i in range(1, N + 1) if bs & (1 << (i - 1))]
    
    def get_nonmembers(bs):
        return [i for i in range(1, N + 1) if not (bs & (1 << (i - 1)))]
    
    no_improve = 0
    max_no_improve = 150
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Reset to best and perturb
            current_bitset = best_bitset
            current_score = best_score
            no_improve = 0
            members = get_members(current_bitset)
            if len(members) > 1:
                # Remove 1-2 random elements to perturb
                k = random.randint(1, 2)
                for x_rem in random.sample(members, min(k, len(members))):
                    current_bitset &= ~(1 << (x_rem - 1))
                    current_score -= recip_float[x_rem]
        
        members = get_members(current_bitset)
        nonmembers = get_nonmembers(current_bitset)
        
        improved = False
        
        # Try pure additions (in order of best reciprocal first)
        for x_add in all_nonmembers_sorted:
            if not (current_bitset & (1 << (x_add - 1))):
                if can_add(current_bitset, x_add):
                    current_bitset |= (1 << (x_add - 1))
                    current_score += recip_float[x_add]
                    improved = True
        
        # Try remove-and-best-add (1-swap)
        for x_rem in members[:]:
            trial = current_bitset & ~(1 << (x_rem - 1))
            trial_score = current_score - recip_float[x_rem]
            best_add = None
            best_add_score = -1.0
            for x_add in all_nonmembers_sorted:
                if not (trial & (1 << (x_add - 1))):
                    if can_add(trial, x_add):
                        s = trial_score + recip_float[x_add]
                        if s > best_add_score:
                            best_add_score = s
                            best_add = x_add
            if best_add is not None and best_add_score > current_score + 1e-15:
                current_bitset = trial | (1 << (best_add - 1))
                current_score = best_add_score
                improved = True
                members = get_members(current_bitset)
                nonmembers = get_nonmembers(current_bitset)
        
        # Try 2-swap: remove 2 elements, add 2 elements
        if len(members) >= 2 and len(nonmembers) >= 2:
            # Sample a subset to keep it fast
            sample_size = min(10, len(members))
            rem_candidates = random.sample(members, sample_size)
            add_candidates = [x for x in all_nonmembers_sorted[:20]]
            
            for i in range(len(rem_candidates)):
                for j in range(i + 1, len(rem_candidates)):
                    x1, x2 = rem_candidates[i], rem_candidates[j]
                    trial = current_bitset & ~(1 << (x1 - 1)) & ~(1 << (x2 - 1))
                    trial_score = current_score - recip_float[x1] - recip_float[x2]
                    
                    best_add1 = None
                    best_add2 = None
                    best_total = -1.0
                    
                    for a1 in add_candidates:
                        if (trial & (1 << (a1 - 1))):
                            continue
                        if not can_add(trial, a1):
                            continue
                        trial1 = trial | (1 << (a1 - 1))
                        score1 = trial_score + recip_float[a1]
                        for a2 in add_candidates:
                            if a2 == a1 or (trial1 & (1 << (a2 - 1))):
                                continue
                            if not can_add(trial1, a2):
                                continue
                            total = score1 + recip_float[a2]
                            if total > best_total:
                                best_total = total
                                best_add1 = a1
                                best_add2 = a2
                    
                    if best_add1 is not None and best_total > current_score + 1e-15:
                        current_bitset = trial | (1 << (best_add1 - 1)) | (1 << (best_add2 - 1))
                        current_score = best_total
                        improved = True
                        members = get_members(current_bitset)
                        nonmembers = get_nonmembers(current_bitset)
        
        if not improved:
            no_improve += 1
            # Random perturbation
            if members and nonmembers:
                x_rem = random.choice(members)
                x_add = random.choice(nonmembers)
                trial = current_bitset & ~(1 << (x_rem - 1))
                if can_add(trial, x_add):
                    new_score = current_score - recip_float[x_rem] + recip_float[x_add]
                    if new_score > current_score or random.random() < 0.15:
                        current_bitset = trial | (1 << (x_add - 1))
                        current_score = new_score
        else:
            no_improve = 0
        
        if current_score > best_score + 1e-15:
            best_score = current_score
            best_bitset = current_bitset
    
    return best_bitset, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bitset = 0
    best_score = 0.0
    
    # Strategy 1: Standard greedy (ascending)
    order1 = list(range(1, N + 1))
    bs1, sc1 = greedy_construct(order1)
    if sc1 > best_score:
        best_score = sc1
        best_bitset = bs1
    
    # Strategy 2: Base-11 digit-free greedy
    bs2, sc2 = greedy_base11()
    if sc2 > best_score:
        best_score = sc2
        best_bitset = bs2
    
    # Strategy 3: Randomized greedy (multiple trials)
    for trial in range(30):
        if time.time() > deadline - 25:
            break
        # Weighted random ordering: smaller numbers get higher priority
        weights = [1.0 / (x ** 0.7) for x in range(1, N + 1)]
        total_w = sum(weights)
        order = list(range(1, N + 1))
        # Create a random permutation biased toward small numbers
        for i in range(N - 1, 0, -1):
            # Pick random index weighted by reciprocal
            r = random.random() * sum(weights[j] for j in range(i + 1))
            cum = 0
            pick = 0
            for j in range(i + 1):
                cum += weights[j]
                if r <= cum:
                    pick = j
                    break
            order[i], order[pick] = order[pick], order[i]
        bs, sc = greedy_construct(order)
        if sc > best_score:
            best_score = sc
            best_bitset = bs
    
    # Systematic local search
    best_bitset, best_score = local_search_systematic(best_bitset, best_score, deadline)
    
    # Final polish: add any remaining in order of best reciprocal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (best_bitset & (1 << (x - 1))):
                if can_add(best_bitset, x):
                    best_bitset |= (1 << (x - 1))
                    improved = True
    
    # One more local search pass with remaining time
    if time.time() < deadline - 5:
        best_bitset, best_score = local_search_systematic(best_bitset, best_score, deadline)
    
    result = []
    for x in range(1, N + 1):
        if best_bitset & (1 << (x - 1)):
            result.append(x)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
