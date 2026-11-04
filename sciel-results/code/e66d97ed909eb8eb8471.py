import time
import random
from math import log, log2
import numpy as np
from itertools import combinations

def fast_score(U):
    """
    Score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def generate_bit_sparse_candidates(max_bits, max_weight, exclude_set):
    """
    Generate candidates with sparse binary representations (at most max_weight 1-bits).
    """
    candidates = set()
    for bits in range(1, max_bits + 1):
        # Limit combinations to avoid explosion
        max_combo = min(max_weight, bits)
        for combo in combinations(range(bits), max_combo):
            val = 0
            for b in combo:
                val |= (1 << b)
            if val > 0 and val not in exclude_set:
                candidates.add(val)
    return list(candidates)

def local_search_bit_aware(U, deadline, max_iter=10000):
    """
    Local search that respects binary structure: prefer moves that maintain
    sparsity of binary representations.
    """
    U = sorted(set(U))
    current_score = fast_score(U)
    if current_score < 0:
        return U, current_score
    
    best_local_score = current_score
    best_local_set = list(U)
    
    no_improvement_count = 0
    max_no_improvement = 15
    temp = 0.1
    cooling = 0.999
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        n = len(U)
        if n < 2:
            break
            
        max_u = max(U)
        max_bits = max_u.bit_length() + 2
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to ADD an element with sparse bits
            if n < 25:  # Limit size to keep score computation fast
                candidates = generate_bit_sparse_candidates(max_bits, 3, set(U))
                if not candidates:
                    # Fallback to random
                    candidates = [random.randint(1, max_u * 2) for _ in range(10)]
                
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U or c <= 0:
                        continue
                    new_U = sorted(U + [c])
                    new_score = fast_score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                    elif new_score > current_score - temp:
                        # Metropolis accept
                        if random.random() < 1.0 / (1.0 + (current_score - new_score) / temp):
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
                            
        elif move_type < 0.6:
            # Try to REMOVE an element
            if n > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = fast_score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                    elif new_score > current_score - temp:
                        if random.random() < 1.0 / (1.0 + (current_score - new_score) / temp):
                            U = new_U
                            current_score = new_score
                            improved = True
        else:
            # Try to REPLACE with a bit-flip variant
            if n > 1:
                idx = random.randint(1, n - 1)
                old_val = U[idx]
                
                # Generate variants by flipping one bit
                variants = []
                for bit in range(old_val.bit_length() + 2):
                    new_val = old_val ^ (1 << bit)
                    if 0 < new_val <= max_u * 2 and new_val not in U:
                        variants.append(new_val)
                
                if variants:
                    random.shuffle(variants)
                    for new_val in variants[:10]:
                        new_U = U.copy()
                        new_U[idx] = new_val
                        new_U = sorted(set(new_U))
                        if len(new_U) != n:
                            continue
                        new_score = fast_score(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
                        elif new_score > current_score - temp:
                            if random.random() < 1.0 / (1.0 + (current_score - new_score) / temp):
                                U = new_U
                                current_score = new_score
                                improved = True
                                break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
            temp *= cooling
            
    return best_local_set, best_local_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: evaluator kills at 150s
    
    random.seed(12345)
    
    # Multiple seed strategies
    seeds = []
    
    # Basic seeds
    seeds.append([0, 1, 4, 9, 11])
    seeds.append([0, 1, 2, 4, 8, 16])
    seeds.append([0, 1, 3, 7, 15, 31])
    
    # Bit-sparse seeds (from inspiration 1)
    for max_bits in [8, 10, 12]:
        for max_weight in [2, 3]:
            candidates = generate_bit_sparse_candidates(max_bits, max_weight, set())
            if len(candidates) > 5:
                # Take a random subset
                sample_size = min(10, len(candidates))
                sample = random.sample(candidates, sample_size)
                seed = sorted([0] + sample)
                if len(set(seed)) == len(seed):
                    seeds.append(seed)
    
    # Geometric seeds with various parameters
    for base in [1, 2, 3, 5]:
        for ratio in [1.5, 2.0, 2.5, 3.0]:
            for n in [5, 6, 7, 8]:
                U = [0]
                val = base
                for _ in range(n - 1):
                    U.append(val)
                    val = int(val * ratio + 0.5)
                if 0 in U and len(set(U)) == len(U):
                    seeds.append(U)
    
    # Base-expansion hierarchy seeds
    for b1 in [2, 3, 4]:
        for b2 in [2, 3, 4, 5]:
            for b3 in [2, 3, 4]:
                for levels in [5, 6, 7, 8]:
                    bases = [b1, b2, b3]
                    if levels > len(bases):
                        bases = bases + [bases[-1]] * (levels - len(bases))
                    else:
                        bases = bases[:levels]
                    U = [0]
                    val = 1
                    for level in range(levels):
                        base = bases[level]
                        val = val * base
                        U.append(val)
                    if 0 in U and len(set(U)) == len(U):
                        seeds.append(U)
    
    # Random seeds
    for _ in range(20):
        n = random.randint(4, 8)
        max_v = random.randint(50, 300)
        R = [0] + sorted(random.sample(range(1, max_v), n - 1))
        seeds.append(R)
    
    best_set = seeds[0]
    best_score = fast_score(best_set)
    
    # Evaluate all seeds and keep the best
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        s = fast_score(seed)
        if s > best_score:
            best_score = s
            best_set = seed
    
    # Local search from best seed
    if time.time() < deadline - 15:
        refined, refined_score = local_search_bit_aware(best_set, deadline - 10, max_iter=10000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Local search from other promising seeds
    for seed in seeds[:10]:
        if time.time() > deadline - 10:
            break
        if seed == best_set:
            continue
        s = fast_score(seed)
        if s > best_score * 0.95:  # Only refine seeds that are close to best
            refined, refined_score = local_search_bit_aware(seed, deadline - 5, max_iter=5000)
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined
    
    # Final polish: more iterations on the best set
    if time.time() < deadline - 3:
        refined, refined_score = local_search_bit_aware(best_set, deadline - 1, max_iter=10000)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    return best_set

if __name__ == "__main__":
    print(f"Best set: {construct()}")
