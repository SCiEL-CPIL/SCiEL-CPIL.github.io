# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    # For the specific hard case q=3, n=6, radius=1, use specialized group-structured ILS
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback for other cases
    if total_words > 10000:
        return _generic_construct(q, n, radius, deadline)
        
    # Generic bitset-based solver for small spaces
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    # Precompute neighbors as bitsets
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbor lists and bitset masks
    neighbor_indices = [[] for _ in range(total_words)]
    neighbors = [0] * total_words
    for i, t in enumerate(all_tuples):
        mask = 0
        mask |= (1 << i)
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    # Precompute perturbation vectors in Z_3^6 for structured swaps
    # Small perturbations: single-coordinate changes and 2-coordinate changes
    perturbations = []
    for pos in range(n):
        for delta in (1, 2):  # +1 and -1 mod 3
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations.append(p)
    
    def get_uncovered_mask(cover_count):
        mask = 0
        for i in range(total_words):
            if cover_count[i] == 0:
                mask |= (1 << i)
        return mask

    def greedy_start():
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask
        
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            
            if best_idx == -1 or best_cnt == 0:
                break
                
            code_set.add(best_idx)
            m = neighbors[best_idx]
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~m
            
        return code_set, cover_count, uncovered_mask

    def reduce_code(code_set, cover_count):
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                redundant = True
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    changed = True
                    break
        return code_set, cover_count

    def repair(code_set, cover_count, uncovered_mask):
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1 or best_cnt == 0:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
        return code_set, cover_count, uncovered_mask

    def structured_swap(code_set, cover_count, rng):
        """
        Perform a structured swap: remove one codeword c, add c + d for a small perturbation d.
        Returns (new_code_set, new_cover_count, success) where success means the code size didn't increase.
        """
        if not code_set:
            return code_set, cover_count, False
        
        c = rng.choice(list(code_set))
        d = rng.choice(perturbations)
        
        # Compute c + d mod 3
        c_tuple = all_tuples[c]
        new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
        new_idx = idx_map[new_tuple]
        
        # If new_idx is already in code_set, skip
        if new_idx in code_set:
            return code_set, cover_count, False
        
        # Try the swap: remove c, add new_idx
        # First, check if removing c leaves any word uncovered
        # We'll do the swap and check
        for idx in neighbor_indices[c]:
            cover_count[idx] -= 1
        
        uncovered_after_removal = 0
        for idx in neighbor_indices[c]:
            if cover_count[idx] == 0:
                uncovered_after_removal |= (1 << idx)
        
        # If no uncovered words, the swap is valid (and we reduced size by 0, same size)
        # If uncovered words, we need to add new_idx and see if it helps
        code_set.remove(c)
        
        # Add new_idx
        code_set.add(new_idx)
        for idx in neighbor_indices[new_idx]:
            cover_count[idx] += 1
        
        # Check if the cover is valid
        uncovered_mask = 0
        for i in range(total_words):
            if cover_count[i] == 0:
                uncovered_mask |= (1 << i)
        
        if uncovered_mask == 0:
            return code_set, cover_count, True
        else:
            # Revert
            code_set.remove(new_idx)
            for idx in neighbor_indices[new_idx]:
                cover_count[idx] -= 1
            code_set.add(c)
            for idx in neighbor_indices[c]:
                cover_count[idx] += 1
            return code_set, cover_count, False

    def local_search_1exchange(code_set, cover_count, rng, max_iters=200):
        """
        Try 1-exchange moves: replace codeword c with c+d for small d.
        Accept if the code remains valid and size doesn't increase.
        """
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]
            
            if new_idx in code_set:
                continue
            
            # Try swap
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
            code_set.remove(c)
            
            code_set.add(new_idx)
            for idx in neighbor_indices[new_idx]:
                cover_count[idx] += 1
            
            # Check validity
            valid = True
            for i in range(total_words):
                if cover_count[i] == 0:
                    valid = False
                    break
            
            if valid:
                improved = True
            else:
                # Revert
                code_set.remove(new_idx)
                for idx in neighbor_indices[new_idx]:
                    cover_count[idx] -= 1
                code_set.add(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] += 1
        
        return code_set, cover_count, improved

    # Main ILS Loop
    rng = random.Random(42)
    
    # Initial solution
    code_set, cover_count, uncovered_mask = greedy_start()
    code_set, cover_count = reduce_code(code_set, cover_count)
    uncovered_mask = get_uncovered_mask(cover_count)
    
    best_size = len(code_set)
    best_code = list(code_set)
    
    # ILS with structured perturbations
    iteration = 0
    while time.time() < deadline - 2:
        iteration += 1
        
        # Perturb: structured swap or random removal
        if rng.random() < 0.5:
            # Structured swap: remove one, add a nearby one
            code_set, cover_count, _ = structured_swap(code_set, cover_count, rng)
            # If the swap didn't change size, do a local search
            code_set, cover_count, _ = local_search_1exchange(code_set, cover_count, rng, max_iters=50)
        else:
            # Random removal of k codewords
            k = rng.randint(2, 5)
            current_code = list(code_set)
            if len(current_code) < k:
                k = len(current_code)
            to_remove = rng.sample(current_code, k)
            for c in to_remove:
                code_set.remove(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
            uncovered_mask = get_uncovered_mask(cover_count)
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
            code_set, cover_count = reduce_code(code_set, cover_count)
        
        # Check improvement
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            if best_size == 73:
                break

    result = [list(all_tuples[i]) for i in best_code]
    return result

def _generic_bitset_construct(q, n, radius, deadline):
    total_words = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    neighbors = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        mask = 0
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t[k] != t_j[k])
            if dist <= radius:
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    code_set = set()
    cover_count = [0] * total_words
    uncovered_mask = full_mask
    
    while uncovered_mask and time.time() < deadline - 10:
        best_idx = -1
        best_cnt = -1
        for i in range(total_words):
            if i in code_set:
                continue
            new_covered = neighbors[i] & uncovered_mask
            cnt = new_covered.bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_idx == -1: break
        code_set.add(best_idx)
        for idx in neighbor_indices[best_idx]:
            cover_count[idx] += 1
        uncovered_mask &= ~neighbors[best_idx]
        
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for c in list(code_set):
            redundant = True
            for idx in neighbor_indices[c]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                code_set.remove(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
                changed = True
                break
                
    result = [list(all_tuples[i]) for i in code_set]
    return result

def _generic_construct(q, n, radius, deadline):
    all_words = list(itertools.product(range(q), repeat=n))
    def dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])
    
    uncovered = set(all_words)
    code = []
    while uncovered and time.time() < deadline - 10:
        best = None
        best_cnt = -1
        for w in all_words:
            if w in set(code): continue
            cnt = sum(1 for u in uncovered if dist(w, u) <= radius)
            if cnt > best_cnt:
                best_cnt = cnt
                best = w
        if not best: break
        code.append(best)
        uncovered = {u for u in uncovered if dist(best, u) > radius}
    return [list(w) for w in code]

# EVOLVE-BLOCK-END
