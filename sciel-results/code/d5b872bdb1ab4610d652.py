"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 16
    k = 4

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)

    m = len(all_masks)
    mask_to_idx = {mask: i for i, mask in enumerate(all_masks)}

    # Precompute forbidden pairs
    # forbidden[i][j] is a list of indices k such that i, j, k form a sunflower
    forbidden = [[[] for _ in range(m)] for _ in range(m)]
    
    # To speed up precomputation, we can iterate over all triples? 
    # C(1820, 3) is huge (~1 billion). 
    # Instead, iterate over pairs (i, j) and find valid k's.
    # For each pair (i, j), determine the constraints on k.
    
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            mj = all_masks[j]
            inter = mi & mj
            diff_i = mi & ~mj
            diff_j = mj & ~mi
            
            # k must contain inter
            # k must be disjoint from diff_i and diff_j
            # k must have size 4
            
            # Let allowed_mask be the bits that k can have.
            # k must be subset of (~diff_i & ~diff_j)
            # k must be superset of inter
            # Note: inter is disjoint from diff_i and diff_j.
            
            avail_mask = (~diff_i & ~diff_j) & ((1 << n) - 1)
            # k must be a 4-subset of avail_mask that contains inter
            
            # Let free_mask = avail_mask & ~inter
            # We need to choose (4 - popcount(inter)) elements from free_mask
            # such that inter is included.
            
            core_size = bin(inter).count('1')
            if core_size > 4:
                continue # Should not happen for distinct 4-sets with intersection > 4? No, max intersection is 4 if equal, but i!=j so max 3?
                # Wait, if i != j, max intersection is 3?
                # If |A|=4, |B|=4, A!=B, max |A cap B| is 3.
                # So core_size <= 3.
                
            need = 4 - core_size
            if need < 0:
                continue
            
            free_mask = avail_mask & ~inter
            
            # Iterate through all combinations of 'need' elements from free_mask
            # free_mask has at most 16 - core_size - 2*need? 
            # Actually, the number of free elements is |avail_mask| - core_size.
            # We just need to find all 4-subsets of avail_mask containing inter.
            # This is equivalent to choosing 'need' elements from free_mask.
            
            # Get list of bits in free_mask
            free_bits = []
            fm = free_mask
            while fm:
                lsb = fm & -fm
                free_bits.append(lsb)
                fm -= lsb
            
            # Generate combinations of 'need' from free_bits
            # Since need is small (0 to 4), and free_bits is small, this is fast.
            if need == 0:
                # k must be exactly inter (size 4). But i!=j so inter size < 4.
                # So no such k unless inter size is 4, which implies i=j.
                pass
            elif need == 1:
                for bit in free_bits:
                    mk = inter | bit
                    if mk in mask_to_idx:
                        k_idx = mask_to_idx[mk]
                        if k_idx != i and k_idx != j:
                            forbidden[i][j].append(k_idx)
                            forbidden[j][i].append(k_idx)
            elif need == 2:
                for b1_idx in range(len(free_bits)):
                    b1 = free_bits[b1_idx]
                    for b2_idx in range(b1_idx + 1, len(free_bits)):
                        b2 = free_bits[b2_idx]
                        mk = inter | b1 | b2
                        if mk in mask_to_idx:
                            k_idx = mask_to_idx[mk]
                            if k_idx != i and k_idx != j:
                                forbidden[i][j].append(k_idx)
                                forbidden[j][i].append(k_idx)
            elif need == 3:
                for b1_idx in range(len(free_bits)):
                    b1 = free_bits[b1_idx]
                    for b2_idx in range(b1_idx + 1, len(free_bits)):
                        b2 = free_bits[b2_idx]
                        for b3_idx in range(b2_idx + 1, len(free_bits)):
                            b3 = free_bits[b3_idx]
                            mk = inter | b1 | b2 | b3
                            if mk in mask_to_idx:
                                k_idx = mask_to_idx[mk]
                                if k_idx != i and k_idx != j:
                                    forbidden[i][j].append(k_idx)
                                    forbidden[j][i].append(k_idx)
            elif need == 4:
                # inter is 0. Choose 4 from free_bits.
                # This happens when A and B are disjoint.
                for b1_idx in range(len(free_bits)):
                    b1 = free_bits[b1_idx]
                    for b2_idx in range(b1_idx + 1, len(free_bits)):
                        b2 = free_bits[b2_idx]
                        for b3_idx in range(b2_idx + 1, len(free_bits)):
                            b3 = free_bits[b3_idx]
                            for b4_idx in range(b3_idx + 1, len(free_bits)):
                                b4 = free_bits[b4_idx]
                                mk = b1 | b2 | b3 | b4
                                if mk in mask_to_idx:
                                    k_idx = mask_to_idx[mk]
                                    if k_idx != i and k_idx != j:
                                        forbidden[i][j].append(k_idx)
                                        forbidden[j][i].append(k_idx)

    def mask_to_list(mask):
        """Convert bitmask to sorted list of 1..16 elements."""
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(i + 1)
        return result

    best_size = 0
    best_family_idx = []

    def greedy_construct(order):
        family = []
        conf_count = [0] * m
        
        for idx in order:
            if time.time() > deadline:
                break
            
            if conf_count[idx] == 0:
                family.append(idx)
                # Update conf_count for all pairs involving idx
                for a_idx in family[:-1]:
                    forbids = forbidden[a_idx][idx]
                    for c_idx in forbids:
                        conf_count[c_idx] += 1
        
        return family

    # Strategy 1: Greedy in natural order
    family = greedy_construct(list(range(m)))
    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles
    random.seed(12345)
    max_attempts = 200 # Reduced to allow local search time
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break

        order = list(range(m))
        random.shuffle(order)
        family = greedy_construct(order)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    def local_search(family):
        if time.time() > deadline - 5:
            return family
        
        current_family = family[:]
        in_family = set(current_family)
        conf_count = [0] * m
        
        # Initialize conf_count
        for i in range(len(current_family)):
            for j in range(i+1, len(current_family)):
                for c_idx in forbidden[current_family[i]][current_family[j]]:
                    conf_count[c_idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add a set
            for c_idx in range(m):
                if c_idx not in in_family and conf_count[c_idx] == 0:
                    current_family.append(c_idx)
                    in_family.add(c_idx)
                    for a_idx in current_family[:-1]:
                        for f_idx in forbidden[a_idx][c_idx]:
                            conf_count[f_idx] += 1
                    improved = True
                    break
            
            if not improved:
                # 2. Try swaps
                # To make this efficient, we only try swaps that might help.
                # We iterate through current_family members to remove, and candidates to add.
                # This is O(|F| * M) which is ~ 250 * 1820 = 455,000 checks.
                # Each check involves updating conf_count temporarily?
                # Too slow if done naively.
                
                # Heuristic Swap:
                # Pick a random member A to remove.
                # Find a valid B to add.
                # If found, accept.
                
                # To keep it simple and fast, we just try to remove one element 
                # and see if we can add a different one that was previously blocked.
                
                # Let's try a limited number of random swaps
                for _ in range(10):
                    if time.time() > deadline - 2:
                        break
                    
                    if not current_family:
                        break
                    
                    remove_idx = random.choice(current_family)
                    
                    # Temporarily remove
                    # Decrement conf_count for pairs involving remove_idx
                    for a_idx in current_family:
                        if a_idx == remove_idx:
                            continue
                        for c_idx in forbidden[a_idx][remove_idx]:
                            conf_count[c_idx] -= 1
                    
                    # Find a valid add
                    added = False
                    for c_idx in range(m):
                        if c_idx not in in_family and conf_count[c_idx] == 0:
                            # Add it
                            current_family.append(c_idx)
                            in_family.add(c_idx)
                            for a_idx in current_family[:-1]:
                                if a_idx == c_idx: # shouldn't happen
                                    continue
                                for f_idx in forbidden[a_idx][c_idx]:
                                    conf_count[f_idx] += 1
                            added = True
                            improved = True
                            break
                    
                    if not added:
                        # Restore remove_idx
                        current_family.append(remove_idx)
                        in_family.add(remove_idx)
                        for a_idx in current_family[:-1]:
                            if a_idx == remove_idx:
                                continue
                            for f_idx in forbidden[a_idx][remove_idx]:
                                conf_count[f_idx] += 1

        return current_family

    # Apply local search to the best family found
    if time.time() < deadline - 10:
        improved_family = local_search(best_family_idx)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family_idx = improved_family[:]

    # More random restarts with local search
    for seed_val in [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579]:
        if time.time() > deadline - 5:
            break
        random.seed(seed_val)
        order = list(range(m))
        random.shuffle(order)
        family = greedy_construct(order)
        
        if time.time() < deadline - 2:
            family = local_search(family)
            
        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Cap at 250
    if len(best_family_idx) > 250:
        best_family_idx = best_family_idx[:250]

    # Convert back to sorted lists of 1..16
    result = [mask_to_list(all_masks[idx]) for idx in best_family_idx]

    return result
# EVOLVE-BLOCK-END
