import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1, 16]
ALL_SETS = [frozenset(c) for c in combinations(range(1, 17), 4)]
N_SETS = len(ALL_SETS)

# Convert to bitmasks for fast intersection
ALL_MASKS = [0] * N_SETS
for i, s in enumerate(ALL_SETS):
    m = 0
    for bit in s:
        m |= (1 << (bit - 1))
    ALL_MASKS[i] = m

# Map frozenset to index for fast lookup
SET_TO_IDX = {s: i for i, s in enumerate(ALL_SETS)}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_family = []
    best_size = 0
    
    # Helper function to check if a set can be added to a family
    # family is a list of indices
    # Returns True if valid, False otherwise
    def can_add(idx_new, family):
        m_new = ALL_MASKS[idx_new]
        
        # Bucket family members by their intersection with m_new
        buckets = {}
        for idx_a in family:
            m_a = ALL_MASKS[idx_a]
            inter = m_new & m_a
            if inter not in buckets:
                buckets[inter] = []
            buckets[inter].append(idx_a)
        
        # For each bucket, check if any pair A, B has A & B == inter
        for inter, members in buckets.items():
            m_len = len(members)
            if m_len < 2:
                continue
            
            # Check pairs in this bucket
            # We need to verify if A & B == inter
            for i in range(m_len):
                a_idx = members[i]
                m_a = ALL_MASKS[a_idx]
                for j in range(i + 1, m_len):
                    b_idx = members[j]
                    m_b = ALL_MASKS[b_idx]
                    if (m_a & m_b) == inter:
                        return False
        
        return True

    # Helper to check if a list of indices forms a valid family
    # This is O(N^3) effectively if done naively, but we use the incremental logic
    # For local search swaps, we can just re-verify the whole family or use incremental logic.
    # Given N~200, full verification is 200^3/6 ~ 1.3M ops. Might be slow for many iterations.
    # Better: Use the 'can_add' logic sequentially? No, order matters for 'can_add' if not careful?
    # Actually, 'can_add' is symmetric. If we build the family from scratch, it's fine.
    # For a swap, we remove A and add B.
    # We can check validity of (F \ {A}) + {B} by checking B against (F \ {A}).
    # This is O(|F|^2) per check.
    
    def is_valid_family(family):
        # Check all triples
        # Optimization: Use the bucket method for each set against the rest?
        # Or just triple loop.
        # With N=200, N^3 = 8M. In Python, this is ~1-2 seconds. Too slow for tight loop.
        # We need a faster validity check.
        
        # Alternative: For each pair (i, j), compute I_ij.
        # For each k, check if (I_ik == I_ij == I_jk).
        # This is still O(N^3).
        
        # Let's stick to the bucketed check for adding one item.
        # For a full family check, we can sort and add one by one?
        # If we add in a fixed order, and 'can_add' is correct, then the result is valid.
        # But 'can_add' relies on the existing family being valid.
        # So, to check a candidate family F:
        # 1. Sort F (or keep in original order).
        # 2. Start with empty.
        # 3. Try to add each element of F to the current valid set.
        # 4. If all added, F is valid.
        
        current = []
        for idx in family:
            if not can_add(idx, current):
                return False
            current.append(idx)
        return True

    # Greedy Pass
    def greedy_pass(order):
        family = []
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            if can_add(idx, family):
                family.append(idx)
        return family

    # Local Search
    def local_search(family):
        # family is a list of indices
        # We try to improve it
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Build a set of indices in family for O(1) lookup
            in_family = set(family)
            not_in_family = [i for i in range(N_SETS) if i not in in_family]
            
            # 1. Try to add a set
            random.shuffle(not_in_family)
            for idx_new in not_in_family:
                if time.time() > deadline - 5.0:
                    return family
                if can_add(idx_new, family):
                    family.append(idx_new)
                    in_family.add(idx_new)
                    improved = True
                    # Update not_in_family? Or just recompute in next loop.
                    # Recomputing is safer and cheap enough for N=1820.
                    break
            
            if improved:
                continue
            
            # 2. Try Swap (1 for 1)
            # Pick a random set to remove
            if family:
                idx_remove = random.choice(family)
                # Try to find a replacement
                random.shuffle(not_in_family)
                for idx_add in not_in_family:
                    if time.time() > deadline - 5.0:
                        return family
                    if idx_add == idx_remove:
                        continue
                    
                    # Check if (family - {idx_remove}) + {idx_add} is valid
                    # This is equivalent to checking if idx_add can be added to (family - {idx_remove})
                    temp_family = [x for x in family if x != idx_remove]
                    if can_add(idx_add, temp_family):
                        # Perform swap
                        family = temp_family + [idx_add]
                        improved = True
                        break
            
            if improved:
                continue

            # 3. Try 1-for-2 (Remove 1, Add 2)
            # This can increase size
            if family:
                idx_remove = random.choice(family)
                temp_family = [x for x in family if x != idx_remove]
                
                # Pick two random candidates
                random.shuffle(not_in_family)
                # We need to find a pair (a, b) such that adding both to temp_family is valid.
                # This is harder. We can just try pairs.
                # To save time, limit the number of pairs tried.
                found_pair = False
                for i in range(min(20, len(not_in_family))):
                    if time.time() > deadline - 5.0:
                        return family
                    idx_a = not_in_family[i]
                    if idx_a == idx_remove:
                        continue
                    # Check if idx_a can be added to temp_family
                    if not can_add(idx_a, temp_family):
                        continue
                    
                    # Now check if idx_b can be added to temp_family + [idx_a]
                    # We only need to check against the new set and existing ones
                    # can_add(idx_b, temp_family + [idx_a])
                    for j in range(i + 1, min(20, len(not_in_family))):
                        idx_b = not_in_family[j]
                        if idx_b == idx_remove:
                            continue
                        if can_add(idx_b, temp_family + [idx_a]):
                            # Valid!
                            family = temp_family + [idx_a, idx_b]
                            found_pair = True
                            improved = True
                            break
                    if found_pair:
                        break
            
            if improved:
                continue

        return family

    # Main Loop
    random.seed(42)
    
    # Phase 1: Multiple Greedy Restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        order = list(range(N_SETS))
        random.shuffle(order)
        
        family = greedy_pass(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
    # Phase 2: Local Search on Best Family
    if best_size > 0 and time.time() < deadline - 5.0:
        current_family = best_family[:]
        
        # Run local search multiple times or until no improvement
        # Since local search can get stuck, we can do a few iterations
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            
            improved_family = local_search(current_family)
            
            if len(improved_family) > len(current_family):
                current_family = improved_family
                if len(current_family) > best_size:
                    best_size = len(current_family)
                    best_family = current_family[:]
            else:
                # If no improvement, maybe perturb and try again?
                # For now, just break or do a small random restart from this family
                if time.time() > deadline - 2.0:
                    break
                # Perturb: remove 2 random sets
                if len(current_family) > 2:
                    to_remove = random.sample(current_family, 2)
                    current_family = [x for x in current_family if x not in to_remove]
                else:
                    break

    # Convert back to sorted lists
    return [sorted(list(ALL_SETS[idx])) for idx in best_family]
