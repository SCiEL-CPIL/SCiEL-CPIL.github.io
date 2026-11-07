import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 85  # 5s safety margin

    # Precompute subset/superset relationships for faster lookup
    # For each mask, precompute its submasks count and supmasks count
    # Actually, let's use a different approach: precompute for each mask, 
    # the list of masks that are subsets of it (or supersets)
    
    # Better: use bit manipulation for fast comparability check
    def is_comparable_strict(a, b):
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0

    def calc_score_fast(family_set):
        """Calculate number of comparable pairs in family."""
        fam = list(family_set)
        score = 0
        for i in range(len(fam)):
            for j in range(i + 1, len(fam)):
                a, b = fam[i], fam[j]
                if (a & ~b) == 0 or (b & ~a) == 0:
                    score += 1
        return score

    def get_comparables_in_family(mask, family_set):
        """Get count of masks in family_set that are comparable with mask (strict)."""
        count = 0
        for other in family_set:
            if other == mask:
                continue
            if (mask & ~other) == 0 or (other & ~mask) == 0:
                count += 1
        return count

    def hill_climb(start_family, max_iters=5000, deadline_check=None):
        """Local search: try swaps that improve the score."""
        family_set = set(start_family)
        current_score = calc_score_fast(family_set)
        
        # Precompute comparable counts for each member
        comp_counts = {}
        for a in family_set:
            comp_counts[a] = get_comparables_in_family(a, family_set)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            if deadline_check and time.time() > deadline_check:
                break
            improved = False
            iterations += 1
            
            best_delta = 0
            best_a = None
            best_b = None
            
            # Try all swaps: remove a, add b
            for a in family_set:
                comp_a = comp_counts[a]
                for b in range(total_masks):
                    if b in family_set:
                        continue
                    # If we add b, how many new comparables does it have with F\{a}?
                    # It's comparable with all in F\{a} that are comparable with b
                    # minus 1 if a was comparable with b
                    is_comp_ab = 1 if (a != b and ((a & ~b) == 0 or (b & ~a) == 0)) else 0
                    # Count comparables of b with F\{a}
                    comp_b_without_a = 0
                    for c in family_set:
                        if c == a:
                            continue
                        if (b & ~c) == 0 or (c & ~b) == 0:
                            comp_b_without_a += 1
                    
                    delta = comp_b_without_a - comp_a
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_delta > 0 and best_a is not None:
                # Apply swap
                family_set.remove(best_a)
                family_set.add(best_b)
                current_score += best_delta
                
                # Update comp_counts
                # Remove best_a's count
                del comp_counts[best_a]
                # Add best_b's count: comparables with new family
                comp_counts[best_b] = get_comparables_in_family(best_b, family_set)
                # Update counts for all others that were comparable with best_a or best_b
                for c in family_set:
                    if c == best_b:
                        continue
                    was_comp_with_a = (c != best_a and ((c & ~best_a) == 0 or (best_a & ~c) == 0))
                    is_comp_with_b = (c != best_b and ((c & ~best_b) == 0 or (best_b & ~c) == 0))
                    if was_comp_with_a or is_comp_with_b:
                        comp_counts[c] = get_comparables_in_family(c, family_set)
                
                improved = True
                
        return list(family_set), current_score

    # Generate diverse initial families
    def generate_initial_families():
        families = []
        
        # Strategy 1: All subsets of {0,...,5} (64 subsets), pick 40
        small_universe = [i for i in range(1 << 6)]
        random.shuffle(small_universe)
        families.append(small_universe[:m])
        
        # Strategy 2: Chain-based - create several chains and merge
        # A chain in B12 has at most 13 elements
        # Create 4 chains of ~10 elements each
        chains = []
        used = set()
        for _ in range(4):
            chain = [0]
            current = 0
            while len(chain) < 10 and len(used) < total_masks:
                # Add a random element not in current
                candidates = [i for i in range(n) if not (current >> i) & 1 and i not in [j for j in range(n) if (current >> j) & 1]]
                if not candidates:
                    break
                # Pick a random unused element
                available = [i for i in range(n) if not (current >> i) & 1]
                if not available:
                    break
                idx = random.choice(available)
                new_mask = current | (1 << idx)
                if new_mask not in used:
                    chain.append(new_mask)
                    used.add(new_mask)
                    current = new_mask
            chains.append(chain)
        
        # Flatten and pick 40
        all_chain_elements = []
        for chain in chains:
            all_chain_elements.extend(chain)
        random.shuffle(all_chain_elements)
        if len(all_chain_elements) >= m:
            families.append(all_chain_elements[:m])
        
        # Strategy 3: Random
        for _ in range(3):
            random_family = random.sample(range(total_masks), m)
            families.append(random_family)
        
        # Strategy 4: Subsets of {0,...,7} (256 subsets)
        medium_universe = [i for i in range(1 << 8)]
        random.shuffle(medium_universe)
        families.append(medium_universe[:m])
        
        return families

    best_family = None
    best_score = -1

    initial_families = generate_initial_families()
    
    for start in initial_families:
        if time.time() > deadline - 5:
            break
        fam, score = hill_climb(start, max_iters=3000, deadline_check=deadline - 2)
        if score > best_score:
            best_score = score
            best_family = fam

    # If we still have time, do more restarts
    if time.time() < deadline - 10 and best_family is not None:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb the best family
            perturbed = list(best_family)
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, len(perturbed) - 1)
                new_mask = random.randint(0, total_masks - 1)
                while new_mask in perturbed:
                    new_mask = random.randint(0, total_masks - 1)
                perturbed[idx] = new_mask
            
            fam, score = hill_climb(perturbed, max_iters=2000, deadline_check=deadline - 2)
            if score > best_score:
                best_score = score
                best_family = fam

    if best_family is None:
        best_family = list(range(m))

    # Convert to lists of vertices
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
