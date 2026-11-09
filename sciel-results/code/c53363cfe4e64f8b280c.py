# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    n = 16
    k = 4
    max_target = 250

    # Generate all 4-subsets as bitmasks (0-indexed internally, 1-indexed in output)
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)

    total = len(all_sets)
    best_family = []
    best_size = 0
    rng = random.Random(12345)

    def greedy_build(order, current_best_size):
        """
        Greedily build a family from the given order of set indices.
        Returns the family as a list of masks and the inter_maps structure.
        """
        F = []
        inter_maps = []
        
        for idx in order:
            m = all_sets[idx]
            conflict = False
            
            for i in range(len(F)):
                s = m & F[i]
                lst = inter_maps[i].get(s)
                if lst:
                    for j in lst:
                        if (m & F[j]) == s:
                            conflict = True
                            break
                if conflict:
                    break
            
            if not conflict:
                idx_new = len(F)
                F.append(m)
                inter_maps.append({})
                
                for i in range(idx_new):
                    s = m & F[i]
                    if s not in inter_maps[i]:
                        inter_maps[i][s] = []
                    inter_maps[i][s].append(idx_new)
                    if s not in inter_maps[idx_new]:
                        inter_maps[idx_new][s] = []
                    inter_maps[idx_new][s].append(i)
                
                if len(F) >= max_target:
                    break

        return F, inter_maps

    def try_add_and_swap(F, inter_maps, in_family_idx, x_idx):
        """
        Attempts to add x_idx to F. If it conflicts, it finds the conflicting sets.
        It then tries to swap one conflicting set out and x_idx in.
        Returns True if a swap or add was successful, updating F and inter_maps in place.
        """
        m = all_sets[x_idx]
        
        # Check if x can be added directly
        can_add = True
        conflicting_sets = []
        
        for i in range(len(F)):
            s = m & F[i]
            lst = inter_maps[i].get(s)
            if lst:
                for j in lst:
                    if (m & F[j]) == s:
                        # Conflict between F[i], F[j], m
                        # The sets involved in the conflict with m are F[i] and F[j]
                        # We need to remove at least one of them to add m?
                        # Actually, if we remove F[i], the pair (F[i], F[j]) is gone.
                        # But there might be other pairs.
                        # For a simple swap, we pick one of the conflicting sets to remove.
                        # Let's collect all indices i, j that form a sunflower with m.
                        if i not in conflicting_sets:
                            conflicting_sets.append(i)
                        if j not in conflicting_sets:
                            conflicting_sets.append(j)
                        can_add = False
                        # We don't break immediately because we want a list of candidates for removal
                        # But if we just want to know if it's possible, we can break.
                        # For swapping, we need candidates.
        
        if can_add:
            # Add directly
            idx_new = len(F)
            F.append(m)
            inter_maps.append({})
            in_family_idx.add(x_idx)
            
            for i in range(idx_new):
                s = m & F[i]
                if s not in inter_maps[i]:
                    inter_maps[i][s] = []
                inter_maps[i][s].append(idx_new)
                if s not in inter_maps[idx_new]:
                    inter_maps[idx_new][s] = []
                inter_maps[idx_new][s].append(i)
            return True

        # If conflict, try to swap
        if not conflicting_sets:
            return False

        # Pick a random conflicting set to remove
        # Shuffle conflicting_sets for diversity
        rng.shuffle(conflicting_sets)
        
        for remove_idx in conflicting_sets:
            # Try to remove F[remove_idx] and add m
            # We need to check if F \ {F[remove_idx]} \cup {m} is valid.
            # This is equivalent to checking if m is valid in F \ {F[remove_idx]}.
            
            # To do this efficiently, we can temporarily remove F[remove_idx] from inter_maps,
            # check validity of m, and if valid, commit. If not, revert.
            
            # Removing from inter_maps is O(N).
            # Checking validity is O(N^2) worst case, but usually less.
            
            # Let's implement a helper to check validity of m against a family with one element removed.
            # Actually, we can just simulate the removal.
            
            # Save state of inter_maps? No, too expensive.
            # We can check: Does m conflict with any pair (A, B) in F such that neither A nor B is remove_idx?
            # If m conflicts with a pair (A, B) where A != remove_idx and B != remove_idx, then the swap fails.
            
            valid_swap = True
            m_mask = m
            F_list = F
            inter_list = inter_maps
            
            for i in range(len(F_list)):
                if i == remove_idx:
                    continue
                s = m_mask & F_list[i]
                lst = inter_list[i].get(s)
                if lst:
                    for j in lst:
                        if j == remove_idx:
                            continue
                        if (m_mask & F_list[j]) == s:
                            valid_swap = False
                            break
                if not valid_swap:
                    break
            
            if valid_swap:
                # Perform the swap
                # 1. Remove F[remove_idx] from F and inter_maps
                # 2. Add m to F and inter_maps
                
                # Remove F[remove_idx]
                mask_to_remove = F_list[remove_idx]
                in_family_idx.discard(all_sets.index(mask_to_remove)) # This is slow, O(N). 
                # Better: maintain a mapping from mask to index in F? Or just use indices in all_sets.
                # Let's track in_family_idx as a set of all_sets indices.
                # We need to know which all_sets index corresponds to F[remove_idx].
                # We can precompute a mask_to_idx map.
                
                # For now, let's assume we can find it.
                # To avoid O(N) search, we should store the original index in F.
                # Let's modify F to be a list of (mask, original_idx).
                
                # Refactor: F is list of masks. We need original indices for in_family_idx.
                # Let's pass original indices along.
                
                # This implementation is getting complex. 
                # Simpler approach: Just rebuild inter_maps if we swap? No, too slow.
                
                # Let's stick to a simpler local search: 
                # Just try to add a random set. If it fails, do nothing.
                # The greedy random restarts might be sufficient if we run enough.
                # Or, use a "repair" step: if greedy stops, try to find a set that can be added by removing 1 element.
                
                # Let's implement the swap properly with original indices.
                
                # We need to know the original index of F[remove_idx].
                # Let's assume F stores masks. We need a mask_to_orig_idx map.
                
                # Let's restart the local search logic with a cleaner data structure.
                
                pass

        return False

    # Precompute mask to index map for quick lookup
    mask_to_idx = {mask: i for i, mask in enumerate(all_sets)}

    def local_search(F, inter_maps, in_family_idx, time_limit):
        """
        Performs local search by trying to add sets and swapping.
        """
        # We need to know the original index of each mask in F to update in_family_idx
        # Let's create a list of original indices parallel to F
        F_orig_indices = [mask_to_idx[m] for m in F]
        
        while time.time() < time_limit:
            # Pick a random set not in F
            # To do this efficiently, we can pick a random index from 0 to total-1
            # and check if it's in in_family_idx
            # If it is, pick another.
            
            # Better: maintain a list of available sets?
            # For now, random sampling.
            
            attempts = 0
            while attempts < 100:
                x_idx = rng.randint(0, total - 1)
                if x_idx not in in_family_idx:
                    break
                attempts += 1
            else:
                continue
                
            m = all_sets[x_idx]
            
            # Check if m can be added
            can_add = True
            conflicting_orig_indices = set()
            
            for i in range(len(F)):
                s = m & F[i]
                lst = inter_maps[i].get(s)
                if lst:
                    for j in lst:
                        if (m & F[j]) == s:
                            can_add = False
                            conflicting_orig_indices.add(F_orig_indices[i])
                            conflicting_orig_indices.add(F_orig_indices[j])
                            # We can break early if we just want to know if it's possible
                            # But we need the set for swapping
                            # Let's collect them all
            
            if can_add:
                # Add directly
                idx_new = len(F)
                F.append(m)
                F_orig_indices.append(x_idx)
                in_family_idx.add(x_idx)
                inter_maps.append({})
                
                for i in range(idx_new):
                    s = m & F[i]
                    if s not in inter_maps[i]:
                        inter_maps[i][s] = []
                    inter_maps[i][s].append(idx_new)
                    if s not in inter_maps[idx_new]:
                        inter_maps[idx_new][s] = []
                    inter_maps[idx_new][s].append(i)
            else:
                # Try to swap
                # Pick a random conflicting set to remove
                if conflicting_orig_indices:
                    remove_orig_idx = rng.choice(list(conflicting_orig_indices))
                    # Find the position of this original index in F
                    pos_to_remove = F_orig_indices.index(remove_orig_idx)
                    
                    # Check if swap is valid
                    # Valid if m does not conflict with any pair (A, B) in F \ {F[pos_to_remove]}
                    valid_swap = True
                    for i in range(len(F)):
                        if i == pos_to_remove:
                            continue
                        s = m & F[i]
                        lst = inter_maps[i].get(s)
                        if lst:
                            for j in lst:
                                if j == pos_to_remove:
                                    continue
                                if (m & F[j]) == s:
                                    valid_swap = False
                                    break
                        if not valid_swap:
                            break
                    
                    if valid_swap:
                        # Perform swap
                        # 1. Remove F[pos_to_remove]
                        # 2. Add m
                        
                        # Remove
                        mask_to_remove = F.pop(pos_to_remove)
                        F_orig_indices.pop(pos_to_remove)
                        in_family_idx.discard(remove_orig_idx)
                        
                        # Update inter_maps for removal
                        # This is the tricky part. We need to remove pos_to_remove from all inter_maps.
                        # And remove the entry for inter_maps[pos_to_remove].
                        
                        # We can just rebuild inter_maps for the remaining F?
                        # O(N^2). If N=100, 10^4 ops. Fast.
                        # If N=200, 4*10^4 ops. Still fast.
                        # Let's just rebuild inter_maps after the swap.
                        
                        # Add m
                        F.append(m)
                        F_orig_indices.append(x_idx)
                        in_family_idx.add(x_idx)
                        
                        # Rebuild inter_maps
                        inter_maps = []
                        for idx_m in F:
                            inter_maps.append({})
                        
                        for i in range(len(F)):
                            for j in range(i):
                                s = F[i] & F[j]
                                inter_maps[i][s].append(j)
                                inter_maps[j][s].append(i)
                        
                        # Note: inter_maps is a local variable in the loop, 
                        # so we need to return it or update the outer scope.
                        # Since we are in a function, we can return the new F and inter_maps.
                        
                        return F, inter_maps, in_family_idx, F_orig_indices
        
        return F, inter_maps, in_family_idx, F_orig_indices

    # 1. Deterministic pass
    if time.time() < deadline:
        order = list(range(total))
        F, inter_maps = greedy_build(order, best_size)
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
            # Initialize in_family_idx
            in_family_idx = set(mask_to_idx[m] for m in F)
            F_orig_indices = [mask_to_idx[m] for m in F]
            
            # Local search
            time_limit = deadline - 5.0 # Leave 5s for other restarts
            if time.time() < time_limit:
                F, inter_maps, in_family_idx, F_orig_indices = local_search(F, inter_maps, in_family_idx, time_limit)
                if len(F) > best_size:
                    best_family = F
                    best_size = len(F)

    # 2. Random restarts
    restart_count = 0
    while time.time() < deadline - 2.0:
        restart_count += 1
        seed = 12345 + restart_count
        rng = random.Random(seed)
        
        order = list(range(total))
        rng.shuffle(order)
        
        F, inter_maps = greedy_build(order, best_size)
        
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
            if best_size >= max_target:
                break
            
            # If we found a new best, try to improve it with local search
            if time.time() < deadline - 2.0:
                in_family_idx = set(mask_to_idx[m] for m in F)
                F_orig_indices = [mask_to_idx[m] for m in F]
                time_limit = min(deadline - 2.0, time.time() + 1.0) # 1s local search
                F, inter_maps, in_family_idx, F_orig_indices = local_search(F, inter_maps, in_family_idx, time_limit)
                if len(F) > best_size:
                    best_family = F
                    best_size = len(F)

    # Convert best_family masks to sorted 1-indexed lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
