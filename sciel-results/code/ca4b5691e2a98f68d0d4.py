import time
import random
import numpy as np

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 28.0  # Safety margin for 30s limit
    
    best_size = N + 1
    best_set = []
    
    # Precompute bit masks or just use sets/lists? 
    # For N=256, sums are in [2, 512]. 
    # We can use a boolean array of size 513 for sums.
    
    def get_maximal_set(initial_S, rng):
        """
        Greedily extend initial_S to a maximal Sidon set.
        Returns the maximal set as a list.
        """
        S = np.array(sorted(initial_S), dtype=np.int64)
        if len(S) == 0:
            S = np.array([], dtype=np.int64)
            
        # sums_mask: boolean array where index s is True if s is a sum of two elements in S
        # Max sum is 2*N = 512.
        sums_mask = np.zeros(2 * N + 1, dtype=bool)
        
        def update_sums(S):
            nonlocal sums_mask
            sums_mask[:] = False
            if len(S) == 0:
                return
            # Generate all a+b for a,b in S
            # S is 1D array. S[:, None] + S[None, :] gives 2D matrix of sums.
            # We need to mark these in sums_mask.
            # Note: The Sidon condition requires all sums a+b (a<=b) to be unique.
            # The verifier checks a+b for a<=b.
            # If we just mark all a+b, we might mark duplicates if S is not Sidon, 
            # but we assume S is always Sidon during construction.
            # Actually, we just need to know which sums are "used".
            # If a sum is used, it's "blocked" for future additions?
            # No, an element x is blocked if:
            # 1. 2x is a sum in S (diagonal collision)
            # 2. There exists y in S such that x+y is a sum in S (off-diagonal collision)
            #    i.e., x+y = a+b for some a,b in S.
            #    This is equivalent to x = a+b-y.
            #    Or x+y = s where s is in sums_mask.
            
            if len(S) > 0:
                pair_sums = S[:, None] + S[None, :]
                sums_mask[pair_sums.ravel()] = True
        
        update_sums(S)
        
        # Available elements: those in [1, N] not in S
        # We maintain a set of available elements.
        all_elements = set(range(1, N + 1))
        S_set = set(S.tolist())
        available = all_elements - S_set
        
        while available:
            # Find the best candidate to add
            # Heuristic: Maximize the number of available elements that become blocked.
            best_z = -1
            max_blocked = -1
            best_score = -1.0
            
            # To speed up, we can sample if available is large, but N=256 is small.
            # Let's iterate all available.
            avail_list = list(available)
            rng.shuffle(avail_list)
            
            for z in avail_list:
                # Check if z is valid to add
                # 1. 2z not in sums_mask
                if sums_mask[2 * z]:
                    continue
                # 2. No collision: (sums_mask >> z) & S_mask == 0
                # This means: for no a in S is (z+a) in sums_mask?
                # Wait, the condition for adding z is:
                # All new sums z+z, z+a (a in S) must be new (not in current sums_mask) 
                # AND distinct from each other.
                # Since S is Sidon, z+a are distinct for distinct a.
                # z+z is distinct from z+a (since a != z? No, a can be z? No z not in S).
                # So we just need:
                # - 2z not in sums_mask
                # - For all a in S, z+a not in sums_mask
                
                valid = True
                for a in S:
                    if sums_mask[z + a]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # If valid, calculate how many available elements are blocked
                # An element w is blocked if adding z to S makes w invalid.
                # w is invalid if:
                # 1. 2w in new_sums_mask
                # 2. Exists a in new_S such that w+a in new_sums_mask
                
                # new_S = S U {z}
                # new_sums_mask includes old sums + 2z + {z+a for a in S}
                
                # We can approximate the blocked count.
                # Or just compute it exactly.
                blocked_count = 0
                for w in available:
                    if w == z:
                        continue
                    # Check if w is blocked by adding z
                    # 1. 2w in new_sums?
                    # 2w is in new_sums if 2w was in old_sums OR 2w == 2z (w=z, skipped) OR 2w == z+a.
                    # 2w == z+a => 2w - z = a. So if 2w-z is in S, then 2w is in new_sums.
                    # Also if 2w is in old_sums, it was already blocked? 
                    # No, w is in `available`, so w was NOT blocked before.
                    # So 2w was NOT in old_sums.
                    # So 2w is in new_sums iff 2w - z is in S.
                    
                    # 2. w+a in new_sums?
                    # w+a is in new_sums if w+a in old_sums OR w+a == 2z OR w+a == z+b.
                    # w+a in old_sums: If w was available, w+a was NOT in old_sums?
                    # Wait. If w was available, it means w was NOT blocked.
                    # w not blocked means: 2w not in old_sums AND for all a in S, w+a not in old_sums.
                    # So w+a is NOT in old_sums.
                    
                    # So w+a is in new_sums iff w+a == 2z OR w+a == z+b for some b in S.
                    # w+a == 2z => w = 2z - a.
                    # w+a == z+b => w = z + b - a.
                    
                    # So w is blocked if:
                    # (2w - z) in S_set 
                    # OR (w == 2z - a for some a in S)  => 2z - w in S_set
                    # OR (w == z + b - a for some b,a in S) => w - z in (S - S)
                    
                    # Let's use sets for S for fast lookup
                    # S is small, so S_set is fine.
                    
                    if (2 * w - z) in S_set:
                        blocked_count += 1
                        continue
                    if (2 * z - w) in S_set:
                        blocked_count += 1
                        continue
                    # Check w - z in S - S
                    # S - S is {b - a | a, b in S}
                    # We can precompute S_diffs? Or just loop.
                    # Since S is small, loop is fine.
                    blocked = False
                    for b in S:
                        for a in S:
                            if w - z == b - a:
                                blocked = True
                                break
                        if blocked:
                            break
                    if blocked:
                        blocked_count += 1
                    else:
                        # Is it possible w is blocked by other means?
                        # The condition for w to be valid is that all sums involving w are new.
                        # We established that old sums don't collide.
                        # So only new sums matter.
                        pass

                score = blocked_count + rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_z = z
            
            if best_z == -1:
                # No valid element found? This implies maximal.
                break
            
            # Commit best_z
            S = np.append(S, best_z)
            S = np.sort(S)
            S_set.add(best_z)
            available.remove(best_z)
            
            update_sums(S)
            
            # Update available: remove any w that is now blocked
            new_available = set()
            for w in available:
                # Check if w is still valid
                if sums_mask[2 * w]:
                    continue
                valid = True
                for a in S:
                    if sums_mask[w + a]:
                        valid = False
                        break
                if valid:
                    new_available.add(w)
            available = new_available
        
        return S.tolist()

    # Local search helper
    def local_search(S_list, rng):
        S_list = S_list[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each element
            for i in range(len(S_list)):
                if time.time() > deadline:
                    return S_list
                
                # Remove S_list[i]
                candidate = S_list[:i] + S_list[i+1:]
                # Extend
                new_S = get_maximal_set(candidate, rng)
                if len(new_S) < len(S_list):
                    S_list = new_S
                    improved = True
                    break
        return S_list

    # Main Loop
    for iter in range(100): # Increased iterations due to speedup
        if time.time() > deadline:
            break
            
        rng = random.Random(iter)
        
        # Start with empty set
        S_list = get_maximal_set([], rng)
        
        # Local search
        S_list = local_search(S_list, rng)
        
        if len(S_list) < best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            # print(f"Iter {iter}: Found size {best_size}")

    if not best_set:
        # Fallback: just return a valid maximal set, even if not minimal
        best_set = get_maximal_set([], random.Random(0))
        
    return best_set
