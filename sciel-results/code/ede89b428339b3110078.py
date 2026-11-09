"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 115  # safety margin before 120s evaluator limit
    N = 284
    PRESCRIBED = {1, 2, 5, 11}
    PRESCRIBED_LIST = sorted(PRESCRIBED)

    def compute_sums(S):
        s = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s.add(a + b)
        return s

    def can_add(S, sums, x):
        if x in S:
            return False
        for y in S:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def is_valid_sidon(S):
        """Verify S is a valid Sidon set."""
        if len(S) != len(set(S)):
            return False
        if not PRESCRIBED.issubset(set(S)):
            return False
        if any(x < 1 or x > N for x in S):
            return False
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def calc_collisions(S):
        """Count the number of duplicate-sum collisions. 0 means valid Sidon set."""
        sums_count = {}
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                sums_count[s] = sums_count.get(s, 0) + 1
        collisions = 0
        for c in sums_count.values():
            if c > 1:
                collisions += c * (c - 1) // 2
        return collisions

    def calc_collisions_incremental(S, removed, added):
        """
        Efficiently compute collision count change when removing 'removed' and adding 'added' to S.
        Returns (new_total_collisions, new_sums_count) or None if infeasible.
        """
        S_set = set(S)
        # Remove 'removed' from S
        S_new = [x for x in S if x != removed]
        S_new.append(added)
        
        # Check for duplicates
        if len(S_new) != len(set(S_new)):
            return None, None
        
        # Recompute sums (for small sets this is fast enough)
        sums_count = {}
        for i, a in enumerate(S_new):
            for b in S_new[i:]:
                s = a + b
                sums_count[s] = sums_count.get(s, 0) + 1
        
        collisions = 0
        for c in sums_count.values():
            if c > 1:
                collisions += c * (c - 1) // 2
        return collisions, sums_count

    def greedy_with_order(order, S_init):
        """Greedy construction using a specific candidate order."""
        S = list(S_init)
        S.sort()
        sums = compute_sums(S)
        S_set = set(S)
        for x in order:
            if x in S_set:
                continue
            if can_add(S, sums, x):
                S.append(x)
                S_set.add(x)
                for y in S[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
        S.sort()
        return S

    def local_search_improved(S, rng, max_rounds=50):
        """Aggressive local search with add and swap moves on valid sets."""
        S = list(S)
        S.sort()
        S_set = set(S)
        sums = compute_sums(S)
        best_S = list(S)
        best_size = len(S)
        no_improve_count = 0

        for round_num in range(max_rounds):
            if time.time() - start > DEADLINE:
                break

            improved = False

            # Try to add a new element
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            rng.shuffle(candidates)
            for x in candidates:
                if time.time() - start > DEADLINE:
                    break
                if can_add(S, sums, x):
                    S.append(x)
                    S.sort()
                    S_set.add(x)
                    for y in S[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break

            # Try swap: remove one non-prescribed element, add another
            if not improved:
                removable_indices = [i for i, x in enumerate(S) if x not in PRESCRIBED]
                rng.shuffle(removable_indices)
                for i in removable_indices:
                    if time.time() - start > DEADLINE:
                        break
                    removed = S[i]
                    S_without = S[:i] + S[i+1:]
                    S_without.sort()
                    S_without_set = set(S_without)
                    sums_without = compute_sums(S_without)
                    
                    candidates = [x for x in range(1, N + 1) if x not in S_without_set]
                    rng.shuffle(candidates)
                    found = False
                    for x in candidates:
                        if time.time() - start > DEADLINE:
                            break
                        if can_add(S_without, sums_without, x):
                            S = S_without + [x]
                            S.sort()
                            S_set = set(S)
                            sums = compute_sums(S)
                            improved = True
                            found = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = list(S)
                            break
                    if found:
                        break

            # Try 2-swap: remove two non-prescribed elements, add two others
            if not improved and len(S) >= 6:
                removable_indices = [i for i, x in enumerate(S) if x not in PRESCRIBED]
                if len(removable_indices) >= 2:
                    pairs = list(combinations(removable_indices, 2))
                    rng.shuffle(pairs)
                    for i, j in pairs[:20]:  # Limit to 20 pairs per round
                        if time.time() - start > DEADLINE:
                            break
                        removed1, removed2 = S[i], S[j]
                        S_without = [x for k, x in enumerate(S) if k != i and k != j]
                        S_without.sort()
                        S_without_set = set(S_without)
                        sums_without = compute_sums(S_without)
                        
                        candidates = [x for x in range(1, N + 1) if x not in S_without_set]
                        rng.shuffle(candidates)
                        
                        found_pair = False
                        for x in candidates[:50]:  # Limit candidate checks
                            if time.time() - start > DEADLINE:
                                break
                            if can_add(S_without, sums_without, x):
                                S_temp = S_without + [x]
                                S_temp.sort()
                                sums_temp = compute_sums(S_temp)
                                S_temp_set = set(S_temp)
                                
                                for y in candidates:
                                    if y == x or y in S_temp_set:
                                        continue
                                    if can_add(S_temp, sums_temp, y):
                                        S = S_temp + [y]
                                        S.sort()
                                        S_set = set(S)
                                        sums = compute_sums(S)
                                        improved = True
                                        found_pair = True
                                        if len(S) > best_size:
                                            best_size = len(S)
                                            best_S = list(S)
                                        break
                                if found_pair:
                                    break
                        if found_pair:
                            break

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= 5:
                    break

        return best_S

    def sa_collision_based(target_size, rng, max_steps=5000):
        """
        Simulated annealing using collision count as cost.
        Tries to find a valid Sidon set of exactly target_size containing PRESCRIBED.
        """
        num_vars = target_size - len(PRESCRIBED_LIST)
        if num_vars < 0:
            return None
        
        available = [x for x in range(1, N + 1) if x not in PRESCRIBED]
        
        # Initialize: prescribed + random samples
        S = list(PRESCRIBED_LIST) + rng.sample(available, num_vars)
        rng.shuffle(S)
        
        current_cost = calc_collisions(S)
        if current_cost == 0:
            return S
        
        T = 5.0
        T_min = 0.001
        cooling_rate = 0.9995
        
        best_S = list(S)
        best_cost = current_cost
        
        for step in range(max_steps):
            if time.time() - start > DEADLINE:
                break
            
            # Pick a random position (variable element, not prescribed)
            # Find indices of non-prescribed elements
            var_indices = [i for i, x in enumerate(S) if x not in PRESCRIBED]
            if not var_indices:
                break
            
            idx = rng.choice(var_indices)
            old_val = S[idx]
            
            # Choose a new value: random from available
            new_val = rng.choice(available)
            if new_val == old_val:
                continue
            # Ensure no duplicates
            if new_val in S:
                continue
            
            S[idx] = new_val
            new_cost = calc_collisions(S)
            
            delta_cost = new_cost - current_cost
            if delta_cost <= 0 or rng.random() < pow(2.718281828, -delta_cost / max(T, T_min)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_S = list(S)
                if new_cost == 0:
                    return best_S
            else:
                S[idx] = old_val
            
            T *= cooling_rate
        
        return best_S if best_cost == 0 else None

    def greedy_with_priority_order(order, S_init):
        """
        Greedy construction that prioritizes candidates with fewer conflicts.
        """
        S = list(S_init)
        S.sort()
        sums = compute_sums(S)
        S_set = set(S)
        
        # Score each candidate by how many existing elements it conflicts with
        scored_candidates = []
        for x in order:
            if x in S_set:
                continue
            conflict_count = 0
            for y in S:
                if (x + y) in sums:
                    conflict_count += 1
            if 2 * x in sums:
                conflict_count += 1
            scored_candidates.append((conflict_count, x))
        
        # Sort by conflict count (fewer conflicts first), then by value
        scored_candidates.sort()
        
        for _, x in scored_candidates:
            if can_add(S, sums, x):
                S.append(x)
                S_set.add(x)
                for y in S[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
        
        S.sort()
        return S

    def sa_multi_restart(target_size, num_restarts=20):
        """Multiple SA restarts targeting a specific size."""
        for r in range(num_restarts):
            if time.time() - start > DEADLINE:
                break
            rng = random.Random(10000 * r + target_size)
            result = sa_collision_based(target_size, rng, max_steps=3000)
            if result is not None:
                return result
        return None

    # Strategy: Multi-phase approach
    best_S = list(PRESCRIBED_LIST)
    best_size = len(PRESCRIBED_LIST)

    # Phase 1: Greedy constructions with different random orders
    for seed in range(100):
        if time.time() - start > DEADLINE:
            break
        rng = random.Random(seed)
        candidates = [x for x in range(1, N + 1) if x not in PRESCRIBED]
        rng.shuffle(candidates)
        S = greedy_with_order(candidates, PRESCRIBED_LIST)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Phase 1b: Priority-based greedy (fewer conflicts first)
    if time.time() - start < DEADLINE:
        candidates = [x for x in range(1, N + 1) if x not in PRESCRIBED]
        S = greedy_with_priority_order(candidates, PRESCRIBED_LIST)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # Phase 2: Local search on best greedy result
    if time.time() - start < DEADLINE:
        rng = random.Random(12345)
        S = local_search_improved(best_S, rng, max_rounds=50)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # Phase 3: Collision-based SA targeting sizes from 20 down to best_size+1
    for target_size in range(20, best_size, -1):
        if time.time() - start > DEADLINE:
            break
        result = sa_multi_restart(target_size, num_restarts=10)
        if result is not None:
            best_size = target_size
            best_S = result
            break  # Found a valid set of this size, done

    # Phase 4: If we found a set, try to improve with local search
    if best_size > 4 and time.time() - start < DEADLINE - 10:
        rng = random.Random(99999)
        S = local_search_improved(best_S, rng, max_rounds=80)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # Phase 5: Final polish - try more SA restarts at best_size+1 if time remains
    if best_size < 20 and time.time() - start < DEADLINE - 20:
        result = sa_multi_restart(best_size + 1, num_restarts=5)
        if result is not None:
            best_size = best_size + 1
            best_S = result

    # Verify result
    if not is_valid_sidon(best_S):
        # Fallback: try a simple greedy
        a = list(PRESCRIBED_LIST)
        sums = compute_sums(a)
        for x in range(12, N + 1):
            new = {x + y for y in a} | {2 * x}
            if len(new) == len(a) + 1 and not (new & sums):
                a.append(x)
                sums |= new
        if is_valid_sidon(a):
            best_S = a
        else:
            best_S = list(PRESCRIBED_LIST)

    return best_S
# EVOLVE-BLOCK-END
