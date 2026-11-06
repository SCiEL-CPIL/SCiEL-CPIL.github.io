import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 50.0  # safety margin before 60s limit

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)

    # Filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build conflict bitsets
    conflict_bits = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                conflict_bits[i] |= (1 << j)

    all_mask = (1 << num_valid) - 1

    def greedy_min_degree(remaining):
        """Greedy min-degree: repeatedly pick vertex with min degree in remaining."""
        selected = 0
        while remaining:
            best_v = -1
            best_deg = float('inf')
            v = remaining
            while v:
                lowbit = v & (-v)
                idx = lowbit.bit_length() - 1
                deg = (conflict_bits[idx] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = idx
                v ^= lowbit
            selected |= (1 << best_v)
            remaining &= ~(conflict_bits[best_v] | (1 << best_v))
        return selected

    def greedy_max_degree(remaining):
        """Greedy max-degree: repeatedly pick vertex with max degree in remaining."""
        selected = 0
        while remaining:
            best_v = -1
            best_deg = -1
            v = remaining
            while v:
                lowbit = v & (-v)
                idx = lowbit.bit_length() - 1
                deg = (conflict_bits[idx] & remaining).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = idx
                v ^= lowbit
            selected |= (1 << best_v)
            remaining &= ~(conflict_bits[best_v] | (1 << best_v))
        return selected

    def greedy_random(remaining, rng):
        """Greedy with random tie-breaking."""
        selected = 0
        while remaining:
            best_v = -1
            best_deg = float('inf')
            v = remaining
            while v:
                lowbit = v & (-v)
                idx = lowbit.bit_length() - 1
                deg = (conflict_bits[idx] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = idx
                elif deg == best_deg and rng.random() < 0.5:
                    best_v = idx
                v ^= lowbit
            selected |= (1 << best_v)
            remaining &= ~(conflict_bits[best_v] | (1 << best_v))
        return selected

    def branch_and_bound(remaining, current_size, best_size_ref, time_budget):
        """Branch-and-bound for MIS. Returns (best_bits, best_size).
        Uses degree-based upper bound: |IS| <= current_size + max_degree_bound(remaining).
        """
        # Upper bound: current_size + size of remaining (trivial)
        # Better bound: greedy coloring of remaining gives upper bound on IS size
        # We use: upper_bound = current_size + number of vertices in remaining
        # But that's too loose. Use a packing bound.
        
        # Simple bound: current_size + popcount(remaining)
        ub = current_size + remaining.bit_count()
        if ub <= best_size_ref[0]:
            return current_size, 0  # 0 bits means "no improvement"

        if not remaining:
            return current_size, 0

        # Pick a vertex to branch on
        # Use the vertex with highest degree in remaining (most constraining)
        best_v = -1
        best_deg = -1
        v = remaining
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            deg = (conflict_bits[idx] & remaining).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = idx
            v ^= lowbit

        # Branch 1: include best_v
        if time_budget[0] - time.time() < 0.1:
            return current_size, 0

        new_remaining_include = remaining & ~(conflict_bits[best_v] | (1 << best_v))
        size_inc, bits_inc = branch_and_bound(
            new_remaining_include, current_size + 1, best_size_ref, time_budget
        )
        if bits_inc:
            current_bits_inc = (1 << best_v) | bits_inc
        else:
            current_bits_inc = 1 << best_v
        current_size_inc = current_size + 1 + (bits_inc.bit_count() if bits_inc else 0)

        # Branch 2: exclude best_v
        if time_budget[0] - time.time() < 0.1:
            if current_size_inc > best_size_ref[0]:
                best_size_ref[0] = current_size_inc
                best_size_ref[1] = current_bits_inc
            return current_size_inc, current_bits_inc

        new_remaining_exclude = remaining & ~(1 << best_v)
        size_exc, bits_exc = branch_and_bound(
            new_remaining_exclude, current_size, best_size_ref, time_budget
        )
        if bits_exc:
            current_size_exc = current_size + bits_exc.bit_count()
        else:
            current_size_exc = current_size

        if current_size_inc >= current_size_exc:
            if current_size_inc > best_size_ref[0]:
                best_size_ref[0] = current_size_inc
                best_size_ref[1] = current_bits_inc
            return current_size_inc, current_bits_inc
        else:
            if current_size_exc > best_size_ref[0]:
                best_size_ref[0] = current_size_exc
                best_size_ref[1] = bits_exc
            return current_size_exc, bits_exc

    def improved_bnb(remaining, current_bits, current_size, best_size_ref, time_budget, depth_limit):
        """Improved branch-and-bound with depth limit and better pruning."""
        if depth_limit <= 0:
            # Greedy completion
            greedy_bits = greedy_min_degree(remaining)
            total_size = current_size + greedy_bits.bit_count()
            if total_size > best_size_ref[0]:
                best_size_ref[0] = total_size
                best_size_ref[1] = current_bits | greedy_bits
            return

        if not remaining:
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
                best_size_ref[1] = current_bits
            return

        # Upper bound using greedy coloring (simplified: just count)
        ub = current_size + remaining.bit_count()
        if ub <= best_size_ref[0]:
            return

        if time_budget[0] - time.time() < 0.05:
            return

        # Pick vertex: highest degree in remaining
        best_v = -1
        best_deg = -1
        v = remaining
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            deg = (conflict_bits[idx] & remaining).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = idx
            v ^= lowbit

        # Branch include
        new_rem_inc = remaining & ~(conflict_bits[best_v] | (1 << best_v))
        improved_bnb(new_rem_inc, current_bits | (1 << best_v), current_size + 1,
                     best_size_ref, time_budget, depth_limit - 1)

        # Branch exclude
        if time_budget[0] - time.time() < 0.05:
            return
        new_rem_exc = remaining & ~(1 << best_v)
        improved_bnb(new_rem_exc, current_bits, current_size,
                     best_size_ref, time_budget, depth_limit - 1)

    # Strategy: multiple restarts with different initializations, then B&B refinement
    rng = random.Random(42)
    
    # Phase 1: Get initial solutions from different greedy strategies
    initial_solutions = []
    initial_solutions.append(greedy_min_degree(all_mask))
    initial_solutions.append(greedy_max_degree(all_mask))
    for _ in range(5):
        initial_solutions.append(greedy_random(all_mask, rng))
    
    best_size = max(s.bit_count() for s in initial_solutions)
    best_bits = max(initial_solutions, key=lambda s: s.bit_count())

    # Phase 2: Local search improvement on best solution
    def local_search(bits, time_budget):
        """Iterated local search: try adding/removing/swapping to improve."""
        current = bits
        current_size = current.bit_count()
        best_local = bits
        best_local_size = current_size
        no_improve_count = 0
        
        while time_budget[0] - time.time() > 1.0:
            improved = False
            
            # Try to add any non-conflicting vertex
            candidates = ~current & all_mask
            c = candidates
            while c:
                lowbit = c & (-c)
                idx = lowbit.bit_length() - 1
                if not (conflict_bits[idx] & current):
                    new_bits = current | lowbit
                    new_size = new_size = new_bits.bit_count()
                    if new_size > best_local_size:
                        best_local_size = new_size
                        best_local = new_bits
                    current = new_bits
                    current_size = new_size
                    improved = True
                    no_improve_count = 0
                    break
                c ^= lowbit
            
            if improved:
                continue
            
            # Try swaps: remove one, add one
            if current_size > 0 and rng.random() < 0.7:
                # Pick a random vertex to remove
                count = current_size
                pick = rng.randint(0, count - 1)
                temp = current
                for _ in range(pick):
                    temp &= temp - 1
                lowbit_rm = temp & (-temp)
                idx_rm = lowbit_rm.bit_length() - 1
                
                base = current & ~lowbit_rm
                candidates_add = ~base & all_mask
                if candidates_add:
                    count_c = candidates_add.bit_count()
                    pick_c = rng.randint(0, count_c - 1)
                    temp_c = candidates_add
                    for _ in range(pick_c):
                        temp_c &= temp_c - 1
                    lowbit_add = temp_c & (-temp_c)
                    idx_add = lowbit_add.bit_length() - 1
                    
                    if not (conflict_bits[idx_add] & base):
                        new_bits = base | lowbit_add
                        new_size = new_bits.bit_count()
                        if new_size > best_local_size:
                            best_local_size = new_size
                            best_local = new_bits
                        current = new_bits
                        current_size = new_size
                        improved = True
                        no_improve_count = 0
                        continue
            
            # Try 2-for-1: remove two, add one (rare)
            if current_size >= 2 and rng.random() < 0.1:
                count = current_size
                pick1 = rng.randint(0, count - 1)
                temp1 = current
                for _ in range(pick1):
                    temp1 &= temp1 - 1
                lb1 = temp1 & (-temp1)
                idx1 = lb1.bit_length() - 1
                
                remaining_after_rm1 = current & ~lb1
                count2 = remaining_after_rm1.bit_count()
                if count2 > 0:
                    pick2 = rng.randint(0, count2 - 1)
                    temp2 = remaining_after_rm1
                    for _ in range(pick2):
                        temp2 &= temp2 - 1
                    lb2 = temp2 & (-temp2)
                    idx2 = lb2.bit_length() - 1
                    
                    base2 = current & ~lb1 & ~lb2
                    candidates_add2 = ~base2 & all_mask
                    if candidates_add2:
                        count_c2 = candidates_add2.bit_count()
                        pick_c2 = rng.randint(0, count_c2 - 1)
                        temp_c2 = candidates_add2
                        for _ in range(pick_c2):
                            temp_c2 &= temp_c2 - 1
                        lb_add = temp_c2 & (-temp_c2)
                        idx_add = lb_add.bit_length() - 1
                        
                        if not (conflict_bits[idx_add] & base2):
                            new_bits = base2 | lb_add
                            new_size = new_bits.bit_count()
                            if new_size > best_local_size:
                                best_local_size = new_size
                                best_local = new_bits
                            current = new_bits
                            current_size = new_size
                            improved = True
                            no_improve_count = 0
                            continue
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            
            # Occasionally kick: remove a random vertex to escape local optimum
            if no_improve_count % 10 == 0 and current_size > 1:
                count = current_size
                pick = rng.randint(0, count - 1)
                temp = current
                for _ in range(pick):
                    temp &= temp - 1
                lb = temp & (-temp)
                current = current & ~lb
                current_size = current.bit_count()
                no_improve_count = 0
        
        return best_local, best_local_size

    # Run local search on the best initial solution
    time_budget = [start_time + deadline]
    best_bits, best_size = local_search(best_bits, time_budget)

    # Phase 3: Try B&B on the complement of the best solution's "hard" part
    # Identify vertices not in best solution that are "close" to being addable
    if time.time() < start_time + deadline - 5:
        # Try B&B with depth limit from the current best
        best_size_ref = [best_size, best_bits]
        time_budget_bnb = [start_time + deadline - 1]
        
        # B&B on the full problem with depth limit
        depth = 30
        improved_bnb(all_mask, 0, 0, best_size_ref, time_budget_bnb, depth)
        
        if best_size_ref[0] > best_size:
            best_size = best_size_ref[0]
            best_bits = best_size_ref[1]

    # Phase 4: More restarts with different seeds if time permits
    if time.time() < start_time + deadline - 10:
        for restart in range(10):
            if time.time() > start_time + deadline - 3:
                break
            rng_restart = random.Random(42 + restart * 1000)
            init = greedy_random(all_mask, rng_restart)
            init_size = init.bit_count()
            
            if init_size < best_size - 5:
                continue
            
            time_budget_rs = [start_time + deadline - 2]
            improved_bits, improved_size = local_search(init, time_budget_rs)
            
            if improved_size > best_size:
                best_size = improved_size
                best_bits = improved_bits

    # Phase 5: Final greedy completion
    for i in range(num_valid):
        if not (best_bits >> i) & 1:
            if not (conflict_bits[i] & best_bits):
                best_bits |= (1 << i)
                best_size += 1

    # Convert to list of values
    result = []
    v = best_bits
    while v:
        lowbit = v & (-v)
        idx = lowbit.bit_length() - 1
        result.append(valid_numbers[idx])
        v ^= lowbit

    result.sort()

    # Verify
    result_set = set(result)
    valid = True
    for a in result:
        if 2 * a in squares:
            valid = False
            break
        for b in result:
            if a <= b and (a + b) in squares:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use greedy_min_degree result
        fb = greedy_min_degree(all_mask)
        result = []
        v = fb
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            result.append(valid_numbers[idx])
            v ^= lowbit
        result.sort()

    return result
