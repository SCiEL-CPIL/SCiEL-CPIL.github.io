import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit

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

    # Greedy min-degree solution as initial solution
    def greedy_min_degree():
        remaining = all_mask
        selected = 0  # bitset of selected vertices
        while remaining:
            # Find vertex with minimum degree in remaining
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

    # Check if a bitset is a valid independent set
    def is_valid_is(bits):
        v = bits
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            if conflict_bits[idx] & bits:
                return False
            v ^= lowbit
        return True

    # Repair: remove conflicting vertices to make it a valid IS
    def repair(bits):
        changed = True
        while changed:
            changed = False
            v = bits
            while v:
                lowbit = v & (-v)
                idx = lowbit.bit_length() - 1
                if conflict_bits[idx] & bits:
                    bits &= ~(1 << idx)
                    changed = True
                v ^= lowbit
        return bits

    # Initial solution
    best_bits = greedy_min_degree()
    best_size = best_bits.bit_count()

    # Simulated Annealing
    rng = random.Random(42)
    T = 1.0
    T_min = 0.001
    cooling = 0.9995  # very slow cooling for many iterations

    current_bits = best_bits
    current_size = current_bits.bit_count()

    while time.time() < deadline:
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Add a vertex
            # Find vertices not in current_bits that don't conflict
            candidates = ~current_bits & all_mask
            if candidates:
                v = candidates
                # Pick a random candidate
                count = candidates.bit_count()
                pick = rng.randint(0, count - 1)
                # Find the pick-th set bit
                temp = candidates
                for _ in range(pick):
                    temp &= temp - 1
                lowbit = temp & (-temp)
                idx = lowbit.bit_length() - 1
                new_bits = current_bits | (1 << idx)
                new_size = new_size = new_bits.bit_count()
                # Check validity
                if not (conflict_bits[idx] & current_bits):
                    # Valid addition
                    if new_size > best_size:
                        best_size = new_size
                        best_bits = new_bits
                    current_bits = new_bits
                    current_size = new_size
                else:
                    # Invalid - try removing a conflicting vertex (swap-like)
                    conflicts = conflict_bits[idx] & current_bits
                    if conflicts:
                        # Remove one conflicting vertex, add this one
                        c = conflicts
                        count_c = c.bit_count()
                        pick_c = rng.randint(0, count_c - 1)
                        temp = c
                        for _ in range(pick_c):
                            temp &= temp - 1
                        lowbit_c = temp & (-temp)
                        idx_c = lowbit_c.bit_length() - 1
                        new_bits = (current_bits & ~(1 << idx_c)) | (1 << idx)
                        new_size = new_bits.bit_count()
                        delta = new_size - current_size
                        if delta >= 0 or rng.random() < math.exp(delta / max(T, T_min)):
                            if new_size > best_size:
                                best_size = new_size
                                best_bits = new_bits
                            current_bits = new_bits
                            current_size = new_size
        elif move_type < 0.6:
            # Remove a vertex
            if current_bits:
                v = current_bits
                count = current_bits.bit_count()
                pick = rng.randint(0, count - 1)
                temp = v
                for _ in range(pick):
                    temp &= temp - 1
                lowbit = temp & (-temp)
                idx = lowbit.bit_length() - 1
                new_bits = current_bits & ~(1 << idx)
                new_size = new_bits.bit_count()
                delta = new_size - current_size
                # Always accept removals (they maintain validity)
                # But this shrinks the set, so only do it probabilistically
                if rng.random() < 0.3 or delta >= 0:
                    current_bits = new_bits
                    current_size = new_size
        else:
            # Swap: remove one, add one (net zero or could be invalid)
            if current_bits:
                # Pick vertex to remove
                v = current_bits
                count = current_bits.bit_count()
                pick = rng.randint(0, count - 1)
                temp = v
                for _ in range(pick):
                    temp &= temp - 1
                lowbit = temp & (-temp)
                idx_remove = lowbit.bit_length() - 1

                # Find candidates to add: not in current_bits, and not conflicting with (current_bits - idx_remove)
                base = current_bits & ~(1 << idx_remove)
                candidates = ~base & all_mask
                if candidates:
                    count_c = candidates.bit_count()
                    pick_c = rng.randint(0, count_c - 1)
                    temp = candidates
                    for _ in range(pick_c):
                        temp &= temp - 1
                    lowbit_c = temp & (-temp)
                    idx_add = lowbit_c.bit_length() - 1
                    
                    new_bits = (current_bits & ~(1 << idx_remove)) | (1 << idx_add)
                    new_size = new_bits.bit_count()
                    # Check validity of the new set
                    valid = True
                    # Only need to check idx_add against base
                    if conflict_bits[idx_add] & base:
                        valid = False
                    
                    if valid:
                        delta = new_size - current_size
                        if delta >= 0 or rng.random() < math.exp(delta / max(T, T_min)):
                            if new_size > best_size:
                                best_size = new_size
                                best_bits = new_bits
                            current_bits = new_bits
                            current_size = new_size

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Occasionally reheat
            if rng.random() < 0.05:
                T = 0.5

    # Final local search: try to add any vertex that fits
    for _ in range(1000):
        improved = False
        for i in range(num_valid):
            if not (best_bits >> i) & 1:
                if not (conflict_bits[i] & best_bits):
                    best_bits |= (1 << i)
                    best_size += 1
                    improved = True
        if not improved:
            break

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
    for a in result:
        if 2 * a in squares:
            # Fallback
            result = sorted([valid_numbers[i] for i in range(num_valid) if (best_bits >> i) & 1 and 2 * valid_numbers[i] not in squares])
            break
        for b in result:
            if a <= b and (a + b) in squares:
                result = sorted([valid_numbers[i] for i in range(num_valid) if (best_bits >> i) & 1 and 2 * valid_numbers[i] not in squares])
                break

    return result
