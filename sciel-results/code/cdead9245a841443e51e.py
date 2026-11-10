# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from collections import defaultdict

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a
    nonzero square mod 205.

    Uses:
      - Bitset-based conflict graph for fast compatibility checks
      - Simulated annealing with 1-1, 2-2, and 3-3 swap moves
      - Multi-start with random greedy initializations
      - Systematic local search polish after each annealing run
    """
    m = 205
    deadline = time.time() + 580  # safety margin before 630s

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_set = forbidden

    # Build conflict graph as bitsets: conflict_mask[v] is a 205-bit int
    # where bit u is set iff (v - u) % m is in forbidden_set
    conflict_mask = [0] * m
    for v in range(m):
        mask = 0
        for u in range(m):
            if u == v:
                continue
            d = (v - u) % m
            if d in forbidden_set:
                mask |= (1 << u)
        conflict_mask[v] = mask

    full_mask = (1 << m) - 1

    def is_compatible(candidate, selected_bits):
        """Check if candidate is compatible with all elements in selected_bits (a bitmask)."""
        return (conflict_mask[candidate] & selected_bits) == 0

    def popcount(x):
        return bin(x).count('1')

    def get_vertices(bits):
        """Extract vertex list from bitmask."""
        verts = []
        t = bits
        while t:
            low = t & (-t)
            idx = low.bit_length() - 1
            verts.append(idx)
            t ^= low
        return verts

    def random_greedy():
        """Randomized greedy construction."""
        order = list(range(m))
        random.shuffle(order)
        selected_bits = 0
        for v in order:
            if is_compatible(v, selected_bits):
                selected_bits |= (1 << v)
        return selected_bits

    def local_search_add_only(selected_bits, max_iters=2000):
        """
        Systematic local search: repeatedly try to add any compatible vertex.
        Uses a priority queue approach to be more thorough.
        Returns improved bitmask and size.
        """
        current = selected_bits
        current_size = popcount(current)
        best = current
        best_size = current_size

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            # Try all unselected vertices in order of least conflicts
            unselected = ~current & full_mask
            if unselected == 0:
                break

            # Find the vertex with fewest conflicts among unselected
            best_v = -1
            best_conflicts = m + 1
            t = unselected
            while t:
                low = t & (-t)
                v = low.bit_length() - 1
                conflicts = popcount(conflict_mask[v] & current)
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_v = v
                t ^= low

            if best_v >= 0 and is_compatible(best_v, current):
                current = current | (1 << best_v)
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best = current
                continue
            else:
                break

        return best, best_size

    def systematic_local_search(selected_bits, max_iters=10000):
        """
        More thorough local search: try 1-1 swaps (remove one, add one)
        and 2-2 swaps systematically.
        Returns improved bitmask and size.
        """
        current = selected_bits
        current_size = popcount(current)
        best = current
        best_size = current_size
        all_vertices = list(range(m))

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            # Try to add any compatible vertex
            unselected = ~current & full_mask
            t = unselected
            while t:
                low = t & (-t)
                v = low.bit_length() - 1
                if is_compatible(v, current):
                    current = current | (1 << v)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best = current
                    break
                t ^= low

            if improved:
                continue

            # Try 1-1 swaps: remove one vertex, try to add another
            verts = get_vertices(current)
            found_swap = False
            for rem in verts:
                if time.time() > deadline:
                    break
                trial = current & ~(1 << rem)
                trial_size = current_size - 1

                # Find best candidate to add
                unselected = ~trial & full_mask
                t = unselected
                best_add = -1
                best_add_conflicts = m + 1
                while t:
                    low = t & (-t)
                    v = low.bit_length() - 1
                    if is_compatible(v, trial):
                        # This is a valid add
                        current = trial | (1 << v)
                        # size stays the same
                        found_swap = True
                        break
                    t ^= low

                if found_swap:
                    break

            if found_swap:
                continue

            # Try 2-2 swaps
            if current_size >= 2:
                verts = get_vertices(current)
                found_22 = False
                for i in range(min(len(verts), 20)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, min(len(verts), 20)):
                        rem1, rem2 = verts[i], verts[j]
                        trial = current & ~(1 << rem1) & ~(1 << rem2)
                        trial_size = current_size - 2

                        # Find two compatible vertices to add
                        unselected = ~trial & full_mask
                        t = unselected
                        added = 0
                        new_current = trial
                        while t and added < 2:
                            low = t & (-t)
                            v = low.bit_length() - 1
                            if is_compatible(v, new_current):
                                new_current = new_current | (1 << v)
                                added += 1
                            t ^= low

                        if added == 2:
                            current = new_current
                            # size stays the same
                            found_22 = True
                            break
                    if found_22:
                        break

                if found_22:
                    continue

            # No improvement possible
            break

        return best, best_size

    def simulated_annealing(initial_bits, start_temp=1.0, end_temp=0.001, max_iters=10000):
        """
        Simulated annealing with 1-1 and 2-2 swap moves.
        Returns (best_bits, best_size).
        """
        current = initial_bits
        current_size = popcount(current)
        best = current
        best_size = current_size

        def try_swap_1_1(state, state_size):
            """Try a random 1-1 swap: remove one vertex, add another."""
            if state_size == 0:
                temp = ~state & full_mask
                if temp:
                    candidates = get_vertices(temp)
                    if candidates:
                        v = random.choice(candidates)
                        if is_compatible(v, state):
                            return state | (1 << v), state_size + 1
                return state, state_size

            # Pick a random vertex to remove
            verts = get_vertices(state)
            if not verts:
                return state, state_size
            rem = random.choice(verts)
            trial = state & ~(1 << rem)

            # Try a few random candidates to add
            unselected = ~trial & full_mask
            cands2 = get_vertices(unselected)
            if not cands2:
                return state, state_size

            for _ in range(8):
                add_v = random.choice(cands2)
                if is_compatible(add_v, trial):
                    return trial | (1 << add_v), state_size
            return state, state_size

        def try_swap_2_2(state, state_size):
            """Try a random 2-2 swap: remove two vertices, add two others."""
            if state_size < 2:
                return state, state_size

            verts = get_vertices(state)
            if len(verts) < 2:
                return state, state_size
            rem1, rem2 = random.sample(verts, 2)
            trial = state & ~(1 << rem1) & ~(1 << rem2)

            unselected = ~trial & full_mask
            cands2 = get_vertices(unselected)
            if len(cands2) < 2:
                return state, state_size

            for _ in range(8):
                add1, add2 = random.sample(cands2, 2)
                if is_compatible(add1, trial) and is_compatible(add2, trial | (1 << add1)):
                    return trial | (1 << add1) | (1 << add2), state_size
            return state, state_size

        def try_swap_3_3(state, state_size):
            """Try a random 3-3 swap: remove three vertices, add three others."""
            if state_size < 3:
                return state, state_size

            verts = get_vertices(state)
            if len(verts) < 3:
                return state, state_size
            rem1, rem2, rem3 = random.sample(verts, 3)
            trial = state & ~(1 << rem1) & ~(1 << rem2) & ~(1 << rem3)

            unselected = ~trial & full_mask
            cands2 = get_vertices(unselected)
            if len(cands2) < 3:
                return state, state_size

            for _ in range(5):
                add1, add2, add3 = random.sample(cands2, 3)
                if (is_compatible(add1, trial) and
                    is_compatible(add2, trial | (1 << add1)) and
                    is_compatible(add3, trial | (1 << add1) | (1 << add2))):
                    return trial | (1 << add1) | (1 << add2) | (1 << add3), state_size
            return state, state_size

        n_iters = max_iters
        for it in range(n_iters):
            if time.time() > deadline:
                break

            # Linear cooling
            temp = start_temp * (end_temp / start_temp) ** (it / n_iters)

            # Choose move type
            r = random.random()
            if r < 0.35:
                new_state, new_size = try_swap_1_1(current, current_size)
            elif r < 0.65:
                new_state, new_size = try_swap_2_2(current, current_size)
            elif r < 0.80:
                new_state, new_size = try_swap_3_3(current, current_size)
            else:
                # Try to add a vertex
                temp_add = ~current & full_mask
                cands = get_vertices(temp_add)
                if cands:
                    v = random.choice(cands)
                    if is_compatible(v, current):
                        new_state = current | (1 << v)
                        new_size = current_size + 1
                    else:
                        new_state = current
                        new_size = current_size
                else:
                    new_state = current
                    new_size = current_size

            delta = new_size - current_size
            if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 1e-10)):
                current = new_state
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best = current

        # Final local search add-only polish
        best, best_size = local_search_add_only(best, max_iters=500)
        return best, best_size

    # Main search
    best_bits = 0
    best_size = 0

    # Initial deterministic greedy
    for v in range(m):
        if is_compatible(v, best_bits):
            best_bits |= (1 << v)
    best_size = popcount(best_bits)

    # Local search polish on initial
    ls_bits, ls_size = local_search_add_only(best_bits, max_iters=1000)
    if ls_size > best_size:
        best_size = ls_size
        best_bits = ls_bits

    # Systematic local search on initial
    sls_bits, sls_size = systematic_local_search(best_bits, max_iters=5000)
    if sls_size > best_size:
        best_size = sls_size
        best_bits = sls_bits

    # Multiple restarts with simulated annealing
    restart_count = 0
    while time.time() < deadline - 60:
        restart_count += 1

        # Random greedy initialization
        g_bits = random_greedy()
        g_size = popcount(g_bits)

        # Simulated annealing
        sa_bits, sa_size = simulated_annealing(g_bits, start_temp=1.0, end_temp=0.001, max_iters=5000)

        if sa_size > best_size:
            best_size = sa_size
            best_bits = sa_bits

        # Also try systematic local search on the best found so far
        if restart_count % 3 == 0:
            sls2_bits, sls2_size = systematic_local_search(best_bits, max_iters=3000)
            if sls2_size > best_size:
                best_size = sls2_size
                best_bits = sls2_bits

        # Occasionally try a longer annealing run from the current best
        if restart_count % 7 == 0 and time.time() < deadline - 120:
            sa2_bits, sa2_size = simulated_annealing(best_bits, start_temp=0.5, end_temp=0.001, max_iters=8000)
            if sa2_size > best_size:
                best_size = sa2_size
                best_bits = sa2_bits

    # Final polish
    final_bits, final_size = systematic_local_search(best_bits, max_iters=5000)
    if final_size > best_size:
        best_size = final_size
        best_bits = final_bits

    # Convert bitmask to array
    result = []
    for v in range(m):
        if best_bits & (1 << v):
            result.append(v)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
