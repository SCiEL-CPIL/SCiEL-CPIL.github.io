"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    d = 7
    n = 2 ** d

    # Precompute all edges as (u, v, bit) where u < v
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))

    # Precompute for each edge the list of (other_edge1, other_edge2, other_edge3)
    # such that {edge, oe1, oe2, oe3} forms a C4.
    # Edge (x,y) with bit i: for each j != i, the C4 is {x,y}, {x^j,y^j}, {x,x^j}, {y,y^j}
    # So the 3 other edges are: (min(x^j,y^j), max(x^j,y^j)), (min(x,x^j),max(x,x^j)), (min(y,y^j),max(y,y^j))

    edge_cycles = {}  # (x,y) -> list of tuples of 3 other edges
    for (x, y, bit_i) in all_edges:
        cycles = []
        for j in range(d):
            if j == bit_i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            oe1 = (min(x ^ (1 << j), y ^ (1 << j)), max(x ^ (1 << j), y ^ (1 << j)))
            oe2 = (min(x, c), max(x, c))
            oe3 = (min(y, e), max(y, e))
            cycles.append((oe1, oe2, oe3))
        edge_cycles[(x, y)] = cycles

    def is_valid_add(chosen, edge):
        """Check if adding edge to chosen creates a C4."""
        x, y, bit_i = edge
        for (oe1, oe2, oe3) in edge_cycles[(x, y)]:
            if oe1 in chosen and oe2 in chosen and oe3 in chosen:
                return False
        return True

    def greedy_randomized(seed):
        """Run greedy with randomized edge order."""
        rng = random.Random(seed)
        edges = list(all_edges)
        rng.shuffle(edges)

        chosen = set()
        out = []

        for edge in edges:
            x, y, bit_i = edge
            if is_valid_add(chosen, edge):
                chosen.add((x, y))
                out.append([x, y])

        return out, chosen

    def try_add_edges(chosen, out, rng, max_attempts=100):
        """Try to add as many edges as possible from random candidates."""
        added = 0
        for _ in range(max_attempts):
            candidate = rng.choice(all_edges)
            cx, cy, cbit = candidate
            if (cx, cy) not in chosen and is_valid_add(chosen, candidate):
                chosen.add((cx, cy))
                out.append([cx, cy])
                added += 1
        return added

    def systematic_add(chosen, out):
        """Try to add all possible edges systematically."""
        added = 0
        for edge in all_edges:
            x, y, bit_i = edge
            if (x, y) not in chosen and is_valid_add(chosen, edge):
                chosen.add((x, y))
                out.append([x, y])
                added += 1
        return added

    def local_search(out, chosen, time_remaining, rng):
        """Local search: try 1-remove + 2-add swaps and pure adds."""
        n_edges = len(all_edges)
        best_size = len(out)
        no_improve_count = 0

        while time.time() - start_time < time_limit - 3.0:
            # First try systematic additions
            if systematic_add(chosen, out) > 0:
                best_size = len(out)
                no_improve_count = 0
                continue

            # Try 1-remove, 2-add swap
            idx = rng.randint(0, len(out) - 1)
            ex, ey = out[idx]
            edge_key = (ex, ey)

            # Remove the edge
            chosen.discard(edge_key)
            out.pop(idx)

            # Try to add up to 2 edges systematically
            added_count = 0
            added_edges = []
            for candidate in all_edges:
                cx, cy, cbit = candidate
                if (cx, cy) not in chosen and is_valid_add(chosen, candidate):
                    chosen.add((cx, cy))
                    out.append([cx, cy])
                    added_edges.append((cx, cy))
                    added_count += 1
                    if added_count >= 2:
                        break

            # If we didn't add at least 1, revert
            if added_count == 0:
                chosen.add(edge_key)
                out.append([ex, ey])
                no_improve_count += 1
            else:
                no_improve_count = 0
                if len(out) > best_size:
                    best_size = len(out)

            # If stuck, try a more aggressive random restart of local search
            if no_improve_count > 20:
                # Randomly remove 3-5 edges and try to re-add
                remove_count = rng.randint(3, 5)
                removed = []
                for _ in range(remove_count):
                    if not out:
                        break
                    idx = rng.randint(0, len(out) - 1)
                    removed.append(out.pop(idx))
                    chosen.discard(removed[-1][0], removed[-1][1]) if False else chosen.discard((removed[-1][0], removed[-1][1]))
                
                # Try to re-add
                for edge in all_edges:
                    x, y, bit_i = edge
                    if (x, y) not in chosen and is_valid_add(chosen, edge):
                        chosen.add((x, y))
                        out.append([x, y])
                
                no_improve_count = 0
                if len(out) > best_size:
                    best_size = len(out)

        return out, chosen

    # Run multiple randomized greedy restarts
    best_result = None
    best_count = 0

    # Use more restarts since local search is now faster
    num_restarts = 60
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break

        rng = random.Random(restart * 12345 + 67890)
        out, chosen = greedy_randomized(restart)

        if len(out) > best_count:
            best_count = len(out)
            best_result = [list(e) for e in out]

        # Local search improvement
        if time.time() - start_time < time_limit - 5.0:
            out, chosen = local_search(out, chosen, time_limit - time.time() - start_time, rng)
            if len(out) > best_count:
                best_count = len(out)
                best_result = [list(e) for e in out]

    # Final polish: try to add any remaining edges
    if best_result is not None:
        final_chosen = set(tuple(e) for e in best_result)
        for edge in all_edges:
            x, y, bit_i = edge
            if (x, y) not in final_chosen and is_valid_add(final_chosen, edge):
                best_result.append([x, y])
                final_chosen.add((x, y))

    if best_result is None:
        # Fallback to deterministic greedy
        chosen = set()
        best_result = []
        for edge in all_edges:
            x, y, bit_i = edge
            if is_valid_add(chosen, edge):
                chosen.add((x, y))
                best_result.append([x, y])

    return best_result
# EVOLVE-BLOCK-END
