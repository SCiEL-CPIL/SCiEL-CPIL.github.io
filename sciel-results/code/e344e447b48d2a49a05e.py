import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin before 120s kill

    d = 7
    n = 2 ** d

    # Precompute all edges as (u, v, bit) where u < v
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    # Map edge tuple to index for faster lookup if needed, 
    # but set of tuples is fast enough in Python for 448 items.
    
    # Precompute for each edge the list of (other_edge1, other_edge2, other_edge3)
    # such that {edge, oe1, oe2, oe3} forms a C4.
    # Edge (x,y) with bit i: for each j != i, the C4 is {x,y}, {x^j,y^j}, {x,x^j}, {y,y^j}
    
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

    def simulated_annealing(out, chosen, rng, start_time, time_limit):
        """
        Simulated Annealing to improve the solution.
        Moves:
        1. Try to add a random edge. If valid, accept.
        2. If stuck (no add possible in a batch), remove a random edge with probability p.
        """
        current_score = len(chosen)
        best_score = current_score
        best_out = [list(e) for e in out]
        
        # Annealing parameters
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        # To speed up, we don't check every single add individually if we are stuck,
        # but rather try a few random adds.
        
        while time.time() - start_time < time_limit - 2.0:
            # Try to add a random edge
            added = False
            # Try a few random candidates
            for _ in range(10):
                candidate = rng.choice(all_edges)
                cx, cy, cbit = candidate
                if (cx, cy) not in chosen:
                    if is_valid_add(chosen, candidate):
                        chosen.add((cx, cy))
                        out.append([cx, cy])
                        current_score += 1
                        added = True
                        if current_score > best_score:
                            best_score = current_score
                            best_out = [list(e) for e in out]
                        break
            
            if not added:
                # Stuck. Try to remove an edge to escape local minimum.
                # Probability of removal increases as T is high, but we always want to explore.
                # Simple heuristic: remove a random edge.
                # We accept removal because it allows future additions.
                # In SA, we'd check energy, but here removal is always "worse" locally,
                # but necessary for global search.
                # We can be aggressive: remove 1 edge, then try to add 2.
                if out:
                    idx = rng.randint(0, len(out) - 1)
                    ex, ey = out[idx]
                    edge_key = (ex, ey)
                    
                    # Remove
                    chosen.discard(edge_key)
                    out.pop(idx)
                    current_score -= 1
                    
                    # Try to add back 2 edges
                    added_count = 0
                    added_edges = []
                    for _ in range(20):
                        candidate = rng.choice(all_edges)
                        cx, cy, cbit = candidate
                        if (cx, cy) not in chosen:
                            if is_valid_add(chosen, candidate):
                                chosen.add((cx, cy))
                                out.append([cx, cy])
                                added_edges.append((cx, cy))
                                added_count += 1
                                current_score += 1
                                if current_score > best_score:
                                    best_score = current_score
                                    best_out = [list(e) for e in out]
                                if added_count >= 2:
                                    break
                    
                    # If we didn't recover, we are at a lower score.
                    # This is allowed in SA, but we track best.
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min

        return best_out, best_score

    # Run multiple randomized greedy restarts
    best_result = None
    best_count = 0

    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break

        rng = random.Random(restart * 12345 + 67890)
        out, chosen = greedy_randomized(restart)

        if len(out) > best_count:
            best_count = len(out)
            best_result = [list(e) for e in out]

        # Local search improvement
        if time.time() - start_time < time_limit - 5.0:
            out, score = simulated_annealing(out, chosen, rng, start_time, time_limit - 2.0)
            if score > best_count:
                best_count = score
                best_result = out

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
