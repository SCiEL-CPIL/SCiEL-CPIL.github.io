import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    d = 7
    n = 1 << d
    total_edges = n * d  # 448

    # Generate all edges with IDs
    edges = []
    edge_id_map = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_id = len(edges)
                edges.append((x, y))
                edge_id_map[(x, y)] = edge_id
                edge_id_map[(y, x)] = edge_id

    n_edges = len(edges)

    # Precompute all squares (2D faces) and conflict structure
    # Each square has 4 edges; a C4 is formed when all 4 edges are selected
    # We need to avoid any square having all 4 edges in our set
    squares = []  # list of tuples of 4 edge IDs
    edge_to_squares = {e: [] for e in range(n_edges)}

    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for base in range(n):
                if (base & mask_i) == 0 and (base & mask_j) == 0:
                    v0 = base
                    v1 = base ^ mask_i
                    v2 = base ^ mask_j
                    v3 = base ^ mask_i ^ mask_j
                    e0 = edge_id_map[(min(v0, v1), max(v0, v1))]
                    e1 = edge_id_map[(min(v0, v2), max(v0, v2))]
                    e2 = edge_id_map[(min(v1, v3), max(v1, v3))]
                    e3 = edge_id_map[(min(v2, v3), max(v2, v3))]
                    sq = (e0, e1, e2, e3)
                    squares.append(sq)
                    for e in sq:
                        edge_to_squares[e].append(len(squares) - 1)

    n_squares = len(squares)

    # For each edge, precompute its blocking squares (squares where it's one of the 4 edges)
    # An edge e is blocked if any square containing e has the other 3 edges all in chosen
    edge_blockers = {}
    for e in range(n_edges):
        blockers = []
        for sq_idx in edge_to_squares[e]:
            sq = squares[sq_idx]
            others = tuple(ed for ed in sq if ed != e)
            blockers.append((sq_idx, others))
        edge_blockers[e] = blockers

    def is_edge_blocked(chosen, e):
        """Check if edge e cannot be added to chosen (would create a C4)"""
        for sq_idx, others in edge_blockers[e]:
            if all(o in chosen for o in others):
                return True
        return False

    # Greedy initialization with random edge ordering
    def greedy_init(seed):
        rng = random.Random(seed)
        edge_order = list(range(n_edges))
        rng.shuffle(edge_order)
        chosen = set()
        for e in edge_order:
            if not is_edge_blocked(chosen, e):
                chosen.add(e)
        return chosen

    # Local search with fast targeted swaps
    def local_search(chosen, rng, time_limit):
        """Improve chosen set via local search"""
        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Phase 1: Try adding edges (pure additions)
            edges_to_try = [e for e in range(n_edges) if e not in chosen]
            rng.shuffle(edges_to_try)
            for e in edges_to_try:
                if time.time() >= time_limit:
                    break
                if not is_edge_blocked(chosen, e):
                    chosen.add(e)
                    improved = True

            # Phase 2: Targeted 1-for-1 swaps
            edges_in = list(chosen)
            rng.shuffle(edges_in)

            for e_remove in edges_in:
                if time.time() >= time_limit:
                    break
                # Find edges that would be unblocked by removing e_remove
                unblocked = []
                for sq_idx in edge_to_squares[e_remove]:
                    sq = squares[sq_idx]
                    for e_add in sq:
                        if e_add == e_remove or e_add in chosen:
                            continue
                        # Check if e_add is blocked only because of e_remove
                        blocked = False
                        all_blockers_contain_remove = True
                        for sq_idx2, others in edge_blockers[e_add]:
                            if all(o in chosen for o in others):
                                blocked = True
                                if e_remove not in others:
                                    all_blockers_contain_remove = False
                                    break
                        if blocked and all_blockers_contain_remove:
                            if e_add not in unblocked:
                                unblocked.append(e_add)
                if not unblocked:
                    continue
                rng.shuffle(unblocked)
                for e_add in unblocked:
                    new_chosen = chosen - {e_remove}
                    if not is_edge_blocked(new_chosen, e_add):
                        chosen.discard(e_remove)
                        chosen.add(e_add)
                        improved = True
                        break
                if improved:
                    break

        return chosen

    # Simulated annealing: perturb by removing edges, repair by adding
    def repair(chosen, rng, time_limit):
        """Add edges to chosen to make it maximal (no more edges can be added)"""
        while time.time() < time_limit:
            added = False
            edges_to_try = [e for e in range(n_edges) if e not in chosen]
            rng.shuffle(edges_to_try)
            for e in edges_to_try:
                if time.time() >= time_limit:
                    break
                if not is_edge_blocked(chosen, e):
                    chosen.add(e)
                    added = True
            if not added:
                break
        return chosen

    def simulated_annealing(initial_chosen, rng, time_limit):
        """SA to escape local optima"""
        current = set(initial_chosen)
        best = set(initial_chosen)
        best_size = len(current)
        
        T_start = 5.0
        T_end = 0.1
        cooling = 0.995
        T = T_start
        no_improve = 0
        
        while time.time() < time_limit:
            if T < T_end:
                T = T_start
                no_improve = 0
            
            # Perturb: remove random edges
            num_remove = rng.randint(2, 6)
            to_remove = rng.sample(list(current), min(num_remove, len(current)))
            perturbed = set(current)
            for e in to_remove:
                perturbed.discard(e)
            
            # Repair: add back as many edges as possible
            perturbed = repair(perturbed, rng, time_limit)
            
            # Local search
            perturbed = local_search(perturbed, rng, time_limit)
            
            perturbed_size = len(perturbed)
            delta = perturbed_size - len(current)
            
            if delta <= 0:
                current = perturbed
                no_improve = 0
                if perturbed_size > best_size:
                    best_size = perturbed_size
                    best = set(perturbed)
            else:
                # Accept with probability exp(-delta / T)
                if rng.random() < pow(2.718281828, -delta / T):
                    current = perturbed
                no_improve += 1
            
            T *= cooling
        
        return best

    # Run multiple random restarts
    best_chosen = set()
    best_size = 0

    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() >= deadline - 30:  # leave time for final SA
            break

        seed = restart * 12345 + 67890
        chosen = greedy_init(seed)
        size = len(chosen)

        if size > best_size:
            best_size = size
            best_chosen = set(chosen)

        # Local search on this solution
        rng = random.Random(seed)
        local_time = min(deadline - 25, time.time() + 8)
        chosen = local_search(chosen, rng, local_time)
        size = len(chosen)

        if size > best_size:
            best_size = size
            best_chosen = set(chosen)

    # Final SA polish on best solution
    if time.time() < deadline - 5 and len(best_chosen) > 0:
        rng = random.Random(99999)
        best_chosen = simulated_annealing(best_chosen, rng, deadline - 2)
        best_size = len(best_chosen)

    # Convert to list of [u, v]
    result = [[edges[e][0], edges[e][1]] for e in best_chosen]
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
