"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = 128

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Precompute bad_triples_mask[a][b]: for each pair (a,b), which c make {a,b,c} a bad triple
    bad_triples_mask = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    mask |= (1 << c)
            bad_triples_mask[a][b] = mask
            bad_triples_mask[b][a] = mask

    def popcount(x):
        return bin(x).count('1')

    def greedy(ordering):
        selected = []
        selected_set = set()
        forbidden = 0

        for x in ordering:
            if time.time() > deadline:
                break
            if x in selected_set:
                continue
            if (forbidden >> x) & 1:
                continue

            selected.append(x)
            selected_set.add(x)

            for a in selected[:-1]:
                forbidden |= bad_triples_mask[a][x]

        return selected

    def local_search(selected, max_iterations=100):
        """Try to improve the solution via single-element swaps."""
        if not selected or time.time() > deadline:
            return selected

        all_elements = set(range(1, n + 1))
        best_set = set(selected)
        best_size = len(best_set)

        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            improved = False
            unselected = all_elements - best_set

            for rem in list(best_set):
                if time.time() > deadline:
                    break

                new_set = best_set - {rem}

                new_forbidden = 0
                new_list = list(new_set)
                for i in range(len(new_list)):
                    for j in range(i + 1, len(new_list)):
                        new_forbidden |= bad_triples_mask[new_list[i]][new_list[j]]

                for add in unselected:
                    if add in new_set:
                        continue
                    if (new_forbidden >> add) & 1:
                        continue

                    candidate_set = new_set | {add}
                    if len(candidate_set) > best_size:
                        best_set = candidate_set
                        best_size = len(best_set)
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return list(best_set)

    def branch_and_bound(available_mask, selected_list, current_size, best_size_ref):
        """
        Branch and bound to find a large valid subset.
        Uses a strong upper bound based on the conflict graph structure.
        """
        # Compute forbidden based on selected_list
        if selected_list:
            forbidden = 0
            sel = selected_list
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    forbidden |= bad_triples_mask[sel[i]][sel[j]]
            truly_available = available_mask & ~forbidden
        else:
            truly_available = available_mask

        m_avail = popcount(truly_available)
        upper_bound = current_size + m_avail

        if upper_bound <= best_size_ref[0]:
            return selected_list[:]

        if time.time() > deadline:
            return selected_list[:]

        if truly_available == 0:
            return selected_list[:]

        # Get list of truly available elements
        avail_list = []
        x = 1
        while x <= n:
            if (truly_available >> x) & 1:
                avail_list.append(x)
            x += 1

        m = len(avail_list)
        if m == 0:
            return selected_list[:]

        # Build conflict graph on avail_list
        adj = [0] * m
        for i in range(m):
            ui = avail_list[i]
            for j in range(i + 1, m):
                uj = avail_list[j]
                conflict_mask = bad_triples_mask[ui][uj] & truly_available
                conflict_mask &= ~(1 << ui) & ~(1 << uj)
                if conflict_mask:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        # Compute degrees
        degrees = [popcount(adj[i]) for i in range(m)]
        max_deg = max(degrees) if m > 0 else 0

        # Stronger upper bound: use the fact that in the conflict graph,
        # any independent set can have at most one vertex from each connected component's
        # "core". A simple but effective bound: if there are k vertices with degree > 0,
        # and the graph has e edges, then alpha <= m - e/(m-1) is not valid.
        # Instead, use: alpha <= m - floor(sum_of_degrees / (2 * (m-1))) ... still not clean.
        # 
        # Practical approach: if max_deg >= m - 1, then at most 1 vertex can be selected
        # from the neighborhood of that vertex plus the vertex itself.
        # More generally: for the vertex v with max degree D, any independent set either
        # contains v (then at most 1 + (m-1-D) vertices total) or not (at most m-1).
        # So alpha <= max(m-D, m-1). This is useful when D is large.
        if m > 1:
            ub_from_degree = max(m - max_deg, m - 1) if max_deg > 0 else m
            if ub_from_degree < m_avail:
                upper_bound = current_size + ub_from_degree
                if upper_bound <= best_size_ref[0]:
                    return selected_list[:]

        # If m is small, we can do exhaustive search
        if m <= 20:
            # Try all subsets in order of decreasing size
            # This is only feasible for small m
            best_local = selected_list[:]
            # Use a simple recursive approach for small m
            def small_bb(avail_mask, sel, sz):
                nonlocal best_local, best_size_ref
                if sz + popcount(avail_mask) <= len(best_local):
                    return
                if time.time() > deadline:
                    return
                if avail_mask == 0:
                    if sz > len(best_local):
                        best_local = sel[:]
                    return
                
                # Pick a vertex
                # Find first set bit
                v = 1
                while v <= n and not ((avail_mask >> v) & 1):
                    v += 1
                if v > n:
                    return
                
                # Option 1: exclude v
                small_bb(avail_mask & ~(1 << v), sel, sz)
                
                # Option 2: include v
                new_forbidden = 0
                for a in sel:
                    new_forbidden |= bad_triples_mask[a][v]
                new_avail = avail_mask & ~(1 << v) & ~new_forbidden
                # Also exclude conflicts
                for u in range(1, n + 1):
                    if u == v:
                        continue
                    if (avail_mask >> u) & 1:
                        if bad_triples_mask[v][u] & avail_mask:
                            new_avail &= ~(1 << u)
                small_bb(new_avail, sel + [v], sz + 1)
            
            small_bb(truly_available, selected_list[:], current_size)
            if len(best_local) > len(selected_list):
                return best_local
            return selected_list[:]

        # Branching: pick vertex with maximum degree
        max_deg_vertices = [i for i in range(m) if degrees[i] == max_deg]
        pivot = max_deg_vertices[0]
        pivot_val = avail_list[pivot]
        
        best_result = selected_list[:]
        
        # Option 1: Exclude pivot
        exclude_mask = truly_available & ~(1 << pivot_val)
        result_excl = branch_and_bound(exclude_mask, selected_list, current_size, best_size_ref)
        if len(result_excl) > len(best_result):
            best_result = result_excl
            if len(result_excl) > best_size_ref[0]:
                best_size_ref[0] = len(result_excl)
        
        # Option 2: Include pivot
        if time.time() > deadline:
            return best_result
        
        # If we include pivot, we must exclude all its neighbors in the conflict graph
        # and also exclude any element that would form a bad triple with pivot and
        # some element in selected_list
        if selected_list:
            new_forbidden = 0
            for a in selected_list:
                new_forbidden |= bad_triples_mask[a][pivot_val]
        else:
            new_forbidden = 0
        
        # Elements to exclude: neighbors in conflict graph + forbidden by selected
        exclude_from_include = 0
        for j in range(m):
            if j != pivot and (adj[pivot] >> j) & 1:
                exclude_from_include |= (1 << avail_list[j])
        exclude_from_include |= new_forbidden
        exclude_from_include |= (1 << pivot_val)
        
        include_available = truly_available & ~exclude_from_include
        
        # Check if including pivot can still beat best
        if current_size + 1 + popcount(include_available) > best_size_ref[0]:
            new_selected = selected_list + [pivot_val]
            result_incl = branch_and_bound(include_available, new_selected, current_size + 1, best_size_ref)
            if len(result_incl) > len(best_result):
                best_result = result_incl
                if len(result_incl) > best_size_ref[0]:
                    best_size_ref[0] = len(result_incl)
        
        return best_result

    random.seed(42)

    best_result = []
    best_size = 0

    # Generate diverse base orderings
    base_orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count() * 10 + x % 7),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count() * 10 + x % 5),
        sorted(range(1, n + 1), key=lambda x: (x & 1, x.bit_count(), x)),
        sorted(range(1, n + 1), key=lambda x: (x & 3, -x.bit_count(), x)),
        sorted(range(1, n + 1), key=lambda x: (x & 7, x.bit_count(), x)),
        sorted(range(1, n + 1), key=lambda x: (x % 5, -x.bit_count(), x)),
    ]

    # Phase 1: Greedy + local search to get a good initial solution
    num_restarts = 3000

    for i in range(num_restarts):
        if time.time() > deadline:
            break

        if i < len(base_orderings):
            ordering = base_orderings[i]
        else:
            ordering = list(range(1, n + 1))
            random.shuffle(ordering)

        selected = greedy(ordering)

        if len(selected) > best_size:
            improved = local_search(selected)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved
            else:
                best_size = len(selected)
                best_result = selected
        elif len(selected) == best_size and best_result:
            improved = local_search(selected)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved

    # Phase 2: Branch and bound
    if time.time() < deadline - 5.0:
        full_mask = 0
        for i in range(1, n + 1):
            full_mask |= (1 << i)

        best_size_ref = [best_size]

        # Try B&B from empty set
        result = branch_and_bound(full_mask, [], 0, best_size_ref)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result

        # Try B&B starting from subsets of the current best solution
        if best_result and time.time() < deadline - 3.0:
            for rem in best_result[:15]:
                if time.time() > deadline - 2.0:
                    break

                remaining = [x for x in best_result if x != rem]
                remaining_set = set(remaining)
                avail = 0
                for i in range(1, n + 1):
                    if i not in remaining_set:
                        avail |= (1 << i)

                result = branch_and_bound(avail, remaining, len(remaining), best_size_ref)
                if len(result) > best_size:
                    best_size = len(result)
                    best_result = result

    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
