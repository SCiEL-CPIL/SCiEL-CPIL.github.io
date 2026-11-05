"""Combined program: SAT-based checker + multi-start random search + greedy pruning
for a 5-uniform non-2-colourable hypergraph (minimising edges)."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """Return a 5-uniform hypergraph with no proper 2-colouring, minimising edges.

    Strategy:
      1. Fast SAT-based check (unit propagation + backtracking) for non-2-colourability.
      2. Multi-start: try several n (9..12) and several random initial edge sets.
      3. Greedy pruning: repeatedly try removing an edge; keep if still non-2-colourable.
      4. Keep the best (fewest edges) found within the time budget.
    """
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    # ---------- Fast SAT solver for 2-colourability ----------
    # A hypergraph is 2-colourable iff there exists a 2-colouring (assignment of
    # True/False to vertices) such that every edge has both colours, i.e., no edge
    # is monochromatic.  We want to know if NO such colouring exists.
    #
    # Equivalent: is the CNF "for every edge, not(all same)" satisfiable?
    # For edge e = {v1..v5}, "not all same" means: not(all True) AND not(all False).
    #   not(all True):  (¬v1 ∨ ¬v2 ∨ ¬v3 ∨ ¬v4 ∨ ¬v5)
    #   not(all False): (v1 ∨ v2 ∨ v3 ∨ v4 ∨ v5)
    # So each edge gives two clauses. The hypergraph is 2-colourable iff this CNF
    # is satisfiable. Non-2-colourable iff the CNF is UNSAT.
    #
    # We implement a DPLL with unit propagation.

    def is_non_2colorable(edges, n):
        """Return True if the hypergraph has NO proper 2-colouring (i.e., is not 2-colourable)."""
        if not edges:
            return False  # empty hypergraph is trivially 2-colourable

        # For small n, use bitmask enumeration which is very fast in Python
        if n <= 10:
            edge_masks = []
            for e in edges:
                m = 0
                for v in e:
                    m |= (1 << v)
                edge_masks.append(m)
            for mask in range(1 << n):
                valid = True
                for em in edge_masks:
                    c = bin(em & mask).count('1')
                    if c == 0 or c == 5:
                        valid = False
                        break
                if valid:
                    return False
            return True
        else:
            # Build clauses: for each edge, two clauses (OR of all, OR of all negated)
            clauses = []
            for e in edges:
                clauses.append([(v, 1) for v in e])
                clauses.append([(v, -1) for v in e])

            # DPLL with unit propagation
            def dpll(clauses, assignment):
                """assignment: dict var -> True/False. Returns True if satisfiable."""
                changed = True
                while changed:
                    changed = False
                    new_clauses = []
                    for cl in clauses:
                        satisfied = False
                        new_cl = []
                        for (v, s) in cl:
                            if v in assignment:
                                if s * (1 if assignment[v] else -1) == 1:
                                    satisfied = True
                                    break
                            else:
                                new_cl.append((v, s))
                        if not satisfied:
                            if not new_cl:
                                return False
                            if len(new_cl) == 1:
                                v, s = new_cl[0]
                                assignment[v] = (s == 1)
                                changed = True
                            else:
                                new_clauses.append(new_cl)
                    clauses = new_clauses

                if not clauses:
                    return True

                # Pick a variable with highest frequency in remaining clauses
                var_freq = {}
                for cl in clauses:
                    for (v, s) in cl:
                        var_freq[v] = var_freq.get(v, 0) + 1
                v = max(var_freq, key=var_freq.get)
                
                # Try True first
                assignment[v] = True
                if dpll([list(c) for c in clauses], dict(assignment)):
                    return True
                del assignment[v]
                # Try False
                assignment[v] = False
                if dpll([list(c) for c in clauses], dict(assignment)):
                    return True
                del assignment[v]
                return False

            return not dpll([list(c) for c in clauses], {})

    # ---------- Greedy pruning with edge importance ordering ----------
    def prune_edges(edges, n, time_limit):
        """Try to remove edges while preserving non-2-colourability.
        Edges are tried in random order; when an edge can be removed, we restart
        the scan from the beginning to catch further removals."""
        current = list(edges)
        random.shuffle(current)
        
        while True:
            if time.time() > time_limit:
                break
            i = 0
            removed_any = False
            while i < len(current):
                if time.time() > time_limit:
                    break
                candidate = current[:i] + current[i+1:]
                if is_non_2colorable(candidate, n):
                    current = candidate
                    removed_any = True
                    # Restart from beginning to find more removals
                    break
                else:
                    i += 1
            if not removed_any:
                break
        return current

    # ---------- Multi-start search ----------
    best_edges = None
    best_n = 9
    best_len = float('inf')

    for n in [9, 10, 11, 12]:
        if time.time() > deadline - 8.0:
            break

        all_possible = list(combinations(range(n), 5))
        total = len(all_possible)

        # Try multiple random initial configurations
        for attempt in range(80):
            if time.time() > deadline - 5.0:
                break

            # Start with a moderately dense set - try different density ranges
            if attempt < 40:
                k_init = random.randint(45, min(65, total))
            else:
                k_init = random.randint(35, min(55, total))
            
            random.shuffle(all_possible)
            initial = [list(e) for e in all_possible[:k_init]]

            if not is_non_2colorable(initial, n):
                continue

            # Prune with time budget proportional to current best
            prune_time = min(2.0, max(0.5, (deadline - time.time()) * 0.1))
            pruned = prune_edges(initial, n, min(time.time() + prune_time, deadline - 3.0))

            if len(pruned) < best_len:
                best_edges = pruned
                best_n = n
                best_len = len(pruned)

                # If we've found a very good solution, adjust strategy
                if best_len <= 38:
                    if time.time() > deadline - 15.0:
                        break

        if time.time() > deadline - 12.0:
            break

    # Fallback: complete 5-graph on 9 vertices
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
