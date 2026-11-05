import time
import random as _random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = _random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        rem = adj[:]  # remaining edges
        cliques = []

        # Helper to get degree of vertex v in remaining graph
        def get_deg(v):
            return rem[v].bit_count()

        # Helper to get neighbors of v in remaining graph
        def get_neighbors(v):
            return rem[v]

        # Main loop: repeatedly find a large clique, remove its edges
        while True:
            if time.time() > deadline:
                break

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem[i]:
                    has_edges = True
                    break
            
            if not has_edges:
                break

            # 1. Find vertex with max degree in remaining graph
            max_deg = -1
            start_v = -1
            for i in range(n):
                d = get_deg(i)
                if d > max_deg:
                    max_deg = d
                    start_v = i
            
            if max_deg == 0:
                break

            # 2. Greedily build a clique starting from start_v
            clique = [start_v]
            # Candidates are neighbors of the last added vertex that are connected to all previous vertices
            # Initially, candidates are neighbors of start_v
            candidates = get_neighbors(start_v)
            
            # We need to ensure candidates are mutually connected with the rest of the clique.
            # Standard greedy clique expansion:
            # C = {start_v}
            # P = N(start_v)
            # While P is not empty:
            #   Pick v in P with max degree in P (or max degree in rem)
            #   Add v to C
            #   P = P ∩ N(v)
            
            while candidates:
                if time.time() > deadline:
                    break
                
                # Find vertex in candidates with max degree within candidates
                # This heuristic tends to pick "central" vertices in the potential clique
                max_cand_deg = -1
                next_v = -1
                m = candidates
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    # Degree of v within candidates
                    deg_in_cand = (rem[v] & candidates).bit_count()
                    # Tie-breaker: degree in remaining graph
                    deg_in_rem = rem[v].bit_count()
                    if deg_in_cand > max_cand_deg or (deg_in_cand == max_cand_deg and deg_in_rem > get_deg(next_v) if next_v != -1 else False):
                        max_cand_deg = deg_in_cand
                        next_v = v
                    m ^= lsb
                
                if next_v == -1:
                    break
                
                clique.append(next_v)
                # Update candidates: must be neighbor of next_v AND in current candidates
                candidates &= rem[next_v]
                candidates ^= (1 << next_v) # Remove next_v from candidates (though intersection handles it if next_v not in N(next_v), but safe to be explicit)
                # Actually, candidates &= rem[next_v] is sufficient if we assume no self-loops.
                # rem[next_v] does not contain next_v. So next_v is removed from candidates automatically.

            if len(clique) < 2:
                # Should not happen if we picked max degree > 0, but safety check
                # If we couldn't form a clique of size >= 2, we might be stuck with isolated edges?
                # No, if max_deg > 0, start_v has a neighbor. candidates was non-empty.
                # So clique should have at least 2 vertices.
                pass

            # 3. Remove all edges within the clique from rem
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i + 1, len(clique)):
                    v = clique[j]
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)
            
            cliques.append(clique)

        # Fallback: cover any remaining edges as pairs
        # This handles any edges that might have been missed due to time break or logic errors
        for u in range(n):
            nm = rem[u]
            while nm:
                lsb = nm & -nm
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                nm ^= lsb
                rem[u] &= ~(1 << v)
                rem[v] &= ~(1 << u)

        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
