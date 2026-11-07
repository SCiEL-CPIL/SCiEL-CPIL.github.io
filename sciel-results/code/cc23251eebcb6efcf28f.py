"""Rewritten program for the suite task; new design from scratch."""

def construct():
    n = 256
    max_prod = n * (n - 1)  # 256*255 = 65280

    # Precompute for each possible product p, the list of divisors d in [2, n]
    divisors_of = [[] for _ in range(max_prod + 1)]
    for d in range(2, n + 1):
        for m in range(d, max_prod + 1, d):
            divisors_of[m].append(d)

    # Build adjacency: for each vertex v, the set of other vertices u such that
    # there exists a third vertex w with {u, v, w} forming a conflict.
    # A conflict triple is (a, b, c) distinct with a | (b*c).
    # For a fixed pair (b, c), the conflicting a's are divisors of b*c in S \ {b, c}.
    # So for each pair (b, c), each such a is adjacent to both b and c (and vice versa).
    # We want: for each vertex x, the set of y such that x and y are in some conflict triple.

    # adjacency[x] = set of y != x such that there exists z != x, y with x|y*z or y|x*z or z|x*y
    # Actually the condition is: for distinct a,b,c in S, a does NOT divide b*c.
    # So a conflict exists for triple (a,b,c) if a | b*c.
    # The hyperedge is {a,b,c}. Two vertices are "adjacent" if they share a hyperedge.
    # So for pair (b,c), every a in divisors_of[b*c] with a != b, a != c creates adjacency between:
    #   a-b, a-c, b-c (all three pairs in the hyperedge are adjacent)

    # But for the greedy removal, we want the "degree" of each vertex = number of hyperedges containing it.
    # Let's compute for each vertex v, the count of hyperedges containing v.

    # Approach: iterate over all pairs (b, c) with b < c, find all a in divisors_of[b*c] \ {b, c},
    # and for each such a, increment degree of a, b, c.

    # But we need to be careful: the hyperedge is {a, b, c} where a, b, c are all distinct.
    # For a given pair (b, c), the valid a's are those in divisors_of[b*c] that are not b or c.
    # Each such a forms a hyperedge {a, b, c}.

    # Let's compute the degree of each vertex in the hypergraph.
    degree = [0] * (n + 1)

    # For efficiency, iterate over all pairs (b, c) with 2 <= b < c <= n
    for b in range(2, n + 1):
        for c in range(b + 1, n + 1):
            prod = b * c
            for a in divisors_of[prod]:
                if a == b or a == c:
                    continue
                # a, b, c are distinct, and a | b*c, so {a, b, c} is a conflict hyperedge
                degree[a] += 1
                degree[b] += 1
                degree[c] += 1

    # Now we want to find a maximum independent set in this 3-uniform hypergraph.
    # Greedy approach: start with all vertices, iteratively remove the vertex with highest degree.
    # But we need to recompute degrees after each removal.

    # Let's use a different approach: build the conflict hyperedges explicitly,
    # then use a greedy set cover / vertex removal heuristic.

    # Actually, let's use a smarter greedy: 
    # 1. Start with S = {2, 3, ..., 256}
    # 2. While there's a conflict in S:
    #    a. Find all conflict triples in S
    #    b. For each vertex, count how many conflict triples it's in
    #    c. Remove the vertex with the highest count
    # 3. Return S

    # To make this efficient, let's precompute for each pair (b, c) the list of a's,
    # and then maintain the set S and recompute degrees.

    # Let's store for each pair (b, c) the list of a's that form a conflict with them.
    # conflict_a_for_pair = {}  # (b, c) -> list of a's
    # But that's a lot of memory. Instead, let's just recompute on the fly.

    S = set(range(2, n + 1))

    while True:
        # Compute degree of each vertex in S
        deg = {x: 0 for x in S}
        has_conflict = False
        S_list = sorted(S)
        len_S = len(S_list)

        for i in range(len_S):
            b = S_list[i]
            for j in range(i + 1, len_S):
                c = S_list[j]
                prod = b * c
                for a in divisors_of[prod]:
                    if a in S and a != b and a != c:
                        has_conflict = True
                        deg[a] += 1
                        deg[b] += 1
                        deg[c] += 1

        if not has_conflict:
            break

        # Find vertex with max degree, tie-break by smallest value
        best = min(S, key=lambda x: (-deg[x], x))
        S.remove(best)

    return list(S)
