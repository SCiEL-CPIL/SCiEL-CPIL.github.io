def construct():
    """
    Constructs the unique maximal edge-critical diameter-2 graph on 16 vertices.

    Proof that K_{8,8} (64 edges) is optimal:
      1. Edge-critical diameter 2 requires that for every edge (u,v),
         u and v have NO common neighbor in G (otherwise removing (u,v)
         still leaves a 2-path via that common neighbor).
      2. Therefore G contains no triangle (every triangle edge has the
         third vertex as a common neighbor of the other two).
      3. By Mantel's theorem, a triangle-free graph on n vertices has
         at most floor(n^2/4) edges.
      4. For n=16: floor(256/4) = 64, achieved uniquely by K_{8,8}.

    Verification of K_{8,8} edge-criticality:
      - Diameter is 2: any two vertices in the same partition share
        all 8 vertices in the other partition as common neighbors.
      - Removing edge (a,b) with a in A, b in B: a and b now have
        zero common neighbors (all their other neighbors are in
        opposite partitions), so dist(a,b) = infinity > 2.
    """
    n = 16
    half = n // 2

    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])

    return edges
