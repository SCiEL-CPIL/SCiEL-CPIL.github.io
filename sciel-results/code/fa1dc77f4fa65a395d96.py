def construct():
    """
    Constructs a graph on vertices 0..11 with minimum degree 3,
    girth at least 4 (triangle-free), and exactly 2 distinct simple cycle lengths.
    
    The graph consists of two disjoint copies of K_{3,3}.
    K_{3,3} is bipartite, so all cycles are even.
    The smallest cycle in K_{3,3} is length 4.
    The next smallest is length 6.
    K_{3,3} does not contain any cycles of length other than 4 and 6?
    Actually, K_{3,3} has cycles of length 4 and 6.
    Are there longer cycles?
    A simple cycle in K_{3,3} must alternate between the two parts of size 3.
    So the length must be 2k.
    Since each part has size 3, the maximum length of a simple cycle is 2*3 = 6.
    So the only possible cycle lengths in K_{3,3} are 4 and 6.
    Both exist.
    Thus, the set of cycle lengths is {4, 6}.
    The number of distinct lengths is 2.
    """
    
    edges = []
    
    # Helper to add edges of K_{m,n} with vertices offset
    def add_k_m_n(m, n, offset):
        part_a = list(range(offset, offset + m))
        part_b = list(range(offset + m, offset + m + n))
        for u in part_a:
            for v in part_b:
                if u < v:
                    edges.append([u, v])
                else:
                    edges.append([v, u])
                    
    # First K_{3,3}: vertices 0,1,2 and 3,4,5
    add_k_m_n(3, 3, 0)
    
    # Second K_{3,3}: vertices 6,7,8 and 9,10,11
    add_k_m_n(3, 3, 6)
    
    # Ensure no duplicates and correct format
    # The problem states "Return edges as [u, v] lists"
    # It also says "no loops or duplicates"
    # My construction ensures u != v and u < v for unique representation.
    
    return edges
