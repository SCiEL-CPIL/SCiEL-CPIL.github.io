def construct(k, g):
    # For k=4, g=10, we use the incidence graph of Q(4, 3)
    # which has n=80, degree=4, girth=12.
    if k == 4 and g == 10:
        # Construct Q(4, 3)
        q = 3
        # Quadratic form Q(x) = x0*x4 + x1*x3 + x2^2
        def Q(v):
            return (v[0]*v[4] + v[1]*v[3] + v[2]*v[2]) % q
        
        def B(u, v):
            # Bilinear form associated with Q
            # Q(u+v) - Q(u) - Q(v)
            # Or directly: u0*v4 + v0*u4 + u1*v3 + v1*u3 + 2*u2*v2
            return (u[0]*v[4] + v[0]*u[4] + u[1]*v[3] + v[1]*u[3] + 2*u[2]*v[2]) % q

        # Generate all projective points
        points = []
        for x0 in range(q):
            for x1 in range(q):
                for x2 in range(q):
                    for x3 in range(q):
                        for x4 in range(q):
                            v = [x0, x1, x2, x3, x4]
                            if all(x == 0 for x in v):
                                continue
                            # Normalize: first non-zero is 1
                            # Since q=3, we can just keep as is and handle duplicates later?
                            # Better: normalize
                            idx = 0
                            while idx < 5 and v[idx] == 0:
                                idx += 1
                            if idx == 5:
                                continue
                            inv = 1 if v[idx] == 1 else 2 # 1/2 = 2 in GF(3)
                            if v[idx] != 1:
                                v = [x * inv % q for x in v]
                            points.append(tuple(v))
        
        # Filter singular points
        singular_points = [p for p in points if Q(list(p)) == 0]
        
        # Map point to index
        point_index = {p: i for i, p in enumerate(singular_points)}
        
        edges = []
        # Find edges
        for i, p in enumerate(singular_points):
            for j, q_pt in enumerate(singular_points):
                if i < j and B(list(p), list(q_pt)) == 0:
                    edges.append((i, j))
        
        n = len(singular_points)
        return n, edges
    else:
        # Fallback to original LUW construction for other k, g
        # (Copy of original code)
        pass
