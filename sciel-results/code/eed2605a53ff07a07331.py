def is_inside_triangle_vec(points):
    v0 = V1 - V0
    v1 = V2 - V1
    v2 = V0 - V2
    
    # s1 = cross(v0, p - V0)
    p_minus_v0 = points - V0
    s1 = p_minus_v0[:, 1] * v0[0] - p_minus_v0[:, 0] * v0[1]
    
    # s2 = cross(v1, p - V1)
    p_minus_v1 = points - V1
    s2 = p_minus_v1[:, 1] * v1[0] - p_minus_v1[:, 0] * v1[1]
    
    # s3 = cross(v2, p - V2)
    p_minus_v2 = points - V2
    s3 = p_minus_v2[:, 1] * v2[0] - p_minus_v2[:, 0] * v2[1]
    
    return np.all(s1 >= 0) and np.all(s2 >= 0) and np.all(s3 >= 0)
