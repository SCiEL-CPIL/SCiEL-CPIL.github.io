<<<<<<< SEARCH
    # 3. MD initialized points (usually very good starting points)
    for seed in range(3):
        md_pts = _md_init(N, steps=1000, seed=seed)
        inits.append(md_pts)
=======
    # 3. MD initialized points (usually very good starting points)
    for seed in range(3):
        md_pts = _md_init(N, steps=1000, seed=seed)
        inits.append(md_pts)
    
    # 4. Geodesic dome initialization
    geodesic_pts = _geodesic_dome_init(N, subdivisions=4, seed=42)
    inits.append(geodesic_pts)
    
    # 5. Another geodesic dome with different subdivisions
    geodesic_pts2 = _geodesic_dome_init(N, subdivisions=3, seed=123)
    inits.append(geodesic_pts2)
>>>>>>> REPLACE
