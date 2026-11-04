<<<<<<< SEARCH
def _build_config(r2s, oa):
    """Build centers and angles for the 11 hexagons.
    r2s: 4 radial distances for the second ring (positions at 30, 150, 210, 330 deg)
    oa: outer hexagon rotation angle (radians)
    All inner hexagons are axis-aligned (angle=0).
    """
    d = S3  # first ring distance
    centers = np.empty((N, 2))
    angles = np.zeros(N)
    centers[0] = [0.0, 0.0]
    for i in range(6):
        th = i * math.pi / 3.0
        centers[1 + i] = [d * math.cos(th), d * math.sin(th)]
    # Second ring at 30, 150, 210, 330 degrees (i.e., (2k+1)*30 deg for k=0..3)
    r2_thetas = [math.pi / 6.0, 5.0 * math.pi / 6.0, 7.0 * math.pi / 6.0, 11.0 * math.pi / 6.0]
    for k in range(4):
        centers[7 + k] = [r2s[k] * math.cos(r2_thetas[k]), r2s[k] * math.sin(r2_thetas[k])]
    return centers, angles
=======
def _build_config(r2s, oa):
    """Build centers and angles for the 11 hexagons.
    r2s: 4 radial distances for the second ring (positions at 30, 150, 210, 330 deg)
    oa: outer hexagon rotation angle (radians)
    All inner hexagons are axis-aligned (angle=0).
    """
    d = S3 * 1.001  # first ring distance with epsilon for disjointness
    centers = np.empty((N, 2))
    angles = np.zeros(N)
    centers[0] = [0.0, 0.0]
    for i in range(6):
        th = i * math.pi / 3.0
        centers[1 + i] = [d * math.cos(th), d * math.sin(th)]
    # Second ring at 30, 150, 210, 330 degrees (i.e., (2k+1)*30 deg for k=0..3)
    r2_thetas = [math.pi / 6.0, 5.0 * math.pi / 6.0, 7.0 * math.pi / 6.0, 11.0 * math.pi / 6.0]
    for k in range(4):
        # Ensure r2s are at least 3.0 * 1.001 to avoid overlap with first ring
        r2_val = max(r2s[k], 3.0 * 1.001)
        centers[7 + k] = [r2_val * math.cos(r2_thetas[k]), r2_val * math.sin(r2_thetas[k])]
    return centers, angles
>>>>>>> REPLACE

