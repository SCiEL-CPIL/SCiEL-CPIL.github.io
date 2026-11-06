"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # We want to find the maximum radius r such that we can place N circles.
    # The configuration is determined by r.
    # For a given r, we place circles in concentric rings.
    # Ring 0: 1 circle at center (if r allows, which it always does for small r)
    # Ring k: circles at distance d_k from center.
    # The distance d_k is determined by the constraint that circles in ring k
    # must not overlap with circles in ring k-1 (and vice versa).
    # In a dense hexagonal-like packing, the vertical distance between rings 
    # is r * sqrt(3). However, because the rings are circular, the geometry is slightly different.
    
    # A simpler robust model for "Ring Packing":
    # Let d_k be the radius of the k-th ring (distance from origin).
    # The number of circles in ring k is M_k = floor(2 * pi * d_k / (2 * r)).
    # Actually, the chord length between adjacent circles in a ring is 2*r.
    # So M_k = floor(2 * pi * d_k / (2 * r)) = floor(pi * d_k / r).
    # Constraint: Circles in ring k and k-1 must not overlap.
    # The minimum distance between a circle in ring k and a circle in ring k-1 
    # is approximately sqrt(d_k^2 + d_{k-1}^2 - 2 d_k d_{k-1} cos(angle_diff)).
    # For dense packing, we assume the rings are "interlocked".
    # A common approximation is that the radial separation between rings is r * sqrt(3) 
    # IF the angular offset allows it.
    # However, a stricter constraint is that the distance between centers of adjacent
    # rings must be >= 2r.
    
    # Let's use a simpler iterative construction for a given r:
    # 1. Start with center (0,0). Count = 1.
    # 2. Try to add rings.
    #    Let current_max_radius = 1 - r.
    #    We place rings at distances d_1, d_2, ...
    #    The condition for placing a circle at (d, theta) given existing circles is complex.
    #    
    # Alternative Strategy: 
    # Use the known optimal packing structure for small N and extrapolate? No.
    # 
    # Let's use a "Greedy Ring" approach with binary search on r.
    # For a fixed r, can we place N circles?
    # 
    # Algorithm for fixed r:
    # 1. Place circle at (0,0).
    # 2. For k=1, 2, ...:
    #    Determine max number of circles in ring k.
    #    The distance from center to ring k, d_k, is constrained by the previous ring.
    #    To maximize density, we want d_k to be as small as possible while maintaining 
    #    distance 2r from circles in ring k-1.
    #    
    #    Let M_{k-1} be the number of circles in ring k-1 at distance d_{k-1}.
    #    The angular spacing in ring k-1 is 2*pi/M_{k-1}.
    #    A circle in ring k at angle 0 is closest to circles in ring k-1 at angles 
    #    +/- pi/M_{k-1} (if M_{k-1} is even) or similar.
    #    Distance squared: d_k^2 + d_{k-1}^2 - 2 d_k d_{k-1} cos(pi/M_{k-1}) >= (2r)^2.
    #    This gives a quadratic for d_k.
    #    
    #    Once d_k is determined, M_k = floor(2*pi*d_k / (2*r)).
    #    Also, we must ensure circles in ring k don't overlap with each other:
    #    2*pi*d_k / M_k >= 2r.
    #    
    #    We continue until d_k > 1-r or we have placed N circles.
    
    def can_place(r):
        if r <= 0:
            return True
        if r > 0.5: # Center circle alone
            return N <= 1
        if r > 1.0:
            return False
            
        placed = 1
        if placed >= N:
            return True
            
        d_prev = 0.0
        M_prev = 1
        
        # We will iterate rings
        # Note: The first ring around the center is special.
        # Distance from center (0,0) to ring 1 circles: d_1.
        # d_1^2 >= (2r)^2 => d_1 >= 2r.
        # Also, circles in ring 1 must fit in the container: d_1 + r <= 1 => d_1 <= 1-r.
        # Max circles in ring 1: M_1 = floor(2*pi*d_1 / (2*r)).
        
        # To maximize count, we should pick the smallest valid d_k?
        # Actually, for a fixed r, the positions are somewhat fixed by the "tightest" constraint.
        # If we pick d_k too large, we might skip a ring or reduce density?
        # No, larger d_k allows more circles in that ring.
        # But we want to fit N circles in the smallest possible area?
        # No, we just want to fit N circles in the unit disk.
        # So for a given r, we should try to pack as many as possible.
        # We should use the LARGEST possible d_k that fits in the container?
        # No, if we use largest d_k, we use the outer boundary.
        # But we also have inner rings.
        
        # Let's simulate the "greedy" filling from inside out.
        # Actually, the "tightest" packing usually corresponds to the minimum d_k allowed by the previous ring.
        # If we make d_k larger, the angle between circles in ring k increases (fewer circles),
        # but the distance to ring k-1 increases (easier to satisfy).
        # However, making d_k larger pushes subsequent rings further out, potentially reducing their capacity.
        # So we should use the MINIMUM valid d_k for each ring.
        
        d_curr = 0.0
        M_curr = 1
        
        for k in range(1, 100): # Max 100 rings
            # Calculate minimum d_k based on d_prev and M_prev
            if M_prev == 1:
                # Previous ring is just the center.
                # Distance from center to any point in ring k is d_k.
                # Must be >= 2r.
                d_min = 2.0 * r
            else:
                # Previous ring has M_prev circles at distance d_prev.
                # Angular spacing is 2*pi/M_prev.
                # Closest approach is at angle half the spacing: pi/M_prev.
                # Let circle in ring k be at angle 0.
                # Closest circles in ring k-1 are at angles +/- pi/M_prev.
                # dist^2 = d_k^2 + d_prev^2 - 2*d_k*d_prev*cos(pi/M_prev)
                # We need dist >= 2r
                # d_k^2 - 2*d_prev*cos(pi/M_prev)*d_k + d_prev^2 - 4r^2 >= 0
                # Roots of x^2 - B*x + C = 0 are (B +/- sqrt(B^2 - 4C))/2
                B = 2.0 * d_prev * np.cos(np.pi / M_prev)
                C = d_prev**2 - 4.0 * r**2
                disc = B**2 - 4.0 * C
                if disc < 0:
                    # No real solution? Means always valid? Or never?
                    # If disc < 0, the quadratic is always positive (since coeff of x^2 is 1).
                    # So any d_k > 0 works? 
                    # Wait, if disc < 0, the distance is always > 2r?
                    # Let's check. If d_prev is large and r is small, yes.
                    # In that case, we can set d_k as small as possible?
                    # But d_k must be > d_prev? No, rings are distinct.
                    # Usually d_k > d_prev.
                    # Let's assume d_k > d_prev.
                    d_min = d_prev + 1e-9
                else:
                    sqrt_disc = np.sqrt(disc)
                    # We want the larger root?
                    # The quadratic is >= 0 outside the roots.
                    # Since d_k must be positive and typically > d_prev, we take the larger root.
                    d_min = (B + sqrt_disc) / 2.0
                    # Ensure d_min is at least d_prev + small epsilon to maintain ring order
                    if d_min < d_prev:
                        d_min = d_prev + 1e-9
            
            # Check if this ring fits in the container
            if d_min + r > 1.0:
                break
            
            # Determine max circles in this ring
            # Circumference 2*pi*d_min.
            # Chord length 2r.
            # M_max = floor(2*pi*d_min / (2*r))
            M_max = int(np.floor(2.0 * np.pi * d_min / (2.0 * r)))
            
            if M_max <= 0:
                break
            
            # Add circles
            placed += M_max
            if placed >= N:
                return True
            
            # Update for next iteration
            d_prev = d_min
            M_prev = M_curr = M_max
            
        return placed >= N

    # Binary search for r
    # Lower bound: 0
    # Upper bound: 0.5 (only 1 circle)
    # Actually, upper bound can be tighter.
    # For N=300, r is approx 0.05.
    lo = 0.0
    hi = 0.5
    
    # Quick check to find a valid upper bound if needed, but 0.5 is safe.
    # Let's do 50 iterations of binary search
    for _ in range(50):
        if time.time() > deadline:
            break
        mid = (lo + hi) / 2.0
        if can_place(mid):
            lo = mid
        else:
            hi = mid
            
    r_opt = lo
    
    # Now construct the points for r_opt
    points = [(0.0, 0.0)]
    if r_opt > 0:
        d_prev = 0.0
        M_prev = 1
        for k in range(1, 100):
            if M_prev == 1:
                d_min = 2.0 * r_opt
            else:
                B = 2.0 * d_prev * np.cos(np.pi / M_prev)
                C = d_prev**2 - 4.0 * r_opt**2
                disc = B**2 - 4.0 * C
                if disc < 0:
                    d_min = d_prev + 1e-9
                else:
                    d_min = (B + np.sqrt(disc)) / 2.0
                    if d_min < d_prev:
                        d_min = d_prev + 1e-9
            
            if d_min + r_opt > 1.0:
                break
            
            M_max = int(np.floor(2.0 * np.pi * d_min / (2.0 * r_opt)))
            if M_max <= 0:
                break
            
            # Generate angles for this ring
            # Start angle 0? Or offset to maximize distance to previous ring?
            # The previous ring circles are at angles 2*pi*i/M_prev.
            # To maximize min distance, we should offset the new ring by half the angular step of the previous ring?
            # The constraint was derived assuming the closest approach is at pi/M_prev relative to the previous ring's circle at 0.
            # If we place the new ring's circle at 0, it is at angle 0.
            # The previous ring's circles are at 2*pi*i/M_prev.
            # The angle difference to the closest one is pi/M_prev (if M_prev is even) or similar.
            # So placing at 0 is consistent with the derivation.
            
            angles = np.linspace(0, 2*np.pi, M_max, endpoint=False)
            for theta in angles:
                x = d_min * np.cos(theta)
                y = d_min * np.sin(theta)
                points.append((x, y))
                
            if len(points) >= N:
                break
                
            d_prev = d_min
            M_prev = M_max
            
    # If we have more than N points, take the first N?
    # The first N points are the ones placed in the innermost rings, which are the most "constrained" and thus likely valid.
    # However, if we broke out of the loop because len(points) >= N, we might have added a full ring.
    # We should just take the first N.
    
    if len(points) < N:
        # Fallback to random if construction failed (should not happen if binary search worked)
        # Generate random points
        pts = []
        while len(pts) < N:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y <= (1-r_opt)**2:
                pts.append((x, y))
        points = pts
    else:
        points = points[:N]
        
    return np.array(points)
# EVOLVE-BLOCK-END
