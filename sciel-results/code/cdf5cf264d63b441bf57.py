"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    points = [(0, 0)]
    
    # Directions for the hexagonal spiral
    # Order: North-West, West, South-West, South-East, East, North-East
    # Vectors: (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1)
    dirs = [(-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1)]
    
    # We need 32 points.
    # Ring 0: 1 point
    # Ring 1: 6 points (Total 7)
    # Ring 2: 12 points (Total 19)
    # Ring 3: 18 points (Total 37) -> We take 13 from this ring.
    
    k = 1
    while len(points) < 32:
        # Start of ring k is at (k, 0)
        # But we generate relative to the previous end?
        # It's easier to generate the ring points explicitly.
        # The ring k has 6k points.
        # We can generate the sequence of points for ring k.
        
        # Start point of ring k is (k, 0)
        # However, the spiral continues from the last point of ring k-1.
        # The last point of ring k-1 is (k-1, -(k-1)) ? 
        # Let's check Ring 1 last point: (1, -1).
        # Next point in Ring 2 is (2, 0).
        # Vector from (1, -1) to (2, 0) is (1, 1). Not a neighbor.
        # The spiral "jumps" to the new ring start.
        # But for the purpose of generating the set of points, we just need the coordinates.
        
        ring_points = []
        # Start at (k, 0)
        current = (k, 0)
        ring_points.append(current)
        
        # Traverse the 6 sides
        for i in range(6):
            dx, dy = dirs[i]
            for _ in range(k):
                current = (current[0] + dx, current[1] + dy)
                # The last step of the last direction returns to (k,0), so we skip adding it if it's the very last one of the ring
                if not (i == 5 and _ == k - 1):
                    ring_points.append(current)
        
        # Add points from this ring until we have 32
        for p in ring_points:
            if len(points) >= 32:
                break
            points.append(p)
            
        k += 1
        
    return [list(p) for p in points]
# EVOLVE-BLOCK-END
