"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 unit cell of a square-lattice photonic crystal
    that maximizes the TM band gap between bands 4 and 5.
    
    Strategy: Use a 'rod + hole' or 'cross-shaped' dielectric structure.
    The classic high-gap TM structure for square lattice is a dielectric rod
    in air, but to maximize the *fourth* band gap (which is higher up in frequency),
    we need a more complex geometry. 
    
    However, the standard "largest gap" often refers to the first gap. 
    The task specifies gap between 4th and 5th TM bands.
    For a square lattice, the TM bands are typically:
    1: Gamma
    2: X
    3: M
    4: ...
    
    Actually, let's look at the standard band structure for a square lattice of dielectric rods.
    The first TM gap is between band 2 and 3 (usually).
    The second TM gap is between band 4 and 5? Or is it between 3 and 4?
    
    Let's assume the evaluator orders bands by frequency at each k-point.
    The "gap between 4th and 5th bands" implies we are looking at a specific pair of bands
    that are 4th and 5th in the global ordering across the BZ, or locally at each k-point?
    The prompt says: "max omega_4 - min omega_5 ... between TM bands 4 and 5".
    This usually means the 4th lowest band and the 5th lowest band in the spectrum.
    
    For a square lattice photonic crystal with dielectric rods (epsilon=12),
    the band structure typically has:
    Band 1: Lowest
    Band 2: 
    Band 3:
    Gap 1 (often between 2 and 3 or 3 and 4 depending on density)
    
    If we use a high density of dielectric (rods), the bands shift.
    If we use a low density (small rods), the first gap is small.
    
    A known structure for large higher-order gaps is the "inverted" or "complex" shapes.
    However, a simple approach that often works well for maximizing gaps in the mid-range
    is to use a **square dielectric rod** or a **cross shape** rather than a circle,
    or to use a **hole in a dielectric slab** (if the background is dielectric).
    
    Wait, the problem says "air and dielectric (epsilon 12)".
    Usually, photonic crystals are dielectric rods in air (low epsilon background) 
    or air holes in dielectric (high epsilon background).
    The seed program uses `1 if ... else 0` where 1 is dielectric.
    So it's dielectric rods in air.
    
    For dielectric rods in air (epsilon=12), the TM modes are generally less favorable
    for large gaps than TE modes. The largest TM gap for a square lattice of rods
    is typically the first gap. Higher order gaps are usually smaller.
    
    However, if we invert the lattice (air holes in dielectric, i.e., 0 is dielectric, 1 is air? 
    No, the problem says "1 = dielectric with epsilon 12, 0 = air").
    So we are stuck with dielectric rods in air.
    
    To open a gap between the 4th and 5th bands, we might need to manipulate the band
    dispersion so that bands 4 and 5 are well-separated.
    
    One effective strategy for maximizing specific band gaps is to use **symmetric, non-circular
    shapes** that break the rotational symmetry of the circle while maintaining the square lattice
    symmetry. A **square** dielectric rod often provides better coupling and thus larger gaps
    than a circular one for certain band pairs.
    
    Let's try a **square dielectric rod** centered in the cell.
    The seed uses a circle of radius 0.2.
    Let's try a square of side length ~0.4 (area ~0.16, similar to circle area pi*0.04 ~ 0.125).
    Or slightly larger to increase the average epsilon, which can shift bands.
    
    Another possibility: The "4th and 5th" bands might refer to the bands crossing the X-M line.
    
    Let's try a **square rod** with side 0.45.
    This increases the dielectric filling fraction compared to the circle of r=0.2 (area 0.125 vs 0.2025).
    Higher filling fraction generally opens larger gaps in the first band gap.
    
    If the 4th-5th gap is actually the *second* gap, a higher filling fraction might close the first gap
    and open the second, or shift the bands such that the 4-5 separation increases.
    
    Let's try a square rod.
    """
    size = problem["pixels"]
    # Define a square dielectric region
    # Center is at (15.5, 15.5) for 32x32? No, indices 0..31. Center is 15.5.
    # We want the shape to be symmetric about the center.
    
    # Let's try a square of side 0.5 lattice constants.
    # In pixels: 0.5 * 32 = 16 pixels wide.
    # Centered: from 16 - 8 = 8 to 16 + 8 = 24?
    # Indices 8..23 inclusive is 16 pixels.
    # Let's check symmetry.
    # If we fill pixels where |i - 15.5| < 8 and |j - 15.5| < 8.
    # i ranges from 8 to 23.
    # 15.5 - 8 = 7.5 -> i >= 8
    # 15.5 + 8 = 23.5 -> i <= 23
    # This is symmetric.
    
    # Let's try side length 0.5 (16 pixels).
    half_side = 0.25 * size  # 8 pixels
    
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            # Distance from center in pixels
            dx = abs(i - (size - 1) / 2.0)
            dy = abs(j - (size - 1) / 2.0)
            # Square shape: inside if dx < half_side and dy < half_side
            # But we need to be careful with the exact center.
            # For even size, center is between pixels.
            # dx will be integer + 0.5.
            # If half_side = 8.0, then dx < 8.0 means dx <= 7.5.
            # Max dx for i=15 is 0.5. Min dx for i=8 is 7.5.
            # So i=8..23 are included.
            if dx < half_side and dy < half_side:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
        
    return cell
# EVOLVE-BLOCK-END
