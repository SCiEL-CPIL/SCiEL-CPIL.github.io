import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 4-3-4 Hexagon Packing
    # Unit hexagon side = 1
    # Width w = sqrt(3), Height h = 2
    # Row vertical spacing = 1.5
    # Horizontal offset for staggered rows = w/2
    
    w = np.sqrt(3)
    
    # Define rows
    # Row 1 (Top): 4 hexes, y = 1.5
    # Row 2 (Mid): 3 hexes, y = 0.0
    # Row 3 (Bot): 4 hexes, y = -1.5
    
    # To center the cluster:
    # Row 1 (4 hexes, aligned): x = -1.5w, -0.5w, 0.5w, 1.5w
    # Row 2 (3 hexes, offset):  x = -w, 0, w
    # Row 3 (4 hexes, aligned): x = -1.5w, -0.5w, 0.5w, 1.5w
    
    inner = []
    
    # Row 1
    for x in [-1.5*w, -0.5*w, 0.5*w, 1.5*w]:
        inner.append([x, 1.5, 0.0])
        
    # Row 2
    for x in [-w, 0.0, w]:
        inner.append([x, 0.0, 0.0])
        
    # Row 3
    for x in [-1.5*w, -0.5*w, 0.5*w, 1.5*w]:
        inner.append([x, -1.5, 0.0])
        
    # Outer hexagon
    # Bounding box of cluster:
    # X range: -1.5w - w/2 to 1.5w + w/2 => -2w to 2w => Width 4w?
    # Wait, leftmost hex center -1.5w. Left edge -1.5w - w/2 = -2w.
    # Rightmost hex center 1.5w. Right edge 1.5w + w/2 = 2w.
    # Total Width 4w = 4*sqrt(3) approx 6.928.
    # Y range: 1.5+1=2.5 to -1.5-1=-2.5. Height 5.
    
    # Container Side S
    # If aligned flat-top:
    # S*sqrt(3) >= 4*sqrt(3) => S >= 4.
    # 2*S >= 5 => S >= 2.5.
    # So S=4.
    
    # Can we do better?
    # The 4-3-4 packing is wide.
    # The 2-3-3-3 packing was narrower (Width 2.5w approx 4.33) but taller (6.5).
    # For 2-3-3-3:
    # S*sqrt(3) >= 2.5*sqrt(3) => S >= 2.5.
    # 2*S >= 6.5 => S >= 3.25.
    # So S=3.25 is better than S=4.
    
    # Let's implement 2-3-3-3 packing.
    inner = []
    
    # Row 1 (Top, 2 hexes, Offset): y = 2.25
    # x = -0.5w, 0.5w
    for x in [-0.5*w, 0.5*w]:
        inner.append([x, 2.25, 0.0])
        
    # Row 2 (3 hexes, Aligned): y = 0.75
    # x = -w, 0, w
    for x in [-w, 0.0, w]:
        inner.append([x, 0.75, 0.0])
        
    # Row 3 (3 hexes, Offset): y = -0.75
    # To minimize width, pick 3 from offset positions.
    # Offset positions: -1.5w, -0.5w, 0.5w, 1.5w
    # Pick -0.5w, 0.5w, 1.5w? Or -1.5w, -0.5w, 0.5w?
    # Let's pick -0.5w, 0.5w, 1.5w to balance?
    # Actually, let's center the whole thing.
    # If Row 2 is centered at 0, and Row 1 is centered at 0.
    # Row 3 (Offset) should be centered at 0?
    # Offset row of 3 cannot be centered at 0 with integer half-steps.
    # Positions: -1.5w, -0.5w, 0.5w, 1.5w.
    # If we pick -1.5w, -0.5w, 0.5w, center is -w/3?
    # Let's just pick -0.5w, 0.5w, 1.5w.
    for x in [-0.5*w, 0.5*w, 1.5*w]:
        inner.append([x, -0.75, 0.0])
        
    # Row 4 (3 hexes, Aligned): y = -2.25
    # x = -w, 0, w
    for x in [-w, 0.0, w]:
        inner.append([x, -2.25, 0.0])
        
    # Calculate Bounding Box
    # X min: -w (from Row 2/4) or -0.5w (Row 1/3)?
    # Row 2/4 min x: -w. Left edge: -w - w/2 = -1.5w.
    # Row 1/3 min x: -0.5w. Left edge: -0.5w - w/2 = -w.
    # So Min X Edge = -1.5w.
    # X max: Row 2/4 max x: w. Right edge: w + w/2 = 1.5w.
    # Row 3 max x: 1.5w. Right edge: 1.5w + w/2 = 2w.
    # So Max X Edge = 2w.
    # Total Width = 3.5w = 3.5*sqrt(3) approx 6.06.
    
    # Y min: -2.25 - 1 = -3.25.
    # Y max: 2.25 + 1 = 3.25.
    # Total Height = 6.5.
    
    # Container S
    # S*sqrt(3) >= 3.5*sqrt(3) => S >= 3.5.
    # 2*S >= 6.5 => S >= 3.25.
    # So S = 3.5.
    
    return inner, (0.0, 0.0), 3.5, 0.0
# EVOLVE-BLOCK-END
