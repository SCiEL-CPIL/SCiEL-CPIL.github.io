def construct(k, g):
    # For k=5, g=9, use Boutin's construction with q=4, d=4
    if k == 5 and g == 9:
        q = 4
        d = 4
        # Boutin's construction
        # Number of vertices per side: S = 1 + q + q^2 + q^3 + q^4
        S = 1
        for i in range(1, d+1):
            S += q**i
        n = 2 * S
        edges = []
        # Implementation of Boutin's edges
        # This requires the specific maps.
        # If I cannot implement them, I return LUW.
        pass
