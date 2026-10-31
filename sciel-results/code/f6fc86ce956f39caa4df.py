def priority(vector, n):
    """
    Priority based on the magnitude of the sum of additive characters
    with respect to a set of pseudo-random frequency vectors.
    
    Vectors with a small magnitude of the character sum are more 'random-like'
    and thus more likely to be part of a large cap set (low Fourier mass).
    We return a value that is higher for smaller magnitudes.
    """
    freqs = _get_frequencies(n)
    
    # Initialize real and imaginary parts of the sum
    sum_real = 0.0
    sum_imag = 0.0
    
    # Constants for omega = exp(2*pi*i/3)
    # omega^0 = 1 + 0i
    # omega^1 = -0.5 + sqrt(3)/2 i
    # omega^2 = -0.5 - sqrt(3)/2 i
    sqrt3_2 = math.sqrt(3) / 2
    
    for f in freqs:
        # Compute dot product mod 3
        dot = 0
        for i in range(n):
            dot += vector[i] * f[i]
        dot %= 3
        
        if dot == 0:
            sum_real += 1.0
            sum_imag += 0.0
        elif dot == 1:
            sum_real += -0.5
            sum_imag += sqrt3_2
        else: # dot == 2
            sum_real += -0.5
            sum_imag += -sqrt3_2
            
    # Calculate squared magnitude |Z|^2
    mag_sq = sum_real * sum_real + sum_imag * sum_imag
    
    # We want to prioritize vectors with SMALL mag_sq.
    # The maximum possible |Z| is K_FREQS, so max mag_sq is K_FREQS^2.
    # We map mag_sq to a priority where 0 -> high priority, K^2 -> low priority.
    # priority = 1.0 / (1.0 + mag_sq) is a good smooth decreasing function.
    return 1.0 / (1.0 + mag_sq)
