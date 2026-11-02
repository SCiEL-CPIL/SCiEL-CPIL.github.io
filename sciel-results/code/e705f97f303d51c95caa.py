"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function with N=400 intervals on [-1/4, 1/4].
    
    The function uses a binary sign vector derived from the zero crossings of a
    low-frequency sine wave. Specifically, we use a sine wave with frequency k=1
    (one full period over the domain) to determine the sign at each interval.
    
    The intervals are equally spaced over [-1/4, 1/4]. We compute the midpoint
    of each interval and evaluate sin(2*pi*x) at those points. The sign of this
    value determines the height (+1 or -1) for that interval.
    
    This creates a smooth transition between positive and negative blocks,
    which should minimize high-frequency energy in the autoconvolution.
    """
    # Define the domain [-1/4, 1/4]
    x_min = -0.25
    x_max = 0.25
    
    # Create N equal intervals
    # Midpoints of each interval
    boundaries = np.linspace(x_min, x_max, N + 1)
    midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])
    
    # Evaluate sine wave: sin(2*pi*x) has one full period over [-1/4, 1/4]
    # since the domain length is 1/2, and 2*pi*(1/2) = pi... wait, let me reconsider.
    # Domain length = 1/2. For one full period, we need sin(2*pi*x / L) where L = 1/2.
    # So sin(2*pi*x / (1/2)) = sin(4*pi*x).
    # Let's verify: at x = -1/4, sin(4*pi*(-1/4)) = sin(-pi) = 0
    # at x = 0, sin(0) = 0
    # at x = 1/4, sin(4*pi*(1/4)) = sin(pi) = 0
    # This gives zeros at -1/4, 0, 1/4, creating two half-periods.
    # Actually, sin(4*pi*x) over [-1/4, 1/4]:
    # x=-1/4: sin(-pi)=0, x=-1/8: sin(-pi/2)=-1, x=0: sin(0)=0, 
    # x=1/8: sin(pi/2)=1, x=1/4: sin(pi)=0
    # So it goes from 0 down to -1, back to 0, up to 1, back to 0.
    # This means the sign is negative on (-1/4, 0) and positive on (0, 1/4).
    # This is essentially a step function that's -1 on the left half and +1 on the right half.
    
    # For a smoother transition, let's try a lower frequency or different phase.
    # Let's use sin(2*pi*x) which has period 1. Over [-1/4, 1/4]:
    # x=-1/4: sin(-pi/2)=-1, x=0: sin(0)=0, x=1/4: sin(pi/2)=1
    # This gives a linear-ish transition from -1 to 1.
    
    # Let's use sin(2*pi*x) for a gentle transition
    values = np.sin(2 * np.pi * midpoints)
    
    # Convert to binary sign vector: +1 where sin >= 0, -1 where sin < 0
    signs = np.where(values >= 0, 1.0, -1.0)
    
    return signs
# EVOLVE-BLOCK-END
