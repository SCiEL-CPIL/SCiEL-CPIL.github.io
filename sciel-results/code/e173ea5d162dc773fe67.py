"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # Composite score: density plus a small bonus for absolute value.
    # This helps break ties in density by preferring higher-value items.
    density = remaining_values / remaining_weights
    # Small coefficient to slightly favor higher-value items when densities are close
    score = density + 0.05 * remaining_values
    return int(np.argmax(score))
# EVOLVE-BLOCK-END
