# EVOLVE-BLOCK-START
import numpy as np


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    Uses a non-linear "fan" construction where base positions are chosen such
    that the tube trajectories spread apart at c=0 and converge at c=1.
    This creates a funnel-like geometry that reduces the union volume compared
    to the linear star construction.

    The base position for tube (i, j) is given by:
        x_k = -i / N * (1 - i/(2*N))
        y_k = -j / N * (1 - j/(2*N))

    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner, i.e. numpy meshgrid indexing="ij" then flatten),
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    N = 8
    # Generate grid indices i and j
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    # Non-linear fan construction
    # At c=0: x = -i/N * (1 - i/(2N)), y = -j/N * (1 - j/(2N))
    # At c=1: x = -i/N * (1 - i/(2N)) + i/N = i/N * i/(2N) = i^2/(2N^2)
    #          y = -j/N * (1 - j/(2N)) + j/N = j^2/(2N^2)
    # This creates asymmetric spreading
    base_x = -i_flat / N * (1.0 - i_flat / (2.0 * N))
    base_y = -j_flat / N * (1.0 - j_flat / (2.0 * N))

    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
