<<<<<<< SEARCH
def _quad_value(z, nodes, weights, w0, wa):
    """Compute quadrature approximation of integral_0^z dz'/E(z')."""
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_nodes, w0, wa)
    return z * np.sum(weights / E_vals)

=======
def _quad_value(z, nodes, weights, w0, wa):
    """Compute quadrature approximation of integral_0^z dz'/E(z')."""
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_nodes, w0, wa)
    # Safe division
    inv_E = np.zeros_like(E_vals)
    mask = E_vals > 1e-15
    inv_E[mask] = 1.0 / E_vals[mask]
    return z * np.sum(weights * inv_E)

>>>>>>> REPLACE

