<<<<<<< SEARCH
    # Stage 9: Final aggressive radius expansion
    if time.time() < deadline - 0.5:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out, max_iterations=200)
=======
    # Stage 9: Final aggressive radius expansion using LP
    if time.time() < deadline - 0.5:
        pos_out, radii_out = _try_radius_expansion(pos_out, radii_out)
        pos_out, radii_out = _post_process(pos_out, radii_out)
>>>>>>> REPLACE

