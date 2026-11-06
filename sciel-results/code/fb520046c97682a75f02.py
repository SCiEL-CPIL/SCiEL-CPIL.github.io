    # Final check and scaling
    if best_thicknesses:
        total_t = sum(best_thicknesses)
        if total_t > max_total_nm:
            scale = max_total_nm / total_t
            best_thicknesses = [t * scale for t in best_thicknesses]
            # Recalculate score for safety? The penalty in objective should have prevented this, 
            # but scaling ensures validity.
            best_score = _compute_reflectance(wavelengths, n0, ns, 
                                              [nL if m == "L" else nH for m in best_materials], 
                                              best_thicknesses)
