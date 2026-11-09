<<<<<<< SEARCH
    # Phase 6: Very fine polish around the best
    if best_inner is not None and time.time() < deadline - 30:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        
        for delta_t in np.linspace(-0.5, 0.5, 21):
            for delta_p in np.linspace(-0.5, 0.5, 21):
                if time.time() > deadline - 10:
                    break
                try_configuration(centers_base, best_theta + delta_t, best_phi + delta_p)
=======
    # Phase 6: Very fine polish around the best
    if best_inner is not None and time.time() < deadline - 30:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        
        for delta_t in np.linspace(-0.3, 0.3, 31):
            for delta_p in np.linspace(-0.3, 0.3, 31):
                if time.time() > deadline - 10:
                    break
                try_configuration(centers_base, best_theta + delta_t, best_phi + delta_p)
        
        # Phase 7: Final ultra-fine polish
        if best_inner is not None and time.time() < deadline - 15:
            best_theta = best_inner[0, 2]
            best_phi = best_outer[2]
            
            for delta_t in np.linspace(-0.05, 0.05, 11):
                for delta_p in np.linspace(-0.05, 0.05, 11):
                    if time.time() > deadline - 5:
                        break
                    try_configuration(centers_base, best_theta + delta_t, best_phi + delta_p)
>>>>>>> REPLACE
