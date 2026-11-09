    def get_color_bound(available_mask):
        if not available_mask:
            return 0
            
        verts = []
        for i in range(n_valid):
            if (available_mask >> i) & 1:
                verts.append(i)
                
        verts.sort(key=lambda i: degrees[i])
        
        # color_mask[c] is a bitmask of vertices assigned color c
        # We start with color 1
        # We need to assign colors to verts in order
        
        # To avoid reallocating large lists, we can use a dict or list of ints.
        # Max colors is small.
        
        # We will store for each vertex its color in a list `colors` of size n_valid?
        # No, we only care about available vertices.
        
        # Let's use a list of masks for colors.
        # We don't know max color in advance.
        
        # We will maintain a list `color_masks` where index is color-1.
        
        color_masks = []
        
        for v in verts:
            nb_in_gbar = available_mask & ~conflicts[v] & ~(1 << v)
            
            # Find smallest color c such that nb_in_gbar & color_masks[c-1] == 0
            c = 1
            while c <= len(color_masks):
                if nb_in_gbar & color_masks[c-1]:
                    c += 1
                else:
                    break
                    
            if c > len(color_masks):
                color_masks.append(0)
                
            color_masks[c-1] |= (1 << v)
            
        return len(color_masks)
