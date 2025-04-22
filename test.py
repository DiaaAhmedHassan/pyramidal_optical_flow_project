import numpy as np
import cv2

def PyramidalLK(I1, I2, L, w, K):
    """
    Pyramidal Lucas-Kanade optical flow algorithm
    
    Parameters:
    I1, I2: Input images (grayscale)
    L: Number of pyramid levels
    w: Window size for local flow computation
    K: Number of iterations for refinement at each level
    
    Returns:
    u, v: Flow fields in x and y directions
    """
    
    # Initialize flow fields
    u = np.zeros(I1.shape, dtype=np.float32)
    v = np.zeros(I1.shape, dtype=np.float32)
    
    # Build Gaussian pyramids
    pyramid1 = [I1.copy()]
    pyramid2 = [I2.copy()]
    
    for l in range(1, L+1):
        prev1 = pyramid1[l-1]
        prev2 = pyramid2[l-1]
        pyramid1.append(cv2.pyrDown(prev1))
        pyramid2.append(cv2.pyrDown(prev2))
    
    # Process from coarsest to finest level
    for l in range(L, -1, -1):
        I1_l = pyramid1[l]
        I2_l = pyramid2[l]
        
        # Upsample and scale flow from previous level
        if l < L:
            new_size = (I1_l.shape[1], I1_l.shape[0])
            u = 2 * cv2.resize(u, new_size, interpolation=cv2.INTER_LINEAR)
            v = 2 * cv2.resize(v, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Compute spatial gradients
        Ix = cv2.Sobel(I1_l, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(I1_l, cv2.CV_32F, 0, 1, ksize=3)
        
        height, width = I1_l.shape
        
        # Iterate over each pixel
        for y in range(w//2, height - w//2):
            for x in range(w//2, width - w//2):
                # Initialize local flow
                delta_u = 0.0
                delta_v = 0.0
                
                # Iterative refinement
                for k in range(K):
                    # Warp I2 using current flow
                    x2 = x + u[y, x] + delta_u
                    y2 = y + v[y, x] + delta_v
                    
                    # Skip if outside image
                    if x2 < 0 or x2 >= width or y2 < 0 or y2 >= height:
                        break
                    
                    # Compute temporal difference
                    It = I2_l[int(y2), int(x2)] - I1_l[y, x]
                    
                    # Collect gradients in window
                    window_x = x - w//2
                    window_y = y - w//2
                    
                    Ix_win = Ix[window_y:window_y+w, window_x:window_x+w].flatten()
                    Iy_win = Iy[window_y:window_y+w, window_x:window_x+w].flatten()
                    
                    # Compute ATA and ATb
                    ATA_11 = np.sum(Ix_win * Ix_win)
                    ATA_12 = np.sum(Ix_win * Iy_win)
                    ATA_21 = ATA_12
                    ATA_22 = np.sum(Iy_win * Iy_win)
                    
                    ATb_1 = -np.sum(Ix_win * It)
                    ATb_2 = -np.sum(Iy_win * It)
                    
                    # Solve the system
                    det = ATA_11 * ATA_22 - ATA_12 * ATA_21
                    if det > 1e-6:  # Avoid division by zero
                        delta_du = (ATA_22 * ATb_1 - ATA_12 * ATb_2) / det
                        delta_dv = (-ATA_21 * ATb_1 + ATA_11 * ATb_2) / det
                        
                        # Update local flow
                        delta_u += delta_du
                        delta_v += delta_dv
                
                # Update global flow
                u[y, x] += delta_u
                v[y, x] += delta_v
    
    return u, v