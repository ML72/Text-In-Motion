import math

def fill_invalid_regions(regions):
    """
    Given a list of codebook regions, replace any -1 with the last valid region.
    If the sequence starts with -1s, they are left as -1 until a valid region is encountered.
    """
    filled = []
    last_valid = -1
    for r in regions:
        if r != -1:
            last_valid = r
        
        # In our generation logic, the first frame is guaranteed to be != -1
        # So last valid should theoretically never be -1 when we encounter a -1
        filled.append(last_valid)
        
    return filled

def compute_dna(regions):
    """
    Computes the DNA of a generated dance sequence.
    Step 1: ignores subsequences that repeat less than 10 times.
    Step 2: adds region X (f // 30 + ceil(f % 30)) times to DNA 
            (which perfectly simplifies to math.ceil(f / 30.0) for f >= 10).
    """
    if not regions:
        return []
        
    dna = []
    current_region = regions[0]
    count = 1
    
    for r in regions[1:]:
        if r == current_region:
            count += 1
        else:
            if count >= 10 and current_region != -1:
                repeats = math.ceil(count / 30.0)
                dna.extend([current_region] * repeats)
            current_region = r
            count = 1
            
    # Process the last sequence
    if count >= 10 and current_region != -1:
        repeats = math.ceil(count / 30.0)
        dna.extend([current_region] * repeats)
        
    return [int(x) for x in dna]
