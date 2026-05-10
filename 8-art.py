import numpy as np

# Input patterns
inputs = np.array([
    [1,1,0,0],
    [1,0,0,0],
    [0,0,1,1],
    [0,0,1,0]
])

# Vigilance parameter
#if similarity ≥ 0.5 → join cluster else → create new cluster
vigilance = 0.5

# Cluster list
clusters = []

# ART processing
for pattern in inputs:

    assigned = False

    for i in range(len(clusters)):

        # Similarity calculation
        similarity = np.sum(pattern == clusters[i]) / len(pattern)

        # Vigilance test
        if similarity >= vigilance:

            print(pattern, "belongs to Cluster", i+1)

            assigned = True
            break

    # Create new cluster
    if not assigned:

        clusters.append(pattern)

        print(pattern, "creates new Cluster", len(clusters))