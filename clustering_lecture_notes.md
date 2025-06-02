# Clustering: Complete Lecture Notes

## 1. What is Clustering and When to Use Clustering?

### What is Clustering?
Clustering is an unsupervised machine learning technique that groups similar data points together based on their features. Unlike supervised learning, clustering doesn't require labeled data - it discovers hidden patterns and structures within the data.

**Key Characteristics:**
- **Unsupervised Learning**: No target variable or labels needed
- **Pattern Discovery**: Finds natural groupings in data
- **Similarity-based**: Groups data points with similar characteristics

### When to Use Clustering?

**Common Applications:**
1. **Customer Segmentation**: Group customers based on purchasing behavior, demographics
2. **Market Research**: Identify distinct market segments
3. **Image Segmentation**: Separate different regions in images
4. **Gene Sequencing**: Group genes with similar expression patterns
5. **Social Network Analysis**: Identify communities within networks
6. **Anomaly Detection**: Identify outliers that don't belong to any cluster
7. **Data Preprocessing**: Reduce complexity before applying other algorithms

**Decision Criteria for Using Clustering:**
- When you want to explore data structure without prior knowledge
- When you need to reduce data complexity
- When looking for natural groupings or segments
- When preparing data for further analysis

## 2. K-Means Clustering: The Workhorse of Unsupervised Learning

### Overview
K-Means is one of the most popular clustering algorithms due to its simplicity, efficiency, and effectiveness. It partitions data into k clusters where k is specified beforehand.

### Key Concepts
- **Centroids**: The center point of each cluster
- **Inertia**: Sum of squared distances from points to their cluster centroids
- **Convergence**: When centroids stop moving significantly between iterations

### Assumptions and Limitations
**Assumptions:**
- Clusters are spherical (roughly circular)
- Clusters have similar sizes
- Clusters have similar densities

**Limitations:**
- Need to specify k in advance
- Sensitive to initialization
- Struggles with non-spherical clusters
- Affected by outliers

## 3. The K-Means Algorithm

### Step-by-Step Process

1. **Initialize**: Choose k and randomly place k centroids
2. **Assign**: Assign each data point to the nearest centroid
3. **Update**: Move each centroid to the center of its assigned points
4. **Repeat**: Continue steps 2-3 until convergence

### Detailed Algorithm
```
1. Choose number of clusters k
2. Initialize k centroids randomly
3. Repeat until convergence:
   a. For each data point:
      - Calculate distance to all centroids
      - Assign point to closest centroid
   b. For each cluster:
      - Calculate new centroid as mean of assigned points
   c. Check for convergence (centroids stop moving)
```

### Distance Metrics
**Euclidean Distance** (most common):
- Formula: √[(x₁-x₂)² + (y₁-y₂)²]
- Works well for continuous features

**Other Options:**
- Manhattan Distance: |x₁-x₂| + |y₁-y₂|
- Cosine Distance: For high-dimensional data

## 4. Choosing the Right Number of Clusters

### The Elbow Method
Plot the Within-Cluster Sum of Squares (WCSS) against different values of k. Look for the "elbow" where the rate of decrease sharply changes.

**Steps:**
1. Run K-Means for k = 1, 2, 3, ..., n
2. Calculate WCSS for each k
3. Plot k vs WCSS
4. Find the elbow point

### Silhouette Analysis
Measures how similar a point is to its own cluster compared to other clusters.
- **Range**: -1 to 1
- **Good clustering**: Average silhouette score close to 1
- **Poor clustering**: Scores close to 0 or negative

### Gap Statistic
Compares the total intracluster variation for different values of k with their expected values under null reference distribution.

## 5. Mathematical Intuition of K-Means Clustering

### Objective Function
K-Means minimizes the Within-Cluster Sum of Squares (WCSS):

**WCSS = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²**

Where:
- k = number of clusters
- Cᵢ = cluster i
- x = data point
- μᵢ = centroid of cluster i

### Mathematical Steps

**Assignment Step:**
For each point x, assign to cluster j where:
j = argmin ||x - μⱼ||²

**Update Step:**
Update each centroid μⱼ:
μⱼ = (1/|Cⱼ|) Σₓ∈Cⱼ x

### Convergence Properties
- K-Means always converges (WCSS decreases monotonically)
- May converge to local minimum
- Multiple random initializations help find global minimum

## 6. Trade-offs for Different Values of K

### Under-clustering (k too small):
- **Pros**: Simple interpretation, robust to noise
- **Cons**: May miss important subgroups, high intra-cluster variance

### Over-clustering (k too large):
- **Pros**: Low intra-cluster variance, detailed segmentation
- **Cons**: Overfitting, difficult interpretation, unstable clusters

### Optimal k Considerations:
- **Business Context**: Domain knowledge about expected groups
- **Statistical Measures**: Elbow method, silhouette analysis
- **Interpretability**: Can you explain and act on the clusters?
- **Stability**: Do clusters remain consistent across different runs?

## 7. Hierarchical Clustering: Understanding Data at Different Levels

### Overview
Hierarchical clustering creates a hierarchy of clusters, showing relationships at multiple levels of granularity. Unlike K-Means, you don't need to specify the number of clusters beforehand.

### Types of Hierarchical Clustering

**Agglomerative (Bottom-up):**
- Start with each point as its own cluster
- Merge closest clusters until one cluster remains
- Most common approach

**Divisive (Top-down):**
- Start with all points in one cluster
- Split clusters until each point is separate
- Less common, computationally intensive

### Advantages over K-Means
- No need to specify k in advance
- Provides hierarchy of clusters
- Works with any distance measure
- Deterministic results (no random initialization)

## 8. Mathematics and Intuition behind Hierarchical Clustering

### Distance Measures Between Clusters

**Single Linkage:**
- Distance = minimum distance between any two points in different clusters
- Creates elongated clusters
- Formula: d(A,B) = min{d(a,b) : a∈A, b∈B}

**Complete Linkage:**
- Distance = maximum distance between any two points in different clusters
- Creates compact, spherical clusters
- Formula: d(A,B) = max{d(a,b) : a∈A, b∈B}

**Average Linkage:**
- Distance = average distance between all pairs of points in different clusters
- Balance between single and complete linkage
- Formula: d(A,B) = (1/|A||B|) Σₐ∈ₐ Σᵦ∈ᵦ d(a,b)

**Ward's Method:**
- Minimizes within-cluster variance when merging
- Similar to K-Means objective
- Most commonly used for continuous data

### Dendrogram Interpretation
- **Height**: Indicates dissimilarity at which clusters merge
- **Horizontal cuts**: Different numbers of clusters
- **Vertical distance**: Cluster separation quality

### Algorithm Complexity
- **Time Complexity**: O(n³) for basic implementation
- **Space Complexity**: O(n²) for distance matrix
- **Scalability**: Limited to smaller datasets (< 10,000 points)

## 9. Comparison: K-Means vs Hierarchical Clustering

| Aspect | K-Means | Hierarchical |
|--------|---------|-------------|
| **Cluster Shape** | Spherical | Any shape |
| **Number of Clusters** | Must specify k | Discover from dendrogram |
| **Scalability** | O(nkt) - very scalable | O(n³) - limited |
| **Deterministic** | No (random init) | Yes |
| **Outlier Sensitivity** | High | Medium |
| **Interpretability** | Good for k clusters | Excellent (hierarchy) |
| **Memory Usage** | O(n) | O(n²) |

## 10. Best Practices and Tips

### Data Preprocessing
1. **Scale Features**: Use StandardScaler or MinMaxScaler
2. **Handle Outliers**: Remove or cap extreme values
3. **Feature Selection**: Remove irrelevant features
4. **Handle Missing Values**: Impute or remove

### Algorithm Selection Guidelines
- **Use K-Means when**: Large datasets, spherical clusters expected, k is known
- **Use Hierarchical when**: Small datasets, cluster hierarchy needed, k is unknown

### Validation Techniques
1. **Internal Validation**: Silhouette score, Davies-Bouldin index
2. **External Validation**: If true labels available (Adjusted Rand Index)
3. **Visual Inspection**: Scatter plots, dendrograms
4. **Domain Validation**: Do clusters make business sense?

### Common Pitfalls
- Not scaling features appropriately
- Choosing k without proper analysis
- Ignoring cluster interpretability
- Not validating results with domain knowledge
- Using clustering when supervised learning is more appropriate