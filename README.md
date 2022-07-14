
[![image info](./doc/_static/tractosearch_big_logo.svg)](https://github.com/StongeEtienne/tractosearch/)

[![image info](https://img.shields.io/pypi/v/tractosearch.svg)](https://pypi.python.org/pypi/tractosearch)
[![image info](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](https://github.com/StongeEtienne/tractosearch/blob/master/LICENSE)

## TractoSearch: Fast Tractography Streamline Search
A python package to efficiently search streamlines inside tractograms or Atlases.  
It generalizes k-nearest neighbors and radius search to tractography streamlines.  
Using a space-partitioning tree (i.e. k-d tree from [LpqTree](https://github.com/StongeEtienne/lpqtree)).

### Applications
TractoSearch can:
- Find all similar streamlines
- Compute a sparse adjacency matrix with streamlines
- Search for k-nearest-neighbors
- Employ any Minkowski mixed-norm (Lpq), but optimized for L21

### Installation
```
pip install tractosearch
```

### Scripts Usage
Cluster a full brain tractogram using an atlas:
```
tractosearch_nearest_in_radius.py  sub00_tracto.trk  atlas/*.trk  4.0  result/
```
similar python scripts `tractosearch_all_in_radius.py` / `tractosearch_nearest.py`

### Python Usage
```python
import numpy as np
from tractosearch.search import knn_search, radius_search

# Create 3 streamlines composed of four 2D points
slines_a = [np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])]

# Create 2 streamlines composed of four 2D points
slines_b = [np.array([[0.1, 0.1], [1.5, 0.8], [1.8, 1.0], [2.5, 2.0]]),
            np.array([[3.0, 2.2], [2.0, 1.0], [1.0, 0.0], [0.0, 1.0]])]

# For each streamlines in "a" find the nearest in "b"
nn_ids_b, nn_dist = knn_search(slines_a, slines_b, k=1, resample=4)
# nn_ids_a = np.arange(len(slines_a))

# For each streamlines in "a" all element in "b" in radius
coo_mtx  = radius_search(slines_a, slines_b, 2.5, resample=4)
ids_a = coo_mtx.row; ids_b = coo_mtx.col; rdist = coo_mtx.data
```

### Reference
```
[StOnge2022]
  Fast Streamline Search: An Exact Technique for Diffusion MRI Tractography.
  St-Onge, E., Garyfallidis, E. and Collins, D.L.
  Neuroinformatics, pp. 1-12. Springer, 2022.
```
