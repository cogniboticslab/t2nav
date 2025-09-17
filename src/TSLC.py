import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import gudhi

@dataclass
class PersistencePoint:
    """Represents a point in persistence diagram"""
    dimension: int
    birth: float
    death: float
    
    @property
    def persistence(self):
        return self.death - self.birth
    
    @property
    def midpoint(self):
        return (self.birth + self.death) / 2

class TopologicalLoopDetector:
    """
    Training-free loop closure detection using topological signatures.
    Uses persistent homology to identify loops and spatial structures.
    """
    def __init__(self, 
                 max_edge_length: float = 5.0,
                 persistence_threshold: float = 0.1,
                 wasserstein_threshold: float = 2.0,
                 min_loop_size: int = 10):
        """
        Initialize the topological loop detector.
        
        Args:
            max_edge_length: Maximum edge length for Vietoris-Rips complex
            persistence_threshold: Minimum persistence to consider a feature significant
            wasserstein_threshold: Threshold for matching topological signatures
            min_loop_size: Minimum number of nodes to form a valid loop
        """
        self.max_edge_length = max_edge_length
        self.persistence_threshold = persistence_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.min_loop_size = min_loop_size
        
        self.historical_signatures = []
        self.signature_locations = []
        self.loop_closures_detected = []
        
    def extract_trajectory_points(self, trajectory: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Extract 3D points from robot trajectory.
        
        Args:
            trajectory: List of (x, y, theta) poses
            
        Returns:
            Array of 3D points (x, y, z) where z encodes orientation
        """
        points = []
        for x, y, theta in trajectory:
            # Include orientation as third dimension for richer topology
            z = np.sin(theta) * 0.5  # Scale orientation contribution
            points.append([x, y, z])
        return np.array(points)
    
    def build_vietoris_rips_complex(self, points: np.ndarray):
        """
        Build Vietoris-Rips complex from point cloud.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Simplicial complex for persistence computation
        """
        rips_complex = gudhi.RipsComplex(
            points=points,
            max_edge_length=self.max_edge_length
        )
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        return simplex_tree
    

    
    def compute_persistence(self, complex_data) -> List[PersistencePoint]:
        complex_data.compute_persistence()
        persistence = complex_data.persistence()
        
        persistence_points = []
        for dim, (birth, death) in persistence:
            if death - birth > self.persistence_threshold:
                persistence_points.append(
                    PersistencePoint(dim, birth, death)
                )
        return persistence_points
    

    
    def compute_topological_signature(self, 
                                     trajectory: List[Tuple[float, float, float]],
                                     visual_features: Optional[np.ndarray] = None) -> Dict:
        points = self.extract_trajectory_points(trajectory)
        
        if visual_features is not None:
            combined_points = self._combine_spatial_visual(points, visual_features)
        else:
            combined_points = points
        
        complex_data = self.build_vietoris_rips_complex(combined_points)
        persistence_points = self.compute_persistence(complex_data)
        
        signature = {
            'persistence_diagram': persistence_points,
            'betti_numbers': self._compute_betti_numbers(persistence_points),
            'persistence_landscape': self._compute_persistence_landscape(persistence_points),
            'total_persistence': sum(p.persistence for p in persistence_points),
            'max_persistence': max([p.persistence for p in persistence_points], default=0),
            'num_loops': sum(1 for p in persistence_points if p.dimension == 1),
            'trajectory_length': len(trajectory),
            'spatial_extent': self._compute_spatial_extent(points)
        }
        
        return signature
    
    def _combine_spatial_visual(self, 
                                spatial_points: np.ndarray,
                                visual_features: np.ndarray) -> np.ndarray:
        """
        Combine spatial and visual features for richer topology.
        
        Args:
            spatial_points: Nx3 spatial coordinates
            visual_features: NxD visual feature vectors
            
        Returns:
            Combined point cloud
        """
        visual_norm = visual_features / (np.linalg.norm(visual_features, axis=1, keepdims=True) + 1e-8)
        
        visual_weight = 0.3
        visual_scaled = visual_norm * visual_weight
        
        if visual_features.shape[1] > 3:
            u, s, vt = np.linalg.svd(visual_features, full_matrices=False)
            visual_projected = u[:, :3] * s[:3]
            visual_scaled = visual_projected * visual_weight
        
        combined = np.hstack([spatial_points, visual_scaled[:, :min(3, visual_scaled.shape[1])]])
        
        return combined
    
    def _compute_betti_numbers(self, persistence_points: List[PersistencePoint]) -> Dict[int, int]:
        """
        Compute Betti numbers from persistence diagram.
        
        Args:
            persistence_points: List of persistence points
            
        Returns:
            Dictionary mapping dimension to Betti number
        """
        betti = defaultdict(int)
        for point in persistence_points:
            if point.persistence > self.persistence_threshold:
                betti[point.dimension] += 1
        return dict(betti)
    
    def _compute_persistence_landscape(self, 
                                      persistence_points: List[PersistencePoint],
                                      resolution: int = 50) -> np.ndarray:
        """
        Compute persistence landscape for stable vectorization.
        
        Args:
            persistence_points: List of persistence points
            resolution: Number of samples in landscape
            
        Returns:
            Persistence landscape as vector
        """
        if not persistence_points:
            return np.zeros(resolution)
        
        # Sample points along diagonal
        t_values = np.linspace(0, self.max_edge_length, resolution)
        landscape = np.zeros(resolution)
        
        for point in persistence_points:
            if point.dimension == 1:  # Focus on loops
                for i, t in enumerate(t_values):
                    if point.birth <= t <= point.death:
                        height = min(t - point.birth, point.death - t)
                        landscape[i] = max(landscape[i], height)
        
        return landscape
    
    def _compute_spatial_extent(self, points: np.ndarray) -> float:
        """
        Compute spatial extent of trajectory.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Measure of spatial extent
        """
        if len(points) < 2:
            return 0.0
        
        # Compute convex hull volume as measure of extent
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points[:, :2])  # Use 2D projection
            return hull.volume  # Area in 2D
        except:
            bbox_size = points.max(axis=0) - points.min(axis=0)
            return np.prod(bbox_size[:2])
    
    def wasserstein_distance(self, 
                            signature1: Dict,
                            signature2: Dict,
                            p: int = 2) -> float:
        """
        Compute Wasserstein distance between two topological signatures.
        
        Args:
            signature1: First topological signature
            signature2: Second topological signature
            p: Order of Wasserstein distance (1 or 2)
            
        Returns:
            Wasserstein distance
        """
        diagram1 = signature1['persistence_diagram']
        diagram2 = signature2['persistence_diagram']
        
        if not diagram1 and not diagram2:
            return 0.0
        elif not diagram1 or not diagram2:
            return float('inf')
        
        # Separate by dimension
        dim1_diag1 = [(p.birth, p.death) for p in diagram1 if p.dimension == 1]
        dim1_diag2 = [(p.birth, p.death) for p in diagram2 if p.dimension == 1]
        
        distance = self._compute_wasserstein(dim1_diag1, dim1_diag2, p)
        
        landscape_dist = np.linalg.norm(
            signature1['persistence_landscape'] - signature2['persistence_landscape']
        )
        
        # Weighted combination
        return 0.7 * distance + 0.3 * landscape_dist
    
    def _compute_wasserstein(self, 
                           diagram1: List[Tuple[float, float]],
                           diagram2: List[Tuple[float, float]],
                           p: int = 2) -> float:
        """
        Compute Wasserstein distance between persistence diagrams.
        
        Args:
            diagram1: First persistence diagram as list of (birth, death) pairs
            diagram2: Second persistence diagram  
            p: Order of Wasserstein distance
            
        Returns:
            Wasserstein distance
        """
        if not diagram1 and not diagram2:
            return 0.0
        
        # Add diagonal points for optimal matching
        diag1_aug = diagram1 + [(0, 0)] * len(diagram2)
        diag2_aug = diagram2 + [(0, 0)] * len(diagram1)
        
        n = len(diag1_aug)
        cost_matrix = np.zeros((n, n))
        
        for i, (b1, d1) in enumerate(diag1_aug[:len(diagram1)]):
            for j, (b2, d2) in enumerate(diag2_aug[:len(diagram2)]):
                cost_matrix[i, j] = ((b1 - b2)**2 + (d1 - d2)**2) ** (p/2)
            
            for j in range(len(diagram2), n):
                cost_matrix[i, j] = ((d1 - b1) / 2) ** p
        
        for i in range(len(diagram1), n):
            for j, (b2, d2) in enumerate(diag2_aug[:len(diagram2)]):
                cost_matrix[i, j] = ((d2 - b2) / 2) ** p
        
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        return (cost_matrix[row_ind, col_ind].sum()) ** (1/p)
    
    def detect_loop_closure(self,
                           current_trajectory: List[Tuple[float, float, float]],
                           current_visual_features: Optional[np.ndarray] = None,
                           search_radius: float = 10.0) -> Tuple[bool, Optional[int], float]:
        """
        Detect if current trajectory forms a loop with any historical trajectory.
        
        Args:
            current_trajectory: Current robot trajectory
            current_visual_features: Optional visual features
            search_radius: Spatial search radius for candidate loops
            
        Returns:
            Tuple of (loop_detected, matched_index, confidence)
        """
        if len(current_trajectory) < self.min_loop_size:
            return False, None, 0.0
        
        current_signature = self.compute_topological_signature(
            current_trajectory,
            current_visual_features
        )
        
        current_pos = np.array(current_trajectory[-1][:2])
        
        best_match = None
        best_distance = float('inf')
        
        for idx, (hist_signature, hist_location) in enumerate(
            zip(self.historical_signatures, self.signature_locations)
        ):
            hist_pos = np.array(hist_location[:2])
            if np.linalg.norm(current_pos - hist_pos) > search_radius:
                continue
            
            distance = self.wasserstein_distance(current_signature, hist_signature)
            
            if distance < best_distance:
                best_distance = distance
                best_match = idx
        
        loop_detected = best_distance < self.wasserstein_threshold
        
        if loop_detected:
            confidence = np.exp(-best_distance / self.wasserstein_threshold)
        else:
            confidence = 0.0
        
        self.historical_signatures.append(current_signature)
        self.signature_locations.append(current_trajectory[-1])
        
        if loop_detected:
            self.loop_closures_detected.append({
                'current_idx': len(self.historical_signatures) - 1,
                'matched_idx': best_match,
                'distance': best_distance,
                'confidence': confidence
            })
        
        return loop_detected, best_match, confidence
    
    def visualize_persistence_diagram(self, signature: Dict) -> np.ndarray:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        diagram = signature['persistence_diagram']
        if diagram:
            births = [p.birth for p in diagram if p.dimension == 1]
            deaths = [p.death for p in diagram if p.dimension == 1]
            
            ax1.scatter(births, deaths, c='blue', s=50, alpha=0.6, label='1-cycles (loops)')
            
            # Plot diagonal
            max_val = max(deaths + births) if (deaths + births) else 1
            ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
            
            ax1.set_xlabel('Birth')
            ax1.set_ylabel('Death')
            ax1.set_title('Persistence Diagram')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        landscape = signature['persistence_landscape']
        ax2.plot(landscape, 'g-', linewidth=2)
        ax2.fill_between(range(len(landscape)), landscape, alpha=0.3)
        ax2.set_xlabel('Parameter')
        ax2.set_ylabel('Landscape Height')
        ax2.set_title('Persistence Landscape')
        ax2.grid(True, alpha=0.3)
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        
        return img
    
    def get_loop_closure_constraints(self) -> List[Dict]:
        """
        Get loop closure constraints for pose graph optimization.
        
        Returns:
            List of loop closure constraints
        """
        constraints = []
        for closure in self.loop_closures_detected:
            constraints.append({
                'type': 'loop_closure',
                'from_idx': closure['current_idx'],
                'to_idx': closure['matched_idx'],
                'confidence': closure['confidence'],
                'information_matrix': np.eye(3) * closure['confidence']  # Simple weighting
            })
        return constraints
    
# if __name__ == "__main__":
#     # Demo of the topological loop detector
#     print("Topological Loop Closure Detection Demo")
#     print("=" * 50)
    
#     # Create detector
#     detector = TopologicalLoopDetector()
    
#     # Simulate a trajectory that forms a loop
#     trajectory = []
#     for t in np.linspace(0, 2*np.pi, 50):
#         x = 5 * np.cos(t)
#         y = 5 * np.sin(t)
#         theta = t + np.pi/2
#         trajectory.append((x, y, theta))
    
#     # Add some noise to make it realistic
#     trajectory = [(x + np.random.randn()*0.1, 
#                   y + np.random.randn()*0.1, 
#                   theta) for x, y, theta in trajectory]
    
#     # Compute topological signature
#     signature = detector.compute_topological_signature(trajectory)
    
#     print(f"Trajectory points: {len(trajectory)}")
#     print(f"Betti numbers: {signature['betti_numbers']}")
#     print(f"Number of loops detected: {signature['num_loops']}")
#     print(f"Total persistence: {signature['total_persistence']:.3f}")
#     print(f"Spatial extent: {signature['spatial_extent']:.3f}")
    
#     # Test loop closure detection
#     detector.historical_signatures.append(signature)
#     detector.signature_locations.append(trajectory[-1])
    
#     # Create a similar trajectory (should be detected as loop)
#     similar_trajectory = trajectory[:40] + [(x+0.5, y+0.5, theta) for x, y, theta in trajectory[40:]]
    
#     loop_detected, matched_idx, confidence = detector.detect_loop_closure(similar_trajectory)
    
#     print(f"\nLoop closure detection:")
#     print(f"Loop detected: {loop_detected}")
#     print(f"Confidence: {confidence:.3f}")