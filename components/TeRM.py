import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import deque

class TemporalGraphMemory:
    """
    Temporal Graph Memory Networks - Graphs that evolve over time
    Maintains historical graph states and temporal edges
    """
    def __init__(self, max_history: int = 100, temporal_decay: float = 0.95):
        self.graph_snapshots = deque(maxlen=max_history)
        self.temporal_edges = []
        self.temporal_decay = temporal_decay
        self.time_step = 0
        
    def add_snapshot(self, graph: Dict, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = self.time_step
            
        snapshot = {
            'graph': graph.copy(),
            'timestamp': timestamp,
            'nodes': list(graph.get('nodes', [])),
            'edges': list(graph.get('edges', []))
        }
        self.graph_snapshots.append(snapshot)
        self.time_step += 1
        
        # Create temporal edges to previous snapshots
        if len(self.graph_snapshots) > 1:
            self._create_temporal_edges()
    
    def _create_temporal_edges(self):
        current = self.graph_snapshots[-1]
        previous = self.graph_snapshots[-2]
        
        for curr_node in current['nodes']:
            for prev_node in previous['nodes']:
                if self._nodes_similar(curr_node, prev_node):
                    delta_t = current['timestamp'] - previous['timestamp']
                    appearance_change = self._compute_appearance_delta(curr_node, prev_node)
                    temporal_stability = np.exp(-appearance_change * delta_t)
                    
                    edge = {
                        'type': 'temporal',
                        'from': (previous['timestamp'], prev_node),
                        'to': (current['timestamp'], curr_node),
                        'weight': temporal_stability,
                        'delta_t': delta_t
                    }
                    self.temporal_edges.append(edge)
    
    def _nodes_similar(self, node1: Dict, node2: Dict) -> bool:
        if node1.get('label') == node2.get('label'):
            if 'position' in node1 and 'position' in node2:
                dist = np.linalg.norm(
                    np.array(node1['position']) - np.array(node2['position'])
                )
                return dist < 2.0  # Within 2 meters
            return True
        return False
    
    def _compute_appearance_delta(self, node1: Dict, node2: Dict) -> float:
        if 'color_hist' in node1 and 'color_hist' in node2:
            return np.sum(np.abs(node1['color_hist'] - node2['color_hist']))
        return 0.1
    
    def predict_future_state(self, steps_ahead: int = 1) -> Dict:
        if len(self.graph_snapshots) < 2:
            return self.graph_snapshots[-1]['graph'] if self.graph_snapshots else {}
        
        velocities = self._extract_velocities()
        current = self.graph_snapshots[-1]
        
        predicted = {
            'nodes': [],
            'edges': []
        }
        
        for node in current['nodes']:
            predicted_node = node.copy()
            if node['label'] in velocities:
                # Apply velocity to predict position
                vel = velocities[node['label']]
                if 'position' in predicted_node:
                    predicted_node['position'] = (
                        np.array(predicted_node['position']) + vel * steps_ahead
                    ).tolist()
            predicted['nodes'].append(predicted_node)
        
        return predicted
    
    def _extract_velocities(self) -> Dict:
        velocities = {}
        for edge in self.temporal_edges[-10:]:
            if edge['delta_t'] > 0:
                from_node = edge['from'][1]
                to_node = edge['to'][1]
                if 'position' in from_node and 'position' in to_node:
                    vel = (np.array(to_node['position']) - 
                          np.array(from_node['position'])) / edge['delta_t']
                    velocities[to_node['label']] = vel
        return velocities
    
    def get_temporal_context(self, node_label: str, time_window: int = 5) -> List[Dict]:
        context = []
        for snapshot in list(self.graph_snapshots)[-time_window:]:
            for node in snapshot['nodes']:
                if node.get('label') == node_label:
                    context.append({
                        'timestamp': snapshot['timestamp'],
                        'node': node
                    })
        return context