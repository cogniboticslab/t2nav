import numpy as np
import torch
import skimage
import cv2
from typing import Dict, List, Tuple, Optional
import networkx as nx

import warnings
import sys
import os

from src.TeRM import TemporalGraphMemory
from src.TSLC import TopologicalLoopDetector

class EnhancedUniGoalAgent:
    def __init__(self, args, envs):
        from src.agent.unigoal.agent import UniGoal_Agent
        
        self.base_agent = UniGoal_Agent(args, envs)
        
        self.args = args
        self.envs = envs
        
        self.tslc_detector = TopologicalLoopDetector(max_edge_length=5.0,
            persistence_threshold=0.1,
            wasserstein_threshold=2.0,
            min_loop_size=10
        )
        
        self.temporal_memory = TemporalGraphMemory(max_history=100, temporal_decay=0.95)
        
        self.trajectory_history = []
        self.near_goal_trajectory = []
        self.loop_check_counter = 0
        self.counterfactual_targets = []
        self.stuck_detection_counter = 0
    
    def reset(self):
        obs, rgbd, infos = self.base_agent.reset()
        self.trajectory_history = []
        self.near_goal_trajectory = []
        self.loop_check_counter = 0
        self.counterfactual_targets = []
        return obs, rgbd, infos
    
    def get_planner_inputs(self, agent_input):
        planner_inputs = self.base_agent.get_planner_inputs(agent_input)
        
        # Extract current pose for tracking
        if 'pose_pred' in planner_inputs:
            current_pose = tuple(planner_inputs['pose_pred'][:3])  # x, y, theta
            self.trajectory_history.append(current_pose)
            
            # TSLC
            self.loop_check_counter += 1
            if self.loop_check_counter >= 10 and len(self.trajectory_history) >= self.tslc_detector.min_loop_size:
                try:
                    loop_detected, matched_idx, confidence = self.tslc_detector.detect_loop_closure(
                        self.trajectory_history[-30:]
                    )
                    
                    if loop_detected and confidence > 0.7:
                        print(f"[TSLC] Loop detected with confidence {confidence:.2f}")
                        planner_inputs['loop_detected'] = True
                        planner_inputs['loop_confidence'] = confidence
                        
                        if planner_inputs.get('found_goal', 0) == 0:
                            self._modify_exploration_for_loop(planner_inputs)
                except Exception as e:
                    print(f"[TSLC] Warning: Loop detection failed: {e}")
                
                self.loop_check_counter = 0
            
            # Stuck detection using topology
            if self._detect_stuck_with_topology():
                print("Stuck detected via topological analysis")
                self.base_agent.been_stuck = True
                planner_inputs['stuck'] = True
        
        # Near-goal trajectory tracking for better verification
        if planner_inputs.get('found_goal', 0) == 0:
            goal_confidence = self._estimate_goal_proximity(planner_inputs)
            
            if goal_confidence > 0.3:
                if 'pose_pred' in planner_inputs:
                    current_pose = tuple(planner_inputs['pose_pred'][:3])
                    self.near_goal_trajectory.append(current_pose)
                
                if len(self.near_goal_trajectory) > 15:
                    try:
                        goal_sig = self.tslc_detector.compute_topological_signature(
                            self.near_goal_trajectory[-15:]
                        )
                        
                        if goal_sig['num_loops'] > 0:
                            print("Circling near goal detected - may need threshold adjustment")
                            planner_inputs['circling_goal'] = True
                    except Exception as e:
                        print(f"Warning: Goal circulation check failed: {e}")
        
        if len(self.near_goal_trajectory) > 0 and planner_inputs.get('found_goal', 0) == 0:
            if 'pose_pred' in planner_inputs:
                current_pos = planner_inputs['pose_pred'][:2]
                if self.near_goal_trajectory:
                    last_near_goal_pos = self.near_goal_trajectory[-1][:2]
                    distance_from_last = np.linalg.norm(
                        np.array(current_pos) - np.array(last_near_goal_pos)
                    )
                    
                    if distance_from_last > 200:
                        self.near_goal_trajectory = []
                        print(f"Cleared near-goal trajectory - moved {distance_from_last:.1f} away")
        
        # Counterfactual exploration targets
        if planner_inputs.get('found_goal', 0) == 0 and not planner_inputs.get('stuck', False):
            if len(self.trajectory_history) > 20 and len(self.counterfactual_targets) == 0:
                # Generate counterfactual hypotheses periodically
                self._generate_counterfactual_targets(planner_inputs)
        return planner_inputs
    
    def step(self, agent_input):
        enhanced_input = self._enhance_agent_input(agent_input)
        obs, rgbd, done, infos = self.base_agent.step(enhanced_input)
        
        if not done:
            try:
                if hasattr(self.base_agent, 'graph'):
                    scene_graph_dict = self._convert_graph_to_dict(self.base_agent.graph)
                    self.temporal_memory.add_snapshot(scene_graph_dict)
            except Exception as e:
                print(f"Warning: Temporal memory update failed: {e}")
        
        return obs, rgbd, done, infos
    
    def get_short_term_goal(self, agent_input):
        return self.base_agent.get_short_term_goal(agent_input)
    
    def _enhance_agent_input(self, agent_input):
        enhanced = agent_input.copy()

        if 'pose_pred' in enhanced:
            current_pose = tuple(enhanced['pose_pred'][:3])
            
            if len(self.trajectory_history) > 10:
                for i, past_pose in enumerate(self.trajectory_history[:-10]):
                    dist = np.linalg.norm(np.array(current_pose[:2]) - np.array(past_pose[:2]))
                    if dist < 50:  # Within 50 units
                        enhanced['potential_loop'] = True
                        break
        
        return enhanced
    
    def _modify_exploration_for_loop(self, planner_inputs):
        if 'exp_goal' in planner_inputs and 'pose_pred' in planner_inputs:
            current_pos = planner_inputs['pose_pred'][:2]
            
            exp_goal = planner_inputs['exp_goal']
            if isinstance(exp_goal, np.ndarray):
                h, w = exp_goal.shape
                for i in range(h):
                    for j in range(w):
                        dist = np.sqrt((i - current_pos[0])**2 + (j - current_pos[1])**2)
                        if dist < 30:
                            exp_goal[i, j] *= 0.3
    
    def _detect_stuck_with_topology(self) -> bool:
        if len(self.trajectory_history) < 20:
            return False
        
        try:
            recent_trajectory = self.trajectory_history[-20:]
            recent_signature = self.tslc_detector.compute_topological_signature(recent_trajectory)
            
            spatial_extent = recent_signature.get('spatial_extent', 1)
            trajectory_length = recent_signature.get('trajectory_length', 0)
            
            if spatial_extent < 10 and trajectory_length > 15:
                stuck_ratio = trajectory_length / (spatial_extent + 0.1)
                if stuck_ratio > 50:
                    self.stuck_detection_counter += 1
                    return self.stuck_detection_counter > 3
            else:
                self.stuck_detection_counter = 0
        except Exception as e:
            print(f"Warning: Stuck detection failed: {e}")
        
        return False
    
    def _estimate_goal_proximity(self, planner_inputs) -> float:
        confidence = 0.0
        
        if hasattr(self.base_agent, 'graph'):
            try:
                if hasattr(self.base_agent.graph, 'scene_graph'):
                    scene_nodes = self.base_agent.graph.scene_graph.nodes()
                    # Simple check for goal-related labels
                    for node in scene_nodes:
                        node_data = self.base_agent.graph.scene_graph.nodes[node]
                        if 'goal' in str(node_data.get('label', '')).lower():
                            confidence = 0.5
                            break
            except:
                pass
        
        return confidence
    
    def _generate_counterfactual_targets(self, planner_inputs):
        try:
            frontier_nodes = self._identify_frontiers(planner_inputs)
            
            if frontier_nodes and hasattr(self.base_agent, 'graph'):
                scene_dict = self._convert_graph_to_dict(self.base_agent.graph)

                for frontier in frontier_nodes[:3]:  # Limit to top 3
                    hypothesis = {
                        'position': frontier['position'],
                        'type': 'frontier_extension',
                        'confidence': 0.5
                    }
                    self.counterfactual_targets.append(hypothesis)
                
                if self.counterfactual_targets:
                    print(f"Generated {len(self.counterfactual_targets)} counterfactual targets")
        except Exception as e:
            print(f"Warning: Counterfactual generation failed: {e}")
    
    def _identify_frontiers(self, planner_inputs) -> List[Dict]:
        frontiers = []
        
        try:
            if 'exp_pred' in planner_inputs and 'map_pred' in planner_inputs:
                exp_map = planner_inputs['exp_pred']
                map_pred = planner_inputs['map_pred']
                
                if isinstance(exp_map, np.ndarray) and isinstance(map_pred, np.ndarray):
                    h, w = exp_map.shape[:2]
                    for i in range(0, h, 10):
                        for j in range(0, w, 10):
                            if exp_map[i, j] > 0 and map_pred[i, j] == 0:
                                # This is a frontier
                                frontiers.append({
                                    'position': [i, j, 0],
                                    'type': 'frontier'
                                })
                                
                                if len(frontiers) >= 10:
                                    break
                        if len(frontiers) >= 10:
                            break
        except Exception as e:
            print(f"Warning: Frontier identification failed: {e}")
        
        return frontiers
    
    def _convert_graph_to_dict(self, graph) -> Dict:
        try:
            if hasattr(graph, 'scene_graph'):
                nodes = []
                for n in graph.scene_graph.nodes():
                    node_data = graph.scene_graph.nodes[n]
                    nodes.append({'label': n, **node_data})
                
                edges = []
                for e in graph.scene_graph.edges():
                    edges.append({'source': e[0], 'target': e[1]})
                
                return {'nodes': nodes, 'edges': edges}
        except:
            pass
        
        return {'nodes': [], 'edges': []}
    
    def __getattr__(self, name):
        return getattr(self.base_agent, name)