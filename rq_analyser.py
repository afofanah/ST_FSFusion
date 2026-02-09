import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os
import seaborn as sns

class ResearchQuestionAnalyzer:
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = log_dir
        self.results_dir = os.path.join(log_dir, 'research_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.eval_config = config.get('evaluation', {})
        self.enable_rq1 = self.eval_config.get('enable_rq1', True)
        self.enable_rq2 = self.eval_config.get('enable_rq2', True)
        self.enable_rq3 = self.eval_config.get('enable_rq3', True)
        self.enable_rq4 = self.eval_config.get('enable_rq4', True)
        self.enable_rq5 = self.eval_config.get('enable_rq5', True)
        
    def analyze_spatial_dependency(self, source_model, target_model, source_data, target_data):
        if not self.enable_rq1:
            return {}
            
        print("Analyzing RQ1: Spatial Dependency Preservation...")
        
        def extract_spatial_features(model, data):
            model.eval()
            with torch.no_grad():
                data = data.to(next(model.parameters()).device)
                
                if hasattr(model, 'spatial_encoder') and hasattr(data, 'edge_index'):
                    batch_size = data.x.size(0) if data.x.dim() > 2 else 1
                    node_num = data.node_num if hasattr(data, 'node_num') else data.x.size(1) if data.x.dim() > 2 else data.x.size(0)
                    
                    spatial_features = []
                    edge_index = data.edge_index
                    input_data = data.x
                    
                    for b in range(batch_size):
                        batch_feats = input_data[b] if input_data.dim() > 2 else input_data
                        x = batch_feats
                        for layer in model.spatial_encoder:
                            if hasattr(layer, 'forward'):
                                x = layer(x, edge_index)
                            else:
                                x = layer(x)
                        spatial_features.append(x)
                    
                    if len(spatial_features) > 0:
                        return torch.stack(spatial_features).cpu().numpy()
                
                meta_knowledge = model.mk_learner(data, dim=3).cpu().numpy()
                return meta_knowledge
        
        source_spatial = extract_spatial_features(source_model, source_data)
        target_spatial = extract_spatial_features(target_model, target_data)
        
        source_adj = None
        target_adj = None
        
        if hasattr(source_data, 'edge_index') and hasattr(target_data, 'edge_index'):
            source_edge_index = source_data.edge_index.cpu().numpy()
            target_edge_index = target_data.edge_index.cpu().numpy()
            
            source_nodes = max(np.max(source_edge_index[0]), np.max(source_edge_index[1])) + 1
            target_nodes = max(np.max(target_edge_index[0]), np.max(target_edge_index[1])) + 1
            
            source_adj = np.zeros((source_nodes, source_nodes))
            source_adj[source_edge_index[0], source_edge_index[1]] = 1
            
            target_adj = np.zeros((target_nodes, target_nodes))
            target_adj[target_edge_index[0], target_edge_index[1]] = 1
        
        source_avg = np.mean(source_spatial, axis=0)
        target_avg = np.mean(target_spatial, axis=0)
        
        min_nodes = min(source_avg.shape[0], target_avg.shape[0])
        source_avg = source_avg[:min_nodes]
        target_avg = target_avg[:min_nodes]
        
        if source_avg.shape[1] == target_avg.shape[1]:
            similarity_matrix = cosine_similarity(source_avg, target_avg)
            avg_similarity = np.mean(np.diag(similarity_matrix))
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix[:min(50, min_nodes), :min(50, min_nodes)], 
                       cmap='viridis', annot=False)
            plt.title(f'Spatial Feature Similarity between Source and Target Cities\nAvg Similarity: {avg_similarity:.4f}')
            plt.xlabel('Target City Nodes')
            plt.ylabel('Source City Nodes')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'spatial_similarity.png'))
            plt.close()
        else:
            avg_similarity = None
            print(f"Warning: Feature dimensions don't match for similarity calculation: {source_avg.shape[1]} vs {target_avg.shape[1]}")
            
        if source_adj is not None and target_adj is not None:
            self._visualize_graph_structures(source_adj, target_adj, 'spatial_graph_comparison.png')
        
        results = {
            'source_spatial_dim': source_spatial.shape,
            'target_spatial_dim': target_spatial.shape,
            'avg_similarity': avg_similarity
        }
        
        return results
    
    def analyze_temporal_alignment(self, source_model, target_model, source_data, target_data):
        if not self.enable_rq2:
            return {}
            
        print("Analyzing RQ2: Temporal Alignment...")
        
        def extract_temporal_features(model, data):
            model.eval()
            with torch.no_grad():
                data = data.to(next(model.parameters()).device)
                
                if hasattr(model, 'temporal_encoder'):
                    temporal_features = model.temporal_encoder(data.x)
                    if temporal_features.dim() == 4:
                        temporal_features = temporal_features[:, :, -1, :]
                    return temporal_features.cpu().numpy()
                
                predictions, _, _ = model(data)
                return predictions.cpu().numpy()
        
        source_temporal = extract_temporal_features(source_model, source_data)
        target_temporal = extract_temporal_features(target_model, target_data)
        
        source_avg = np.mean(source_temporal, axis=(0, 1))
        target_avg = np.mean(target_temporal, axis=(0, 1))
        
        min_dim = min(len(source_avg), len(target_avg))
        source_avg = source_avg[:min_dim]
        target_avg = target_avg[:min_dim]
        
        correlation = np.corrcoef(source_avg, target_avg)[0, 1]
        
        plt.figure(figsize=(10, 6))
        plt.plot(source_avg, label='Source City', linewidth=2)
        plt.plot(target_avg, label='Target City', linewidth=2)
        plt.title(f'Temporal Pattern Comparison\nCorrelation: {correlation:.4f}')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Average Activation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'temporal_patterns.png'))
        plt.close()
        
        if hasattr(source_data, 'y') and hasattr(target_data, 'y'):
            with torch.no_grad():
                source_model.eval()
                target_model.eval()
                
                source_data = source_data.to(next(source_model.parameters()).device)
                target_data = target_data.to(next(target_model.parameters()).device)
                
                source_pred, _, _ = source_model(source_data)
                target_pred, _, _ = target_model(target_data)
                
                source_pred = source_pred[0, 0].cpu().numpy()
                source_true = source_data.y[0, 0].cpu().numpy()
                
                target_pred = target_pred[0, 0].cpu().numpy()
                target_true = target_data.y[0, 0].cpu().numpy()
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                ax1.plot(source_true, 'b-', label='Ground Truth')
                ax1.plot(source_pred, 'r--', label='Prediction')
                ax1.set_title('Source City: Prediction vs Ground Truth')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Traffic Flow')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(target_true, 'b-', label='Ground Truth')
                ax2.plot(target_pred, 'r--', label='Prediction')
                ax2.set_title('Target City: Prediction vs Ground Truth')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Traffic Flow')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'prediction_comparison.png'))
                plt.close()
        
        results = {
            'source_temporal_dim': source_temporal.shape,
            'target_temporal_dim': target_temporal.shape,
            'temporal_correlation': correlation
        }
        
        return results
    
    def analyze_feature_fusion(self, source_model, target_model, adapter, source_data, target_data):
        if not self.enable_rq3 or adapter is None:
            return {}
            
        print("Analyzing RQ3: Cross-Domain Feature Fusion...")
        
        def extract_features(model, data):
            model.eval()
            with torch.no_grad():
                data = data.to(next(model.parameters()).device)
                _, _, _ = model(data)
                if hasattr(model, 'mk_learner'):
                    meta_knowledge = model.mk_learner(data, dim=3)
                    return meta_knowledge.cpu().numpy()
                return None
        
        source_features = extract_features(source_model, source_data)
        target_features_before = extract_features(target_model, target_data)
        
        adapter.eval()
        with torch.no_grad():
            source_data = source_data.to(next(adapter.parameters()).device)
            target_data = target_data.to(next(adapter.parameters()).device)
            
            source_model.eval()
            target_model.eval()
            
            _, _, _ = source_model(source_data)
            source_intermediate = self._get_intermediate_features(source_model, source_data)
            source_meta = source_model.mk_learner(source_data, dim=3)
            
            _, _, _ = target_model(target_data)
            target_intermediate = self._get_intermediate_features(target_model, target_data)
            target_meta = target_model.mk_learner(target_data, dim=3)
            
            adapted_features, similarity_scores = adapter(source_intermediate, target_intermediate)
            adapted_meta = adapter.transfer_meta_knowledge(source_meta, target_meta)
            
            adapted_features_np = adapted_features.cpu().numpy()
            similarity_scores_np = similarity_scores.cpu().numpy()
            adapted_meta_np = adapted_meta.cpu().numpy()
        
        feature_change = None
        if target_features_before is not None and adapted_meta_np is not None:
            if target_features_before.shape == adapted_meta_np.shape:
                feature_change = np.mean(np.abs(adapted_meta_np - target_features_before))
            
        if similarity_scores_np is not None:
            avg_sim = np.mean(similarity_scores_np)
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_scores_np[0, :min(30, similarity_scores_np.shape[1]), 
                                              :min(30, similarity_scores_np.shape[2])], 
                       cmap='viridis', annot=False)
            plt.title(f'Cross-City Node Similarity Scores\nAverage Similarity: {avg_sim:.4f}')
            plt.xlabel('Source City Nodes')
            plt.ylabel('Target City Nodes')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'cross_city_similarity.png'))
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.hist(similarity_scores_np.flatten(), bins=50, alpha=0.75)
            plt.title('Distribution of Cross-City Node Similarities')
            plt.xlabel('Similarity Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'similarity_distribution.png'))
            plt.close()
        
        results = {
            'source_features_shape': source_features.shape if source_features is not None else None,
            'target_features_before_shape': target_features_before.shape if target_features_before is not None else None,
            'adapted_features_shape': adapted_features_np.shape,
            'feature_change_magnitude': feature_change,
            'avg_similarity_score': np.mean(similarity_scores_np) if similarity_scores_np is not None else None
        }
        
        return results
    
    def analyze_graph_structure(self, source_model, target_model, source_data, target_data):
        if not self.enable_rq4:
            return {}
            
        print("Analyzing RQ4: Graph Structure Consistency...")
        
        def extract_graph_structure(model, data):
            model.eval()
            with torch.no_grad():
                data = data.to(next(model.parameters()).device)
                _, _, graph = model(data)
                if graph is not None:
                    return graph.cpu().numpy()
                
                if hasattr(data, 'edge_index'):
                    edge_index = data.edge_index.cpu().numpy()
                    nodes = max(np.max(edge_index[0]), np.max(edge_index[1])) + 1
                    adj = np.zeros((nodes, nodes))
                    adj[edge_index[0], edge_index[1]] = 1
                    return adj
                
                return None
        
        source_graph = extract_graph_structure(source_model, source_data)
        target_graph = extract_graph_structure(target_model, target_data)
        
        results = {}
        if source_graph is not None and target_graph is not None:
            def calculate_graph_metrics(adj_matrix):
                if not np.issubdtype(adj_matrix.dtype, np.integer):
                    adj_matrix = (adj_matrix > 0.5).astype(int)
                
                G = nx.from_numpy_array(adj_matrix)
                
                metrics = {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'density': nx.density(G),
                    'avg_clustering': nx.average_clustering(G),
                    'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan,
                    'degree_centrality': np.mean(list(nx.degree_centrality(G).values())),
                    'is_connected': nx.is_connected(G)
                }
                
                return metrics, G
            
            max_nodes = 50
            source_subset = source_graph[:max_nodes, :max_nodes] if source_graph.shape[0] > max_nodes else source_graph
            target_subset = target_graph[:max_nodes, :max_nodes] if target_graph.shape[0] > max_nodes else target_graph
            
            source_metrics, source_G = calculate_graph_metrics(source_subset)
            target_metrics, target_G = calculate_graph_metrics(target_subset)
            
            comparison = {}
            for metric in source_metrics:
                if metric in ['nodes', 'edges']:
                    comparison[metric] = (source_metrics[metric], target_metrics[metric])
                elif not np.isnan(source_metrics[metric]) and not np.isnan(target_metrics[metric]):
                    comparison[metric] = (source_metrics[metric], target_metrics[metric])
                    comparison[f'{metric}_ratio'] = target_metrics[metric] / source_metrics[metric]
            
            results['graph_metrics'] = comparison
            
            self._visualize_graph_structures(source_subset, target_subset, 'graph_structure_comparison.png')
            
            source_degrees = [d for _, d in source_G.degree()]
            target_degrees = [d for _, d in target_G.degree()]
            
            plt.figure(figsize=(10, 6))
            plt.hist(source_degrees, bins=20, alpha=0.5, label='Source City')
            plt.hist(target_degrees, bins=20, alpha=0.5, label='Target City')
            plt.title('Node Degree Distribution Comparison')
            plt.xlabel('Node Degree')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'degree_distribution.png'))
            plt.close()
                
        return results
    
    def analyze_sample_efficiency(self, framework, target_city='pems-bay'):
        if not self.enable_rq5:
            return {}
            
        print("Analyzing RQ5: Sample Efficiency...")
        
        k_shots = [1, 3, 5, 10, 20]
        metrics = {'rmse': [], 'mae': []}
        
        for k in k_shots:
            print(f"Testing {k}-shot learning...")
            
            _, shot_metrics = framework.few_shot_adaptation(
                target_city=target_city,
                n_way=5,
                k_shot=k,
                query_size=100
            )
            
            metrics['rmse'].append(shot_metrics['rmse'])
            metrics['mae'].append(shot_metrics['mae'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_shots, metrics['rmse'], 'o-', linewidth=2, label='RMSE')
        plt.plot(k_shots, metrics['mae'], 's-', linewidth=2, label='MAE')
        plt.title('Sample Efficiency: Error vs. Number of Examples')
        plt.xlabel('Number of Examples (k-shot)')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'sample_efficiency.png'))
        plt.close()
        
        for k in k_shots:
            adapted_model = framework.few_shot_learner.adapt_to_target(
                support_data=framework.target_dataset.get_few_shot_support_query(
                    dataset_name=target_city,
                    k_shot=k,
                    query_size=1
                )[0],
                support_labels=framework.target_dataset.get_few_shot_support_query(
                    dataset_name=target_city,
                    k_shot=k,
                    query_size=1
                )[1],
                num_adaptation_steps=10,
                lr=0.001
            )
            
            query_data = framework.target_dataset.get_few_shot_support_query(
                dataset_name=target_city,
                k_shot=1,
                query_size=1
            )[2]
            
            with torch.no_grad():
                adapted_model.eval()
                query_data = query_data.to(next(adapted_model.parameters()).device)
                pred, _, _ = adapted_model(query_data)
                pred = pred[0, 0].cpu().numpy()
                true = query_data.y[0, 0].cpu().numpy()
                
                plt.figure(figsize=(10, 6))
                plt.plot(true, 'b-', linewidth=2, label='Ground Truth')
                plt.plot(pred, 'r--', linewidth=2, label='Prediction')
                plt.title(f'{k}-shot Learning: Prediction vs Ground Truth')
                plt.xlabel('Time Step')
                plt.ylabel('Traffic Flow')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f'prediction_k{k}.png'))
                plt.close()
        
        results = {
            'k_shots': k_shots,
            'rmse': metrics['rmse'],
            'mae': metrics['mae']
        }
        
        return results
    
    def _visualize_graph_structures(self, source_adj, target_adj, filename):
        max_nodes = 100
        if source_adj.shape[0] > max_nodes or target_adj.shape[0] > max_nodes:
            source_adj = source_adj[:max_nodes, :max_nodes]
            target_adj = target_adj[:max_nodes, :max_nodes]
        
        source_G = nx.from_numpy_array(source_adj)
        target_G = nx.from_numpy_array(target_adj)
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 2, 1)
        pos_source = nx.spring_layout(source_G, seed=42)
        nx.draw(source_G, pos_source, with_labels=False, node_size=50, 
                node_color='blue', alpha=0.7, linewidths=0.5, font_size=10)
        plt.title(f'Source City Graph\n{source_G.number_of_nodes()} nodes, {source_G.number_of_edges()} edges')
        
        plt.subplot(1, 2, 2)
        pos_target = nx.spring_layout(target_G, seed=42)
        nx.draw(target_G, pos_target, with_labels=False, node_size=50, 
                node_color='red', alpha=0.7, linewidths=0.5, font_size=10)
        plt.title(f'Target City Graph\n{target_G.number_of_nodes()} nodes, {target_G.number_of_edges()} edges')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename))
        plt.close()
    
    def _get_intermediate_features(self, model, data):
        if not hasattr(data, 'x'):
            return None
            
        batch_size = data.x.size(0) if data.x.dim() > 2 else 1
        node_num = data.node_num if hasattr(data, 'node_num') else data.x.size(1) if data.x.dim() > 2 else data.x.size(0)
        
        if hasattr(model, 'cross_domain_fusion') and hasattr(model, 'temporal_encoder'):
            spatial_features = None
            if hasattr(data, 'edge_index'):
                edge_index = data.edge_index
                input_data = data.x
                
                if hasattr(model, 'spatial_encoder'):
                    spatial_features = []
                    for b in range(batch_size):
                        batch_feats = input_data[b] if input_data.dim() > 2 else input_data
                        x = batch_feats
                        for layer in model.spatial_encoder:
                            if hasattr(layer, 'forward'):
                                x = layer(x, edge_index)
                            else:
                                x = layer(x)
                        spatial_features.append(x)
                    
                    if len(spatial_features) > 0:
                        spatial_features = torch.stack(spatial_features)
            
            return spatial_features if spatial_features is not None else model.temporal_encoder(data.x)
        else:
            if data.x.dim() > 2:
                return data.x.reshape(batch_size, node_num, -1).mean(dim=1)
            else:
                return data.x.unsqueeze(0) if data.x.dim() < 3 else data.x