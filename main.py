#!/usr/bin/env python3
import os
import argparse
import yaml
import json
import torch
import numpy as np
import random
import time
from yaml_config import yaml_to_framework_config
from train import TrafficPredictionFramework
from datasets import TrafficDataset
from rq_analyser import ResearchQuestionAnalyzer
from models.adaptive_fsl import MetaCrossDomainFusion, CrossCityAdapter, FewShotTrafficLearner

def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Traffic Prediction with Cross-City Knowledge Transfer")
    
    parser.add_argument('--config', type=str, required=False, default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--source_city', type=str, default='metr-la', help='Source city dataset')
    parser.add_argument('--target_city', type=str, default='pems-bay', help='Target city dataset')
    
    parser.add_argument('--train_source', action='store_true', help='Train source model')
    parser.add_argument('--train_adapter', action='store_true', help='Train cross-city adapter')
    parser.add_argument('--perform_few_shot', action='store_true', help='Perform few-shot adaptation')
    parser.add_argument('--analyze_research_questions', action='store_true', help='Analyze research questions')
    
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--seed', type=int, help='Override random seed')
    parser.add_argument('--target_days', type=int, help='Override target days')
    parser.add_argument('--k_shot', type=int, help='Override k-shot value')
    
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU even if not specified in config')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    config = yaml_to_framework_config(args.config)
    
    if args.learning_rate is not None:
        config['model_args']['meta_lr'] = args.learning_rate
    
    if args.batch_size is not None:
        config['task_args']['batch_size'] = args.batch_size
    
    if args.seed is not None:
        config['system']['seed'] = args.seed
    
    if args.target_days is not None:
        config['evaluation']['target_days'] = args.target_days
    
    if args.k_shot is not None:
        config['evaluation']['k_shot'] = args.k_shot
    
    if args.use_gpu:
        config['system']['use_gpu'] = True
    
    device = 'cuda' if config['system'].get('use_gpu', True) and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    seed = config['system'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = config['system'].get('results_dir', './results')
    if args.output_dir:
        results_dir = args.output_dir
        
    output_dir = os.path.join(results_dir, f"traffic_prediction_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Initializing source city dataset...")
    source_dataset = TrafficDataset(
        data_args=config['data_args'],
        task_args=config['task_args'],
        model_args=config['model_args'],
        stage='source',
        test_data=args.target_city,
        add_target=False,
        cache_dir=config['system'].get('cache_dir', './cache'),
        use_weather=False,
        use_time_features=True
    )
    
    print("Initializing target city dataset...")
    target_dataset = TrafficDataset(
        data_args=config['data_args'],
        task_args=config['task_args'],
        model_args=config['model_args'],
        stage='target',
        test_data=args.target_city,
        add_target=True,
        target_days=config['evaluation'].get('target_days', 3),
        cache_dir=config['system'].get('cache_dir', './cache'),
        use_weather=False,
        use_time_features=True
    )
    
    framework = TrafficPredictionFramework(
        model_args=config['model_args'],
        task_args=config['task_args'],
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        device=device,
        learning_rate=config['model_args'].get('meta_lr', 0.001),
        weight_decay=config['training'].get('weight_decay', 1e-5),
        scheduler_factor=config['training'].get('lr_scheduler_factor', 0.5),
        scheduler_patience=config['training'].get('lr_scheduler_patience', 10),
        clip_grad_norm=5.0,
        log_dir=output_dir
    )
    
    if not any([args.train_source, args.train_adapter, args.perform_few_shot, args.analyze_research_questions]):
        args.train_source = True
        args.train_adapter = True
        args.perform_few_shot = True
        args.analyze_research_questions = True
    
    if args.train_source:
        print("\n" + "="*50)
        print("STAGE 1: SOURCE MODEL TRAINING")
        print("="*50)
        
        source_model = framework.train_source_model(
            batch_size=config['task_args'].get('batch_size', 32),
            epochs=config['training'].get('source_epochs', 100), 
            early_stopping_patience=config['training'].get('early_stop_patience', 15),
            source_city=args.source_city,
            val_ratio=0.2
        )
        
        print("\nEvaluating source model on source city...")
        source_metrics = framework.evaluate_multi_horizon(
            model=source_model,
            dataset=source_dataset,
            city_name=args.source_city,
            horizons=config['evaluation'].get('time_horizons', [5, 15, 30, 60])
        )
    
    if args.train_adapter:
        print("\n" + "="*50)
        print("STAGE 2: CROSS-CITY ADAPTER TRAINING")
        print("="*50)
        
        adapter, target_model = framework.train_cross_city_adapter(
            batch_size=config['task_args'].get('batch_size', 32),
            epochs=config['training'].get('target_epochs', 50),
            early_stopping_patience=config['training'].get('early_stop_patience', 10),
            source_city=args.source_city,
            target_city=args.target_city
        )
        
        print("\nEvaluating adapted model on target city...")
        target_metrics = framework.evaluate_multi_horizon(
            model=target_model,
            dataset=target_dataset,
            city_name=args.target_city,
            horizons=config['evaluation'].get('time_horizons', [5, 15, 30, 60])
        )
    
    if args.perform_few_shot:
        print("\n" + "="*50)
        print("STAGE 3: FEW-SHOT ADAPTATION")
        print("="*50)
        
        adapted_model, few_shot_metrics = framework.few_shot_adaptation(
            target_city=args.target_city,
            n_way=5,
            k_shot=config['evaluation'].get('k_shot', 5),
            query_size=100
        )
        
        print("\nEvaluating few-shot adapted model on target city...")
        few_shot_eval_metrics = framework.evaluate_multi_horizon(
            model=adapted_model,
            dataset=target_dataset,
            city_name=args.target_city,
            horizons=config['evaluation'].get('time_horizons', [5, 15, 30, 60])
        )
    
    if args.analyze_research_questions:
        print("\n" + "="*50)
        print("RESEARCH QUESTION ANALYSIS")
        print("="*50)
        
        analyzer = ResearchQuestionAnalyzer(config, output_dir)
        
        source_batch = next(iter(source_dataset.get_dataloader(batch_size=1)))
        target_batch = next(iter(target_dataset.get_dataloader(batch_size=1)))
        
        source_model = framework.source_model
        
        if 'target_model' in locals():
            rq1_results = analyzer.analyze_spatial_dependency(
                source_model, target_model, source_batch[0], target_batch[0]
            )
            
            rq2_results = analyzer.analyze_temporal_alignment(
                source_model, target_model, source_batch[0], target_batch[0]
            )
            
            rq3_results = analyzer.analyze_feature_fusion(
                source_model, target_model, framework.adapter, source_batch[0], target_batch[0]
            )
            
            rq4_results = analyzer.analyze_graph_structure(
                source_model, target_model, source_batch[0], target_batch[0]
            )
            
            if args.perform_few_shot:
                rq5_results = analyzer.analyze_sample_efficiency(
                    framework, target_city=args.target_city
                )
            else:
                rq5_results = {}
            
            research_results = {
                'rq1_spatial_dependency': rq1_results,
                'rq2_temporal_alignment': rq2_results,
                'rq3_feature_fusion': rq3_results,
                'rq4_graph_structure': rq4_results,
                'rq5_sample_efficiency': rq5_results
            }
            
            with open(os.path.join(output_dir, 'research_results.json'), 'w') as f:
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(i) for i in obj]
                    else:
                        return obj
                
                json.dump(convert_numpy(research_results), f, indent=4)
        else:
            print("Warning: Target model not available for research question analysis. Run the adapter stage first.")
    
    framework.plot_training_metrics()
    
    results = {
        'source_metrics': globals().get('source_metrics', None),
        'target_metrics': globals().get('target_metrics', None),
        'few_shot_metrics': globals().get('few_shot_metrics', None),
        'few_shot_eval_metrics': globals().get('few_shot_eval_metrics', None)
    }
    
    def convert_for_json(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        else:
            return obj
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(convert_for_json(results), f, indent=4)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print(f"Results saved to {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()