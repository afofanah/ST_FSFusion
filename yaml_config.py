import yaml
import json
import argparse
import os

def yaml_to_framework_config(yaml_file, output_file=None):
    with open(yaml_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    framework_config = {
        "model_args": {},
        "task_args": {},
        "data_args": {"data_keys": yaml_config["data"]["data_keys"]}
    }
    
    model_params = yaml_config["model"]
    framework_config["model_args"] = {
        "type": model_params.get("type", "MetaCross_DomainFusion"),
        "hidden_dim": model_params.get("hidden_dim", 64),
        "meta_dim": model_params.get("meta_dim", 32),
        "message_dim": model_params.get("message_dim", 16),
        "node_feature_dim": model_params.get("node_feature_dim", 2),
        "edge_feature_dim": model_params.get("edge_feature_dim", 4),
        "num_heads": model_params.get("num_heads", 4),
        "tp": model_params.get("tp", True),
        "sp": model_params.get("sp", True),
        "predict_uncertainty": True,
        "output_dim": model_params.get("output_dim", 1),
        "adapter_dim": 32,
        "update_lr": model_params.get("update_lr", 0.005),
        "meta_lr": model_params.get("meta_lr", 0.001),
        "loss_lambda": model_params.get("loss_lambda", 0.1),
        "update_step": model_params.get("update_step", 5),
        "update_step_test": model_params.get("update_step_test", 10)
    }
    
    task_params = yaml_config["task"]
    framework_config["task_args"] = {
        "his_num": task_params.get("his_num", 12),
        "pred_num": task_params.get("pred_num", 6),
        "batch_size": task_params.get("batch_size", 32),
        "test_batch_size": task_params.get("test_batch_size", 128),
        "task_num": task_params.get("task_num", 4)
    }
    
    for city in yaml_config["data"]["data_keys"]:
        if city in yaml_config["data"]:
            city_data = yaml_config["data"][city]
            framework_config["data_args"][city] = {
                "dataset_path": city_data.get("dataset_path", ""),
                "adjacency_matrix_path": city_data.get("adjacency_matrix_path", ""),
                "time_step": city_data.get("time_step", 0),
                "node_num": city_data.get("node_num", 0),
                "speed_mean": city_data.get("speed_mean", 0),
                "speed_std": city_data.get("speed_std", 1)
            }
    
    framework_config["training"] = yaml_config.get("training", {})
    framework_config["evaluation"] = yaml_config.get("evaluation", {})
    framework_config["system"] = yaml_config.get("system", {})
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(framework_config, f, indent=4)
        print(f"Converted config saved to {output_file}")
    
    return framework_config

def get_run_command(config, source_city="metr-la", target_city="pems-bay"):
    system_config = config.get("system", {})
    training_config = config.get("training", {})
    eval_config = config.get("evaluation", {})
    
    cmd = [
        "python -u train.py",
        f"--config_file config.json",
        f"--source_city {source_city}",
        f"--target_city {target_city}",
        f"--target_days {eval_config.get('target_days', 3)}",
        f"--cache_dir {system_config.get('cache_dir', './cache')}",
        "--use_time_features",
        f"--device {'cuda' if system_config.get('use_gpu', True) else 'cpu'}",
        f"--batch_size {config['task_args'].get('batch_size', 32)}",
        f"--learning_rate {config['model_args'].get('meta_lr', 0.001)}",
        f"--weight_decay {training_config.get('weight_decay', 1e-5)}",
        f"--scheduler_factor {training_config.get('lr_scheduler_factor', 0.5)}",
        f"--scheduler_patience {training_config.get('lr_scheduler_patience', 10)}",
        f"--clip_grad_norm 5.0",
        f"--val_ratio 0.2",
        f"--seed {system_config.get('seed', 42)}",
        f"--log_dir {system_config.get('results_dir', './results')}",
        "--train_source",
        "--train_adapter",
        "--perform_few_shot",
        f"--source_epochs {training_config.get('source_epochs', 100)}",
        f"--source_patience {training_config.get('early_stop_patience', 15)}",
        f"--adapter_epochs {training_config.get('target_epochs', 50)}",
        f"--adapter_patience {training_config.get('early_stop_patience', 10)}",
        f"--n_way 5",
        f"--k_shot {eval_config.get('k_shot', 5)}",
        f"--query_size 100"
    ]
    
    return " \\\n    ".join(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YAML config to framework JSON config")
    parser.add_argument('--yaml_config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--output_file', type=str, default='config.json', help='Path to save JSON config')
    parser.add_argument('--source_city', type=str, default='metr-la', help='Source city name')
    parser.add_argument('--target_city', type=str, default='pems-bay', help='Target city name')
    parser.add_argument('--generate_command', action='store_true', help='Generate run command')
    
    args = parser.parse_args()
    
    converted_config = yaml_to_framework_config(args.yaml_config, args.output_file)
    
    if args.generate_command:
        command = get_run_command(converted_config, args.source_city, args.target_city)
        command_file = os.path.splitext(args.output_file)[0] + "_command.sh"
        with open(command_file, 'w') as f:
            f.write("#!/bin/bash\n\n" + command + "\n")
        os.chmod(command_file, 0o755)
        print(f"Run command saved to {command_file}")