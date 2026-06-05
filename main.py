import os
import argparse
import time
import torch
import numpy as np
from typing import Dict, Tuple, Any

from datasets import DataManager
from train    import MCDTrainer
from utils    import (
    load_config, save_config, set_random_seed, create_experiment_dir,
    get_device, fmt_time, count_parameters, save_predictions,
    create_comprehensive_plots, result_print,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MCD-CKT: Meta Cross-Domain Fusion with Cross-City Knowledge Transfer'
    )
    # Data / experiment
    parser.add_argument('--config',          default='config.yaml', type=str,   help='Configuration file')
    parser.add_argument('--test_dataset',    default='shenzhen',        type=str,   help='Target dataset')
    parser.add_argument('--target_days',     default=3,                 type=int,   help='Few-shot target days')
    parser.add_argument('--experiment_name', default='mcd_ckt',         type=str,   help='Experiment name')
    parser.add_argument('--save_dir',        default='./experiments',   type=str,   help='Save directory')
    parser.add_argument('--device',          default='auto',            type=str,   help='Device (auto/cpu/cuda)')
    parser.add_argument('--seed',            default=42,                type=int,   help='Random seed')
    parser.add_argument('--resume_from',     default=None,              type=str,   help='Checkpoint to resume from')
    parser.add_argument('--eval_only',       default=False,             type=bool,  help='Evaluation only')

    # Training phases
    parser.add_argument('--enable_standard_training', default=True,  type=bool, help='Run standard training')
    parser.add_argument('--enable_finetuning',         default=True,  type=bool, help='Run fine-tuning')
    parser.add_argument('--source_epochs',  default=200, type=int,   help='Standard training epochs')
    parser.add_argument('--target_epochs',  default=300, type=int,   help='Fine-tuning epochs')

    # Batch / compute
    parser.add_argument('--batch_size',      default=1,  type=int,   help='DataLoader batch size')
    parser.add_argument('--test_batch_size', default=1,  type=int,   help='Test batch size')
    _mb = 2048 if torch.cuda.is_available() else 4096
    parser.add_argument('--minibatch_size',  default=_mb, type=int,  help='Mini-batch for grad computation')

    # Model architecture
    parser.add_argument('--hidden_dim',   default=64,   type=int,   help='Hidden dimension')
    parser.add_argument('--meta_dim',     default=32,   type=int,   help='Meta-knowledge dimension')
    parser.add_argument('--message_dim',  default=8,    type=int,   help='Message/feature dimension')
    parser.add_argument('--num_heads',    default=4,    type=int,   help='Attention heads')
    parser.add_argument('--dropout',      default=0.1,  type=float, help='Dropout rate')
    parser.add_argument('--tp',           default=True, type=bool,  help='Use temporal processor')
    parser.add_argument('--sp',           default=True, type=bool,  help='Use spatial processor')

    # Optimisation
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
    return parser.parse_args()


def setup_experiment(args) -> Tuple[Dict, str, torch.device]:
    set_random_seed(args.seed)
    device = get_device(args.device)
    config = load_config(args.config)

    # Override config with CLI args
    config['model']['hidden_dim']   = args.hidden_dim
    config['model']['meta_dim']     = args.meta_dim
    config['model']['message_dim']  = args.message_dim
    config['model']['num_heads']    = args.num_heads
    config['model']['dropout']      = args.dropout
    config['model']['tp']           = args.tp
    config['model']['sp']           = args.sp
    config['training']['learning_rate']   = args.learning_rate
    config['training']['model_batch_size']= int(config['training'].get('model_batch_size', 8))
    config['task']['model_batch_size']    = config['training']['model_batch_size']
    config['task']['batch_size']      = args.batch_size
    config['task']['test_batch_size'] = args.test_batch_size

    exp_dir = create_experiment_dir(args.save_dir, f"{args.experiment_name}_{args.test_dataset}")
    save_config(config,     os.path.join(exp_dir, 'config.yaml'))
    save_config(vars(args), os.path.join(exp_dir, 'args.yaml'))

    print(f"Device         : {device}")
    print(f"Test dataset   : {args.test_dataset}")
    print(f"Seed           : {args.seed}")
    print(f"Minibatch      : {args.minibatch_size}")
    return config, exp_dir, device


def create_data_loaders(config: Dict, args) -> Dict[str, Any]:
    print(f"\nLoading data for: {args.test_dataset}")
    dm = DataManager(config['data'], config['task'])
    loaders = dm.create_all_dataloaders(
        test_data=args.test_dataset,
        target_days=args.target_days,
        minibatch_size=args.minibatch_size,
    )
    # Patch config with actual values read from data
    ds = dm.datasets.get('source') or dm.datasets.get('test')
    if ds is not None:
        config['data'][args.test_dataset]['num_nodes']       = ds.num_nodes
        config['data'][args.test_dataset]['node_feature_dim']= ds.num_features
        config['model']['node_feature_dim']                  = ds.num_features
        print(f"Actual num_nodes={ds.num_nodes}  node_feature_dim={ds.num_features}")
    return loaders


def create_model(config: Dict, test_dataset: str, device: torch.device) -> torch.nn.Module:
    from models.meta_model import create_model as _create
    data_cfg  = config['data'][test_dataset]
    model_cfg = config['model']
    task_cfg  = config['task']

    model_args = {
        'type':             model_cfg.get('type', 'MetaCrossDomainFusion'),
        'node_feature_dim': data_cfg.get('node_feature_dim', 1),
        'hidden_dim':       model_cfg.get('hidden_dim', 64),
        'meta_dim':         model_cfg.get('meta_dim', 32),
        'message_dim':      model_cfg.get('message_dim', 8),
        'num_heads':        model_cfg.get('num_heads', 4),
        'dropout':          model_cfg.get('dropout', 0.1),
        'tp':               model_cfg.get('tp', True),
        'sp':               model_cfg.get('sp', True),
        'output_dim':       model_cfg.get('output_dim', 1),
        # Paper Eq. 4 — low-rank meta-knowledge regularisation
        'nuclear_norm_weight': model_cfg.get('nuclear_norm_weight', 0.001),
        # Paper Eq. 14 — FSL hybrid loss
        'lambda1':          model_cfg.get('lambda1', 0.1),
        'lambda2':          model_cfg.get('lambda2', 0.01),
        # Paper Eq. 16 — proximity constraint
        'epsilon':          model_cfg.get('epsilon', 0.1),
        # Paper Eq. 8 — RobustTransformer noise injection
        'noise_tau':        model_cfg.get('noise_tau', 0.1),
        'noise_alpha':      model_cfg.get('noise_alpha', 0.05),
        'enable_noise':     model_cfg.get('enable_noise', True),
    }
    task_args = {
        'his_num':  task_cfg.get('his_num', task_cfg.get('seq_len', 12)),
        'pred_num': task_cfg['pred_num'],
    }
    return _create(model_args, task_args, str(device))


def run_training(trainer: MCDTrainer, dataloaders: Dict, args, exp_dir: str) -> Dict:
    if not args.enable_standard_training:
        print("\nSkipping standard training (disabled)")
        return {}
    print(f"\n{'='*70}\nSTANDARD TRAINING\n{'='*70}")
    t0  = time.time()
    res = trainer.train(dataloaders['source'], dataloaders['validation'],
                        args.source_epochs, os.path.join(exp_dir, 'models'))
    print(f"Completed in {fmt_time(time.time() - t0)}")
    return res


def run_finetuning(trainer: MCDTrainer, dataloaders: Dict, args, exp_dir: str) -> Dict:
    if not args.enable_finetuning:
        print("\nSkipping fine-tuning (disabled)")
        return {}
    print(f"\n{'='*70}\nFINE-TUNING  ({args.target_days} target days)\n{'='*70}")
    t0  = time.time()
    res = trainer.finetune(dataloaders['target'], dataloaders['test'],
                           args.target_epochs, os.path.join(exp_dir, 'models'))
    print(f"Completed in {fmt_time(time.time() - t0)}")
    return res


def run_evaluation(trainer: MCDTrainer, dataloaders: Dict) -> Dict:
    print(f"\n{'='*70}\nFINAL EVALUATION\n{'='*70}")
    t0  = time.time()
    res = trainer.evaluate(dataloaders['test'])
    print(f"Completed in {fmt_time(time.time() - t0)}")
    result_print(res['metrics'], 'Test Results')
    return res


def save_results(exp_dir: str, args, config: Dict,
                 training_results: Dict, finetune_results: Dict,
                 eval_results: Dict, total_params: int,
                 trainer=None, dataloaders: Dict = None,
                 pretrained_state: dict = None):
    print(f"\n{'='*70}\nSAVING RESULTS\n{'='*70}")

    summary = {
        'experiment': args.experiment_name, 'dataset': args.test_dataset,
        'timestamp':  time.strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': total_params,
        'model':  {'hidden_dim': args.hidden_dim, 'meta_dim': args.meta_dim,
                   'message_dim': args.message_dim, 'num_heads': args.num_heads,
                   'tp': args.tp, 'sp': args.sp},
        'training': {'source_epochs': args.source_epochs, 'target_epochs': args.target_epochs,
                     'learning_rate': args.learning_rate, 'target_days': args.target_days},
    }

    if eval_results and 'metrics' in eval_results:
        m = eval_results['metrics']
        summary['performance'] = {
            'avg_mae_6':  float(np.mean(m['MAE'][:6])),
            'avg_rmse_6': float(np.mean(m['RMSE'][:6])),
            'avg_mape_6': float(np.mean(m['MAPE'][:6]) * 100),
            'best_mae':   float(np.min(m['MAE'])),
            'best_rmse':  float(np.min(m['RMSE'])),
            'best_mape':  float(np.min(m['MAPE']) * 100),
        }

    save_config(summary, os.path.join(exp_dir, 'results.yaml'))

    if eval_results and 'predictions' in eval_results:
        save_predictions(eval_results['predictions'], eval_results['targets'],
                         eval_results.get('metrics', {}),
                         os.path.join(exp_dir, 'predictions'))

    train_losses = training_results.get('loss_history', {}).get('total_loss', [])
    val_losses   = []
    if 'final_val_results' in training_results:
        fvr = training_results['final_val_results']
        val_losses = [fvr.get('loss', 0)]

    pretrain_metrics = None
    if training_results.get('final_val_results', {}).get('metrics'):
        pretrain_metrics = training_results['final_val_results']['metrics']

    if eval_results and 'predictions' in eval_results:
        pretrain_metrics = training_results.get('final_val_results', {}).get('metrics')
        create_comprehensive_plots(
            eval_results['predictions'],
            eval_results['targets'],
            eval_results.get('metrics', {}),
            train_losses,
            val_losses if val_losses else None,
            os.path.join(exp_dir, 'plots'),
            uncertainty=eval_results.get('uncertainty'),
            graph=eval_results.get('graph'),
            pretrain_metrics=pretrain_metrics,
        )

    txt = os.path.join(exp_dir, 'summary.txt')
    with open(txt, 'w') as f:
        f.write(f"MCD-CKT EXPERIMENT SUMMARY\n{'='*50}\n")
        f.write(f"Dataset    : {args.test_dataset}\n")
        f.write(f"Timestamp  : {summary['timestamp']}\n\n")
        f.write(f"MODEL\n{'-'*40}\n")
        f.write(f"Hidden dim : {args.hidden_dim}\n")
        f.write(f"Meta dim   : {args.meta_dim}\n")
        f.write(f"Message dim: {args.message_dim}\n")
        f.write(f"Temporal   : {args.tp}\n")
        f.write(f"Spatial    : {args.sp}\n")
        f.write(f"Parameters : {total_params:,}\n\n")
        f.write(f"TRAINING\n{'-'*40}\n")
        f.write(f"LR         : {args.learning_rate}\n")
        f.write(f"Src epochs : {args.source_epochs}\n")
        f.write(f"Tgt epochs : {args.target_epochs}\n\n")
        if 'performance' in summary:
            p = summary['performance']
            f.write(f"RESULTS (avg first 6 steps)\n{'-'*40}\n")
            f.write(f"MAE  : {p['avg_mae_6']:.4f}\n")
            f.write(f"RMSE : {p['avg_rmse_6']:.4f}\n")
            f.write(f"MAPE : {p['avg_mape_6']:.2f}%\n")

    # ── Research Question analyses (paper §5) ────────────────────────────────
    ev = config.get("evaluation", {})
    if trainer is not None and dataloaders and \
       any(ev.get(f"enable_rq{i}", True) for i in range(1, 6)):
        try:
            from rq_analyser import ResearchQuestionAnalyzer
            analyzer = ResearchQuestionAnalyzer(config, exp_dir)
            analyzer.run(trainer, dataloaders,
                         pretrained_state=pretrained_state)
        except Exception as e:
            print(f"[Analyzer] skipped: {e}")

    print(f"Results saved to: {exp_dir}")


def main():
    print(f"{'='*70}")
    print("MCD-CKT: Meta Cross-Domain Fusion + Cross-City Knowledge Transfer")
    print(f"{'='*70}")

    args                    = parse_args()
    config, exp_dir, device = setup_experiment(args)
    dataloaders             = create_data_loaders(config, args)
    model                   = create_model(config, args.test_dataset, device)
    total_params            = count_parameters(model)

    print(f"\nModel parameters: {total_params:,}")
    print(f"Num nodes       : {config['data'][args.test_dataset]['num_nodes']}")

    trainer = MCDTrainer(model, config, device)

    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print(f"Loaded weights from {args.resume_from}")

    t0               = time.time()
    training_results = {}
    finetune_results = {}

    if not args.eval_only:
        training_results = run_training(trainer, dataloaders, args, exp_dir)
        pretrained_state = {k: v.clone()
                            for k, v in trainer.model.state_dict().items()}
        finetune_results = run_finetuning(trainer, dataloaders, args, exp_dir)
    else:
        pretrained_state = None

    eval_results = run_evaluation(trainer, dataloaders)

    save_results(exp_dir, args, config, training_results, finetune_results,
                 eval_results, total_params,
                 trainer=trainer, dataloaders=dataloaders,
                 pretrained_state=pretrained_state)

    total_t = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETED  |  {fmt_time(total_t)}  |  {exp_dir}")
    if eval_results and 'metrics' in eval_results:
        m = eval_results['metrics']
        print(f"MAE={np.mean(m['MAE'][:6]):.4f}  "
              f"RMSE={np.mean(m['RMSE'][:6]):.4f}  "
              f"MAPE={np.mean(m['MAPE'][:6])*100:.2f}%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()