from pathlib import Path
import model
import torch
import argparse
from eval.accuracy import evaluate_accuracy
from eval.stability import evaluate_stability
from eval.consistency import evaluate_consistency
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', default='CUB2011', type=str)
    parser.add_argument('--data_path', type=str, default='datasets')
    parser.add_argument('--nb_classes', type=int, default=200)
    parser.add_argument('--test_batch_size', type=int, default=30)

    # Model
    parser.add_argument('--base_architecture', type=str, default='dinov2_vitb_exp')  # dinov2_vitb_exp
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--num_prototypes', type=int, default=2000)

    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    output_path = Path(f'outputs/{args.base_architecture}-{args.num_prototypes}')
    output_path.mkdir(parents=True, exist_ok=True)
    filename = 'eval_results.txt'

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.resume, map_location='cpu')
    # ppnet.load_state_dict(checkpoint)
    ppnet = checkpoint
    ppnet.num_prototypes_per_class = 2000

    ppnet.to(device)
    ppnet.eval()

    evaluate_accuracy(net=ppnet, device=device)

    consistency_score = evaluate_consistency(ppnet, args, save_dir=output_path.as_posix())
    print('Consistency Score : {:.2f}%'.format(consistency_score))
    with open(output_path / filename, 'a') as fp:
        fp.write('Consistency Score : {:.2f}%\n'.format(consistency_score))

    stability_score = evaluate_stability(ppnet, args)
    print('Stability Score : {:.2f}%'.format(stability_score))
    with open(output_path / filename, 'a') as fp:
        fp.write('Stability Score : {:.2f}%\n'.format(stability_score))
