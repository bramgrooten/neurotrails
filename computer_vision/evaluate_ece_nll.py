import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from sparselearning.sparse_utils import get_cifar100_dataloaders
from sparselearning.resnet_wr import WideResNet_28
import utils


# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_prediction_disagreement(ensemble_preds):
    """
    Compute prediction disagreement matrix between ensemble members.
    
    Args:
        ensemble_preds: Tensor of shape (batch_size, ensemble_size, num_classes)
                       containing softmax probabilities from ensemble members
    
    Returns:
        disagreement_matrix: numpy array of shape (ensemble_size, ensemble_size)
                            where each element [i,j] is the disagreement between
                            ensemble members i and j
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(ensemble_preds, torch.Tensor):
        ensemble_preds = ensemble_preds.detach().cpu().numpy()
    
    batch_size, ensemble_size, num_classes = ensemble_preds.shape
    
    # Get the predicted class for each ensemble member
    predictions = np.argmax(ensemble_preds, axis=2)  # shape: (batch_size, ensemble_size)
    
    # Initialize disagreement matrix
    disagreement_matrix = np.zeros(shape=(ensemble_size, ensemble_size))
    
    # Calculate pairwise disagreements
    for i in range(ensemble_size):
        preds1 = predictions[:, i]  # predictions from ensemble member i
        for j in range(i, ensemble_size):
            preds2 = predictions[:, j]  # predictions from ensemble member j
            
            # Compute dissimilarity (disagreement)
            disagreement_score = 1 - np.sum(np.equal(preds1, preds2)) / batch_size
            
            # Fill the symmetric matrix
            disagreement_matrix[i, j] = disagreement_score
            if i != j:
                disagreement_matrix[j, i] = disagreement_score
    
    return disagreement_matrix

def evaluate_nll(args,model, device, test_loader):
    """
    Evaluate model using NLL metric only
    """
    model.eval()
    train_loss = 0
    correct = 0
    total = 0
    
    log_softmax = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            
            model.t = target
            output = model(data)
            
            if args.num_ensemble > 1:
                # output size = [#ensemble, batch, #class]
                output = torch.transpose(output, 0, 1)
                output = output.mean(dim=0)
                
            loss = loss_fn(log_softmax(output), target)
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            
            total += target.size(0)
            
            pbar.set_postfix({
                'NLL': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.0*correct/total:.2f}%'
            })

    accuracy = 100.0 * correct / total
    nll = train_loss / (batch_idx+1)
    return nll, accuracy

def main():
    # Parse arguments
    args = utils.parse_args()
    print(args)
    
    # Load dataset
    train_loader, test_loader = get_cifar100_dataloaders(args)
    
    # Track results
    p = defaultdict(list)

    # Compute disagreement
    agreements = []
    
    # Initialize and load model
    model = WideResNet_28(args).to(device)
    model_path = "/path/to/your/checkpoint.pth"  # Update this path

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Compute disagreement
    for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Computing Disagreement"):
        data, target = data.to(device), target.to(device)
        
        model.t = target
        output = model(data)
        agreements.append(compute_prediction_disagreement(output))
    
    mean_disagreement = np.mean(agreements)
    print(f"Seed: {args.seed}, Blocks: {args.blocks_in_head}, Mean Disagreement: {mean_disagreement}")
    p[str(args.blocks_in_head)].append(mean_disagreement)
    
    # Compute NLL
    nll, accuracy = evaluate_nll(args,model, device, test_loader)
    print(f"Seed: {args.seed}, Blocks: {args.blocks_in_head}, NLL: {nll}, Accuracy: {accuracy}%")
            
    return p

if __name__ == "__main__":
    results = main()
    print(results)