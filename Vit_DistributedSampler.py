import os
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, DistributedSampler
import timm
from tqdm import tqdm
import numpy as np
import timm
import os
import torch
import numpy as np
import random


DATA='COVID_PAPER/DATA/classification_aunet_clahe_seg_data _12000'
SAVE_MODEL_PATH='COVID_PAPER/VIT_Base_scratch_batch64_drop0.9_k10/saved'
NUM_CLASSES = 5
NUM_EPOCHS = 35
K_FOLDS = 10
PATIENCE = 100  # Number of epochs to wait before early stopping if no improvement in validation loss
MODEL_NAME='vit_base_patch16_224'
PRETRAIN=False
DROP_OUT=0
learning_rate=0.0001

scratch_learning= not PRETRAIN

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Note: For `torch.backends.cudnn.deterministic` setting,
    # setting this to True can degrade performance
    #! but is necessary for reproducibility.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False




def validate(model, loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct_predictions / total
    return avg_loss, accuracy

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

from sklearn.model_selection import StratifiedKFold

# ... other necessary imports remain the same ...

def main_worker(gpu, ngpus_per_node):
    set_seed(42)
    setup(gpu, ngpus_per_node)
    torch.cuda.set_device(gpu)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=DATA, transform=transform)

    # Get the targets from the dataset for StratifiedKFold
    targets = [label for _, label in dataset.samples]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    criterion = nn.CrossEntropyLoss()
    torch.cuda.synchronize()

    # Use StratifiedKFold to split the dataset
    for fold, (train_indices, test_indices) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        torch.cuda.synchronize()
        print(f"Starting fold {fold+1}")

        # Create Subset based on stratified indices
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        # Create samplers for distributed training
        train_sampler = DistributedSampler(train_subset, num_replicas=ngpus_per_node, rank=gpu)
        test_sampler = DistributedSampler(test_subset, num_replicas=ngpus_per_node, rank=gpu, shuffle=False)

        # Create DataLoaders for training and validation
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, sampler=test_sampler)
        print(f"loaded the data")
        # Load the pre-trained MobileViT model for the specified number of classes
        # Define your custom configuration with desired dropout rates
        model = timm.create_model(MODEL_NAME, pretrained=PRETRAIN, num_classes=NUM_CLASSES,drop_rate=DROP_OUT)
        
        model = model.cuda(gpu)
        model = DDP(model, device_ids=[gpu]) 

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  #! Adding weight decay

        # Early stopping setup
        best_loss = float('inf')
        patience_counter = 0
        torch.cuda.synchronize()

        for epoch in range(NUM_EPOCHS):
            torch.cuda.synchronize()
            model.train()
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            correct_predictions = 0
            total_train_samples = 0

            with tqdm(total=len(train_loader), desc=f"Fold {fold+1}/{K_FOLDS}, Epoch {epoch+1}/{NUM_EPOCHS}", disable=gpu != 0) as pbar:
                torch.cuda.synchronize()
                for images, labels in train_loader:
                    images, labels = images.cuda(gpu), labels.cuda(gpu)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    total_train_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()
                    epoch_loss += loss.item() * images.size(0)
                    pbar.update(1)
                torch.cuda.synchronize()

            # Calculate average loss and accuracy over the epoch
            avg_train_loss = epoch_loss / total_train_samples
            train_accuracy = 100.0 * correct_predictions / total_train_samples
            torch.cuda.synchronize()
            # Validation phase
            if gpu == 0:  # Perform validation only on the main GPU
                val_loss, val_accuracy = validate(model, test_loader, criterion, gpu)
                _,train_accuracy=validate(model, train_loader, criterion, gpu)
                print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
                torch.cuda.synchronize()

                print(f"saving the model with val loss: {val_loss}")
                torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}/fold_{fold+1}_{MODEL_NAME}_epoch_{epoch}_pretrain_{PRETRAIN}_DROPOUT_{DROP_OUT}_lr_{learning_rate}.pth')
                patience_counter = 0
                torch.cuda.synchronize()
        break
    cleanup()

if __name__ == "__main__":
    print(f"training model:{MODEL_NAME} with Dropout:{DROP_OUT} and learning rate:{learning_rate} it is scratch learning:{scratch_learning} also the learning rate is:{learning_rate}.")
    ngpus_per_node = torch.cuda.device_count()
    print(f"Using {ngpus_per_node} GPUs.")
    mp.spawn(main_worker, args=(ngpus_per_node,), nprocs=ngpus_per_node, join=True)
