import torch

def save_checkpoint(epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch + 1, # Save the next epoch number to start from
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss, # Or maybe validation loss
        }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")

def avg(l):
    sum = 0
    for i in l:
        sum += i
    return sum / len(l)

class EarlyStopping():
    def __init__(self, min_delta, tolerance):
        self.min_delta = min_delta
        self.tolerance = tolerance
        self.min_validation_loss = float('inf')
        self.counter = 0

    def early_stop(self, val_loss):
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_validation_loss + self.min_delta):
            self.counter+=1
            if self.counter >= self.tolerance:
                return True
        return False