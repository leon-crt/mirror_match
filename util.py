import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score

MAX_X = 928
MIN_X = 93
MAX_Y = 226
MIN_Y = -42

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

def format_pred(t:torch.Tensor):
    res = [None] * len(t)
    for i in range(len(t)):
        if t[i] > 0.2:
            res[i] = 1
        else:
            res[i] = 0

    return str(res)[1:-1].replace(' ', '')

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
    
def ComputeMetrics(pred_seq: torch.Tensor, target_seq, batch_size, out_size):
    pred_np = pred_seq.cpu().detach().numpy()
    target_np = target_seq.cpu().detach().numpy()
    
    # 2. Flatten 3D sequential data [batch, seq, classes] -> 2D matrix [samples, classes]
    # This resolves the "unknown is not supported" error
    pred_flat = pred_np.reshape(-1, out_size)
    target_flat = target_np.reshape(-1, out_size).astype(int)
    
    # 3. Correct thresholding order (Assuming pred_seq contains probabilities)
    # Convert decimals to 0 or 1, THEN cast to integer
    pred_binary = (pred_flat >= 0.5).astype(int)
    prec = precision_score(target_flat, pred_binary, average=None, zero_division=0)
    rec = recall_score(target_flat, pred_binary, average=None, zero_division=0)
    
    return prec, rec

class PlayerFeatures():
    def __init__(self, posx, posy, health, meter, stun, isStunned, hit, thrown, inputs):
        self.posX = posx
        self.posY = posy
        self.health = health
        self.meter = meter
        self.stun = stun
        self.hit = hit
        self.isStunned = isStunned
        self.thrown = thrown
        self.inputs = inputs
    
    def normalize(self):
        self.posX = (self.posX - MIN_X) / (MAX_X - MIN_X)
        self.posY = (self.posY - MIN_Y) / (MAX_Y - MIN_Y)
        self.health = self.health / 161
        self.meter = self.meter / 336
        self.stun = self.stun / 70

class GameState():
    def __init__(self, s:str):
        self.feats = s.split(sep=',')[-29:-1]
        self.feats = [int(x) for x in self.feats]
        self.P1 = PlayerFeatures(*self.feats[:8], None)
        self.P2 = PlayerFeatures(*self.feats[8:16], self.feats[16:])
    
    def normalize(self):
        self.P1.normalize()
        self.P2.normalize()
        # p1 features
        self.feats[0] = self.P1.posX
        self.feats[1] = self.P1.posY
        self.feats[2] = self.P1.health
        self.feats[3] = self.P1.meter
        self.feats[4] = self.P1.stun
        # p2 features
        self.feats[8] = self.P2.posX
        self.feats[9] = self.P2.posY
        self.feats[10] = self.P2.health
        self.feats[11] = self.P2.meter
        self.feats[12] = self.P2.stun