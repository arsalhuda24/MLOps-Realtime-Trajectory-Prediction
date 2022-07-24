from torch.utils.data import DataLoader
from lstm_encoder_decoder import lstm_seq2seq


model = lstm_seq2seq(2,16,32)

print(model)
a = w
def data_loader(path):
    dset = TrajectoryDataset1(
        path,
        obs_len=8,
        pred_len=12,
        skip=1,
        delim=',')

    loader = DataLoader(
        dset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=seq_collate)
    return dset, loader


train_path = "/home/asyed/Desktop/train/sampled/ver2_12sec_track/"
val_path = "/home/asyed/Desktop/train/sampled/ver1_12sec_track/"


train_dset, train_loader = data_loader(train_path)
val_dset, val_loader = data_loader(val_path)

def train(model, iterator, optimizer, criterion, clip=1):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

        src = obs_traj_rel.float()#type(torch.LongTensor)
        

        trg = pred_traj_gt_rel.float()#type(torch.LongTensor)
        
        
        optimizer.zero_grad()
        
        output = model(src.cuda(), 12)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.cuda().eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            src = obs_traj_rel.float().cuda()

            trg = pred_traj_gt_rel.float().cuda()

            output = model(src , 12) #turn off teacher forcing


            loss = criterion(output.cuda(), trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)



n_epochs = 50
# losses = np.full(n_epochs, np.nan)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()
tb = SummaryWriter()

import math

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')


t_loss = []
b_loss = []
v_loss = []
for epoch in range(0,150):

    start_time = time.time()
    
    train_loss = train(model, train_loader, optimizer, criterion)
    valid_loss = evaluate(model, val_loader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    t_loss.append(train_loss)
    v_loss.append(valid_loss)

    print(train_loss)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        b_loss.append(best_valid_loss)
        torch.save(model.state_dict(), '/home/asyed/Desktop/lstm_enc_dec.pt')
        print("BEST-MODEL-SAVED")
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


    
