import os
import torch


def save_model(save_dir, model, epoch, optimizer, prefix):
    state = {
        'state_dict':model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer,
    }
    save_name = prefix +'_best.t7'
    save_path = os.path.join(save_dir, save_name)
    print("saving {}...".format(save_path))
    torch.save(state, save_path)

def save_best_model(dataset, best_acc, epoch, save_dir, model):
    if (dataset == 'SFEW' and best_acc >= 0.58) or (dataset == 'RAF' and best_acc >= 0.87):
        state = {
            'state_dict':model.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc
        }
        save_name = 'model_acc'+ '%.4f'%(best_acc*100) +'.t7'
        save_path = os.path.join(save_dir, save_name)
        print("saving {}...".format(save_path))
        torch.save(state, save_path)
