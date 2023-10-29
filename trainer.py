#!/mnt/home/fboreiko/home/fboreikonew/bin/python
#SBATCH -p gpu 
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH -t 48:00:00

import torch
import torch.multiprocessing as mp
import optuna
import numpy as np
from construction_site import worker, GetMMstar

catalogue1 = 'catalogues/fof_subhalo_tab_030.hdf5'
catalogue2 = 'catalogues/fof_subhalo_tab_031.hdf5'
catalogue3 = 'catalogues/fof_subhalo_tab_032.hdf5'
#catalogue4 = 'catalogues/fof_subhalo_tab_033.hdf5'
catalogues = [catalogue1, catalogue2, catalogue3]
saving_destination = 'construction_results'
model_name = 'AutoregressiveSpline(CustomNN)MulGPU_MMstar'

#train_set, val_set, test_set = GetMMstar(catalogue)

def objective(trial):

    #This function will be optimized through optuna
    
    ntrial = trial.number
    print('Optuna trial: ', ntrial)
    
    hidden_dims1 = trial.suggest_int('hidden_dims1', 32, 256)
    hidden_dims2 = trial.suggest_int('hidden_dims2', 32, 256)
    count_bins = trial.suggest_int('count_bins', 32, 512)
    bound = trial.suggest_float('bound', 5., 32.)
    flow_length = trial.suggest_int('flow_length', 1, 12)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log = True)
    
    
    num_epochs = 20000
    model_list = (hidden_dims1, hidden_dims2, count_bins, bound, flow_length, learning_rate)
    world_size = torch.cuda.device_count()
    print('world_size: ', world_size)
    mp.spawn(worker, nprocs=world_size, args=(world_size, catalogues, saving_destination, model_list, num_epochs, ntrial,))
    
    with open(f'{saving_destination}/trial{ntrial}-bestloss.txt') as f:
        cont = f.readlines()
        
    best_val_loss = float(cont[0])
    
    return best_val_loss


if __name__ == "__main__":

    # Optuna parameters
    study_name = "multgpu"
    n_trials = 30

    # Define sampler and start optimization
    storage = 'sqlite:///trials_multgpu.db'
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials, gc_after_trial = True)

    print('Optuna finished!')

    # Print info for best trial
    lines = []
    lines.append(f"Model name: {model_name}")
    print("Best trial: ")
    trial = study.best_trial
    lines.append(f"Best trial: {trial.number}")
    print("  Value: ", trial.value)
    lines.append(f"Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        string = "    {}: {}".format(key, value)
        print(string)
        lines.append(string)

    with open(f'optuna_results/{model_name}.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

    with open(f'{model_name}.csv', 'w', newline='') as csvfile:
        fieldnames = ['best_trial']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'best_trial': f'{study.best_trial.number}'})