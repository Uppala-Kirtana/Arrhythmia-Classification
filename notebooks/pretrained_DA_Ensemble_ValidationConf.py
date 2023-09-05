import import_ipynb
from ecg_utilities import *

import torch.nn.functional as Func

from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import WeightCheckpoint, Verbose, LossPlot, EarlyStopping, Callback, CallbackInfo
from pytorch_sklearn.utils.func_utils import to_safe_tensor

patient_ids = pd.read_csv(osj("/content/files", "patient_ids.csv"), header=None).to_numpy().reshape(-1)
valid_patients = pd.read_csv(osj("/content/files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)
DATASET_PATH = osj("/content/dataset_training", "dataset_domain_adapted")
TRIO_PATH = osj("/content/dataset_training", "dataset_beat_trios_domain_adapted")
DICT_PATH = osj("/content/dictionaries", "dictionaries_5min_sorted")
LOAD_PATH = osj("/content/pretrained", "nets")
SAVE_PATH = osj("/content/savefolder")
max_epochs = [-1]
batch_sizes = [1024]
confidences = [0, *np.linspace(0.5, 1, 51)]

all_patient_cms = []
all_cms = []
all_weights = []
all_confs = []
repeats = 10

for repeat in range(repeats):
    patient_cms = {}
    confs = []
    cm = torch.zeros(2, 2)
    
    for i, patient_id in enumerate(valid_patients):
        dataset = load_N_channel_dataset(patient_id, DATASET_PATH, TRIO_PATH)
        train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()
        
        # For consulting through error energy.
        D, F = load_dictionary(patient_id, DICT_PATH)
        D, F = torch.Tensor(D), torch.Tensor(F)

        ## Consulting Exponential - Gaussian.
        BF = BayesianFit()
        EF = ExponentialFit()
        GF = GaussianFit()

        # Train error.
        train_E, E_healthy, E_arrhyth = get_error_one_patient(train_X[:, 0, :].squeeze(), F, y=train_y, as_energy=True)
        # _, E_healthy, E_arrhyth = get_error_per_patient(train_X[:, 0, :].squeeze(), ids=train_ids, DICT_PATH=DICT_PATH, y=train_y, as_energy=True)
        
        EF.fit(E_healthy)
        GF.fit(E_arrhyth)
        consult_train_y = torch.Tensor(BF.predict(train_E, EF, GF) <= 0.5).long()
                
        # Validation error.
        val_E, val_E_healthy, val_E_arrhyth = get_error_one_patient(val_X[:, 0, :].squeeze(), F, y=val_y, as_energy=True)
        
        EF.fit(val_E_healthy)
        GF.fit(val_E_arrhyth)
        consult_val_y = torch.Tensor(BF.predict(val_E, EF, GF) <= 0.5).long()
        
        # Test Error (be careful, we check (<= 0.5) because EF => healthy => label 0)
        test_E = get_error_one_patient(test_X[:, 0, :].squeeze(), F, as_energy=True)
        
        EF.fit(E_healthy)
        GF.fit(E_arrhyth)
        consult_test_y = torch.Tensor(BF.predict(test_E, EF, GF) <= 0.5).long()
        ##

        # Load the neural network.
        model = get_base_model(in_channels=train_X.shape[1])
        model = model.to("cuda")
        crit = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(params=model.parameters())
        
        net = NeuralNetwork.load_class(osj(LOAD_PATH, f"net_{repeat+1}_{patient_id}"), model, optim, crit)
        weight_checkpoint_val_loss = net.cbmanager.callbacks[1]  # <- this needs to change in case weight checkpoint is not the second callback.
        
        net.load_weights(weight_checkpoint_val_loss)
        
        # Validation predictions and probabilities.
        pred_y = net.predict(val_X, batch_size=1024, use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
        prob_y = net.predict_proba(val_X).cpu()
        softmax_prob_y = Func.softmax(prob_y, dim=1).max(dim=1).values
        
        # Choose consult threshold from validation set that maximizes F1.
        maxF1 = float("-inf")
        secondMaxF1 = float("-inf")
        maxConf = -1
        secondMaxConf = -1
        for conf in confidences:
            low_confidence = softmax_prob_y < conf
            high_confidence = softmax_prob_y >= conf

            final_pred_y = torch.Tensor(np.select([low_confidence, high_confidence], [consult_val_y, pred_y])).long()
            val_cm = get_confusion_matrix(final_pred_y, val_y, pos_is_zero=False)
            f1 = get_performance_metrics(val_cm)["f1"]
            f1 = np.nan_to_num(f1)
            
            if f1 >= (maxF1 - 1e-3):
                secondMaxF1 = maxF1
                secondMaxConf = maxConf
                maxF1 = f1
                maxConf = conf
        
        confs.append(maxConf)
        
        # Test predictions and probabilities.
        pred_y = net.predict(test_X, batch_size=1024, use_cuda=True, fits_gpu=True, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
        prob_y = net.predict_proba(test_X).cpu()
        softmax_prob_y = Func.softmax(prob_y, dim=1).max(dim=1).values
        
        # Use the confidence chosen above instead of iterating all confidences.
        for conf in [maxConf]:
            low_confidence = softmax_prob_y < conf
            high_confidence = softmax_prob_y >= conf

            final_pred_y = torch.Tensor(np.select([low_confidence, high_confidence], [consult_test_y, pred_y])).long()
            cm_exp = get_confusion_matrix(final_pred_y, test_y, pos_is_zero=False)

            patient_cms[patient_id] = cm_exp
            cm += cm_exp
            
        print_progress(i + 1, len(valid_patients), opt=[f"{patient_id}"])
        
    all_patient_cms.append(patient_cms)
    all_cms.append(cm)
    all_confs.append(confs)

config = dict(
    learning_rate=0.001,
    max_epochs=max_epochs[0],
    batch_size=batch_sizes[0],
    optimizer=optim.__class__.__name__,
    loss=crit.__class__.__name__,
    early_stopping="true",
    checkpoint_on=weight_checkpoint_val_loss.tracked,
    dataset="default+trio",
    info="Results replicated for GitHub, DA + Ensemble + Validation C."
)

get_performance_metrics(torch.stack(all_cms).sum(dim=0))

if False:
    with open(osj(SAVE_PATH, "cms.pkl"), "wb") as f:
        pickle.dump(all_cms, f)
        
    with open(osj(SAVE_PATH, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
        
    with open(osj(SAVE_PATH, "patient_cms.pkl"), "wb") as f:
        pickle.dump(all_patient_cms, f)
        
    with open(osj(SAVE_PATH, "confidences.pkl"), "wb") as f:
        pickle.dump(all_confs, f)