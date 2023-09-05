from ecg_utilities import *
from NeuralPCI import get_base_model_neural
import torch.nn.functional as Func

from pytorch_sklearn import NeuralNetwork
from pytorch_sklearn.callbacks import WeightCheckpoint, Verbose, LossPlot, EarlyStopping, Callback, CallbackInfo
from pytorch_sklearn.utils.func_utils import to_safe_tensor



patient_ids = pd.read_csv(osj("files", "patient_ids.csv"), header=None).to_numpy().reshape(-1)
valid_patients = pd.read_csv(osj("files", "valid_patients.csv"), header=None).to_numpy().reshape(-1)
DATASET_PATH = osj("dataset_training", "dataset_domain_adapted")
TRIO_PATH = osj("dataset_training", "dataset_beat_trios_domain_adapted")
DICT_PATH = osj("dictionaries", "dictionaries_5min_sorted")
SAVE_PATH = osj("savefolder")
max_epochs = [-1]
batch_sizes = [1024]

all_patient_cms = []
all_cms = []
all_weights = []
repeats = 10

for repeat in range(repeats):
    patient_cms = {}
    cm = torch.zeros(2, 2)
    
    for i, patient_id in enumerate(valid_patients):
        dataset = load_N_channel_dataset(patient_id, DATASET_PATH, TRIO_PATH)
        train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids = dataset.values()

        # Train the neural network.
        model = get_base_model_neural(in_channels=train_X.shape[1])
        crit = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(params=model.parameters())
        
        net = NeuralNetwork(model, optim, crit)
        weight_checkpoint_val_loss = WeightCheckpoint(tracked="val_loss", mode="min")
        early_stopping = EarlyStopping(tracked="val_loss", mode="min", patience=15)

        net.fit(
            train_X=train_X,
            train_y=train_y,
            validate=True,
            val_X=val_X,
            val_y=val_y,
            max_epochs=max_epochs[0],
            batch_size=batch_sizes[0],
            use_cuda=True,
            fits_gpu=True,
            callbacks=[weight_checkpoint_val_loss, early_stopping],
        )

        net.load_weights(weight_checkpoint_val_loss)
        pred_y = net.predict(test_X, decision_func=lambda pred_y: pred_y.argmax(dim=1)).cpu()
        
        ## In order to save the trained weights
        NeuralNetwork.save_class(net, osj(SAVE_PATH, "nets", f"net_{repeat+1}_{patient_id}"))

        cur_cm = get_confusion_matrix(pred_y, test_y, pos_is_zero=False)
        patient_cms[patient_id] = cur_cm
        cm += cur_cm
            
        print_progress(i + 1, len(valid_patients), opt=[f"{patient_id}"])
        
    all_patient_cms.append(patient_cms)
    all_cms.append(cm)

config = dict(
    learning_rate=0.001,
    max_epochs=max_epochs[0],
    batch_size=batch_sizes[0],
    optimizer=optim.__class__.__name__,
    loss=crit.__class__.__name__,
    early_stopping="true",
    checkpoint_on=weight_checkpoint_val_loss.tracked,
    dataset="default+trio",
    info="2-channel run, domain adapted, consulting with default dictionary, and trying all thresholds, saves weights"
)

if False:
    with open(osj(SAVE_PATH, "cms.pkl"), "wb") as f:
        pickle.dump(all_cms, f)
        
    with open(osj(SAVE_PATH, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
        
    with open(osj(SAVE_PATH, "patient_cms.pkl"), "wb") as f:
        pickle.dump(all_patient_cms, f)