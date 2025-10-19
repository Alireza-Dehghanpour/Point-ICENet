#---------------------------
import os
import numpy as np
import yaml
from tqdm import tqdm
import logging
import shutil
from sklearn.metrics import confusion_matrix
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
from lightconvpoint.datasets.dataset import get_dataset
import lightconvpoint.utils.transforms as lcp_T
from lightconvpoint.utils.logs import logs_file
from lightconvpoint.utils.misc import dict_to_device
#import utils.argparseFromFile as argparse
from utils.utils import wblue, wgreen
import utils.metrics as metrics
import networks
from torch.utils.tensorboard import SummaryWriter
import logging
from torch_geometric.data import Dataset
from lightconvpoint.datasets.data import Data
import os, logging, numpy as np, torch
from torch_geometric.data import Dataset
from lightconvpoint.datasets.data import Data


#---------------------------
class Cryosat(Dataset):
    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, dataset_size=None):
        super().__init__(root, transform, None)
        self.root = root

        txt_name = "trainset.txt" if split in ["train", "training"] else "valset.txt"
        with open(os.path.join(root, txt_name)) as f:
            ids = [ln.strip() for ln in f]

        self.filenames = [os.path.join(root, "surf_pts", _id) for _id in ids]
        if dataset_size:
            self.filenames = self.filenames[:dataset_size]
    def get_category(self, f_id):
        return self.filenames[f_id].split("/")[-2]

    def get_object_name(self, f_id):
        return self.filenames[f_id].split("/")[-1]

    def get_class_name(self, f_id):
        return self.metadata[self.get_category(f_id)]["name"]

    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return []
    def _download(self): pass
    def download(self): pass
    def process(self):pass
    def _process(self):pass
    def len(self):return len(self.filenames)
    def get(self, idx):
        filename = self.filenames[idx]
        pts_with_normals = np.load(filename + ".npy") 
        positions = pts_with_normals[:, :3]  
        filename = filename.replace("surf_pts", "query")
        pts_space = np.load(filename + ".npy")
        pts_space = pts_space[:, :3]
        filename = filename.replace("query", "query_label")
        occupancies = np.load(filename + ".npy")
        positions = torch.tensor(positions, dtype=torch.float)
        pts_space = torch.tensor(pts_space, dtype=torch.float)
        occupancies = torch.tensor(occupancies, dtype=torch.long)
        
        data = Data(x=torch.ones_like(positions),  
                    shape_id=idx, 
                    pos=positions,
                    pos_non_manifold=pts_space, occupancies=occupancies, 
                    )

        print("*************************After creating Data object:******************************")
        print("Shape of 'x':", data.x.shape)
        print("Shape of 'pos':", data.pos.shape)
        print("Shape of 'pos_non_manifold':", data.pos_non_manifold.shape)
        print("Shape of 'occupancies':", data.occupancies.shape)
        return data

#---------------------------

CONFIG = {
    "experiment_name": "cryosat",
    "dataset_name": "Cry2",
    "dataset_root": "data/train_data",
    "save_dir": "model",
    "train_split": "training",
    "val_split": "validation",
    "manifold_points": 5000,
    "non_manifold_points": 36000,
    "training_batch_size": 1,
    "training_num_epochs": 20,
    "training_lr_start": 1e-3,
    "val_interval": 5,
    "network_backbone": "FKAConv",
    "network_latent_size": 32,
    "network_decoder": "InterpAttentionKHeadsNet",
    "network_decoder_k": 64,
    "network_n_labels": 2,
    "device": "cuda",
    "threads": 1,
    "log_mode": "interactive",
    "logging": "INFO",
    "resume": False,
    "filter_name": None,
    "val_num_mesh": None,
    "normals": False,
}

#---------------------------main
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def save_config_file(config, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
def main(config):
    disable_log = (config["log_mode"] != "interactive")
    device = torch.device(config['device'])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True
    savedir_root = os.path.join(
    config["save_dir"],
    f"{config['dataset_name']}_{config['experiment_name']}"
     )
    N_LABELS = config["network_n_labels"]
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]
    decoder = {'name':config["network_decoder"], 'k': config['network_decoder_k']}
    def network_function():
        return networks.Network(3, latent_size, N_LABELS, backbone, decoder)


    net = network_function()
    net.to(device)
    DatasetClass = get_dataset(Cryosat)
    train_transform = []
    test_transform = []

    train_transform.append(lcp_T.FixedPoints(config["manifold_points"], item_list=["x", "pos"]))
    test_transform.append(lcp_T.FixedPoints(config["manifold_points"], item_list=["x", "pos"]))
    train_transform.append(lcp_T.FixedPoints(config["non_manifold_points"], item_list=["pos_non_manifold", "occupancies"]))
    test_transform.append(lcp_T.FixedPoints(config["non_manifold_points"], item_list=["pos_non_manifold", "occupancies"]))
    train_transform = train_transform + [
                                            lcp_T.Permutation("pos", [1,0]),
                                            lcp_T.Permutation("pos_non_manifold", [1,0]),
                                            lcp_T.Permutation("x", [1,0]),
                                            lcp_T.ToDict(),]
    test_transform = test_transform + [
                                            lcp_T.Permutation("pos", [1,0]),
                                            lcp_T.Permutation("pos_non_manifold", [1,0]),
                                            lcp_T.Permutation("x", [1,0]),
                                            lcp_T.ToDict(),]


    train_transform = T.Compose(train_transform)
    test_transform = T.Compose(test_transform)
    train_dataset = DatasetClass(config["dataset_root"], 
                split=config["train_split"], 
                transform=train_transform, 
                network_function=network_function, 
                filter_name=config["filter_name"], 
                num_non_manifold_points=config["non_manifold_points"]
                )
    test_dataset = DatasetClass(config["dataset_root"],
                split=config["val_split"], 
                transform=test_transform, 
                network_function=network_function, 
                filter_name=config["filter_name"], 
                num_non_manifold_points=config["non_manifold_points"],
                dataset_size=config["val_num_mesh"]
                )
    print(f"Number of non-manifold points (query points) used: {config['non_manifold_points']}")
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["training_batch_size"],
            shuffle=True,
            num_workers=config["threads"],
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training_batch_size"],
        shuffle=False,
        num_workers=config["threads"],
    )

    optimizer = torch.optim.Adam(net.parameters(),config["training_lr_start"])
    
    if config["resume"] and os.path.exists(savedir_root):
        checkpoint = torch.load(os.path.join(savedir_root, "checkpoint.pth"), map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"]
        train_iter_count = len(train_loader) * epoch_start

    else:
        if os.path.exists(savedir_root):
            shutil.rmtree(savedir_root)
        os.makedirs(savedir_root, exist_ok=True)
        epoch_start = 0
        train_iter_count = 0

    loss_layer = torch.nn.CrossEntropyLoss()
    epoch = epoch_start
    print(f"Starting Epoch: {epoch}")
    print(f"Starting training for {config['training_num_epochs']} epochs "
          f"with {len(train_loader)} iterations per epoch")


    for epoch in range(epoch_start, config["training_num_epochs"]):
        print(f"\n=== Epoch {epoch+1}/{config['training_num_epochs']} ===")

        net.train()
        error = 0
        cm = np.zeros((N_LABELS, N_LABELS))

        t = tqdm(
            train_loader,
            desc="Epoch " + str(epoch),
            ncols=130,
            disable=disable_log,
        )
        for data in t:

            data = dict_to_device(data, device)
            print("\n Shapes before training:")
            print(f"Manifold points (pos): {data['pos'].shape}")
            print(f"Non-manifold query points (pos_non_manifold): {data['pos_non_manifold'].shape}")
            print(f"Occupancy labels (occupancies): {data['occupancies'].shape}")
            print(" Done.\n")
            optimizer.zero_grad()
            outputs = net(data, spectral_only=True)
            occupancies = data["occupancies"]
            loss = loss_layer(outputs, occupancies)
            loss.backward()
            optimizer.step()
            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            target_np = occupancies.cpu().numpy()
            cm_ = confusion_matrix(
                target_np.ravel(), output_np.ravel(), labels=list(range(N_LABELS))
            )
            cm += cm_
            error += loss.item()
            train_oa = metrics.stats_overall_accuracy(cm)
            train_aa = metrics.stats_accuracy_per_class(cm)[0]
            train_iou = metrics.stats_iou_per_class(cm)[0]
            train_aloss = error / cm.sum()
            description = f"Epoch {epoch} | OA {train_oa*100:.2f} | AA {train_aa*100:.2f} | IoU {train_iou*100:.2f} | Loss {train_aloss:.4e}"
            t.set_description_str(wblue(description))
            print(f"Epoch {epoch} - Processing batch {t.n}/{len(train_loader)}")
            print(f"Epoch {epoch} - Batch {t.n} - Loss: {loss.item():.4f}, OA: {train_oa*100:.2f}%, AA: {train_aa*100:.2f}%")

            train_iter_count += 1


        print(f"Epoch {epoch+1} finished | "f"Train OA: {train_oa:.4f}, AA: {train_aa:.4f}, "f"IoU: {train_iou:.4f}, Loss: {train_aloss:.6e}")
	
        train_log_data = {
            "OA_train": train_oa,
            "IoU_train": train_iou,
            "Loss_train": train_aloss,
        }

        os.makedirs(savedir_root, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(savedir_root, "model.pth"),
        )

        logs_file(os.path.join(savedir_root, "logs_train_itr.csv"), train_iter_count, train_log_data)
        logs_file(os.path.join(savedir_root, "logs_train_epo.csv"), epoch + 1, train_log_data)

        #----------------------------------------------------val
        if (epoch+1)%config["val_interval"]==0:
            net.eval()
            error = 0
            cm = np.zeros((N_LABELS, N_LABELS))
            with torch.no_grad():
                t = tqdm(
                    test_loader,
                    desc="  Test " + str(epoch),
                    ncols=100,
                    disable=disable_log,
                )
                for data in t:
                    data = dict_to_device(data, device)
                    outputs = net(data, spectral_only=True)
                    occupancies = data["occupancies"]
                    loss = loss_layer(outputs, occupancies)
                    outputs = F.softmax(outputs, dim=1)
                    outputs_np = outputs.cpu().detach().numpy()
                    targets_np = occupancies.cpu().numpy()
                    pred_labels = np.argmax(outputs_np, axis=1)
                    cm_ = confusion_matrix(targets_np.ravel(), pred_labels.ravel(), labels=list(range(N_LABELS)))
                    cm += cm_
                    error += loss.item()
                    test_oa = metrics.stats_overall_accuracy(cm)
                    test_aa = metrics.stats_accuracy_per_class(cm)[0]
                    test_iou = metrics.stats_iou_per_class(cm)[0]
                    test_aloss = error / cm.sum()
    
                    print(f"Validation Epoch {epoch} - Processing batch {t.n}/{len(test_loader)}")
                    print(f"Validation Epoch {epoch} - Batch {t.n} - Loss: {loss.item():.4f}, OA: {test_oa*100:.2f}%, AA: {test_aa*100:.2f}%")


                    description = f"Val. {epoch}  | OA {test_oa*100:.2f} | AA {test_aa*100:.2f} | IoU {test_iou*100:.2f} | Loss {test_aloss:.4e}"
                    t.set_description_str(wgreen(description))
            
            val_log_data = {
                "OA_val": test_oa,
                "IoU_val": test_iou,
                "Loss_val": test_aloss,
            }
            logs_file(os.path.join(savedir_root, "logs_val_iter.csv"), train_iter_count, val_log_data)
            logs_file(os.path.join(savedir_root, "logs_val_epo.csv"), epoch + 1, val_log_data)

if __name__ == "__main__":
    main(CONFIG)