import os
import logging
import numpy as np
from tqdm import tqdm
import math
from skimage import measure
import open3d as o3d
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
from lightconvpoint.datasets.data import Data
from lightconvpoint.datasets.dataset import get_dataset
import lightconvpoint.utils.transforms as lcp_T
from lightconvpoint.utils.logs import logs_file
from lightconvpoint.utils.misc import dict_to_device
from torch_geometric.data import Dataset
import networks

from networks.backbone.fkaconv_network import FKAConvNetwork


##################################################################


CONFIG_GEN = {
    "dataset_root":"/home/ad100000/project/POCO2/POCO/data/test_data/qml_2019_12_fi3",
    "save_dir": "/home/ad100000/project/POCO2/POCO/model/Cry2_cryosat",    
    "checkpoint_path":"/home/ad100000/project/POCO2/POCO/model/Cry2_cryosat/model.pth",
    "dataset_name": "Cryosat_test",
    "test_split": "validation",
    "manifold_points": -1,
    "non_manifold_points": -1,
    "network_latent_size": 64,
    "network_n_labels": 2,
    "threads": 1,
    "log_mode": "interactive",
    "logging": "INFO",
    "gen_resolution_global": 56,
    "gen_refine_iter": 1,
    "resume": False,
    "filter_name": None,
}


##################################################################

class Cryosat_test(Dataset):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, dataset_size=None, **kwargs):            
        super().__init__(root, transform, None)

        logging.info(f"Dataset  - Cryosat Test - Test only - {dataset_size}")
        self.root = root             
        self.filenames = []
        split_file = os.path.join(root, "testset.txt")

        with open(split_file) as f:
            content = f.readlines()
            content = [line.split("\n")[0] for line in content]
            content = [os.path.join(self.root, "04_pts", line) for line in content]
            self.filenames += content
        self.filenames.sort()

        if dataset_size is not None:
            self.filenames = self.filenames[:dataset_size]

        logging.info(f"Dataset - len {len(self.filenames)}")
        print("======= FILENAMES =======")
        for i, f in enumerate(self.filenames):
            print(f"[{i}] {f}")
        print("==========================")


        logging.info(f"Dataset - len {len(self.filenames)}")
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
        """Get item."""
        filename = self.filenames[idx]

        pts_shp = np.load(filename+".xyz.npy")


        pts_shp = torch.tensor(pts_shp, dtype=torch.float)
        pts_space = torch.ones((1,3), dtype=torch.float)
        occupancies = torch.ones((1,), dtype=torch.long)

        data = Data(x = torch.ones_like(pts_shp),
                    shape_id=idx, 
                    pos=pts_shp,
                    pos_non_manifold=pts_space, occupancies=occupancies, #
                    )

        return data

##################################################################

def export_mesh_and_refine_vertices_region_growing_v2(
    network,latent,
    resolution,
    padding=0,
    mc_value=0,
    device=None,
    num_pts=50000, 
    refine_iter=10, 
    simplification_target=None,
    input_points=None,
    refine_threshold=None,
    out_value=np.nan,
    step = None,
    dilation_size=2,
    whole_negative_component=False,
    return_volume=False
    ):

    bmin=input_points.min()
    bmax=input_points.max()

    if step is None:
        step = (bmax-bmin) / (resolution -1)
        resolutionX = resolution
        resolutionY = resolution
        resolutionZ = resolution
    else:
        bmin = input_points.min(axis=0)
        bmax = input_points.max(axis=0)
        resolutionX = math.ceil((bmax[0]-bmin[0])/step)
        resolutionY = math.ceil((bmax[1]-bmin[1])/step)
        resolutionZ = math.ceil((bmax[2]-bmin[2])/step)

    bmin_pad = bmin - padding * step
    bmax_pad = bmax + padding * step
    pts_ids = (input_points - bmin)/step + padding
    pts_ids = pts_ids.astype(int)

    volume = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), np.nan, dtype=np.float64)
    mask_to_see = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), True, dtype=bool)
    while(pts_ids.shape[0] > 0):

        # print("Pts", pts_ids.shape)
        mask = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), False, dtype=bool)
        mask[pts_ids[:,0], pts_ids[:,1], pts_ids[:,2]] = True

        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i,0])
            yc = int(pts_ids[i,1])
            zc = int(pts_ids[i,2])
            mask[max(0,xc-dilation_size):xc+dilation_size, 
                                 max(0,yc-dilation_size):yc+dilation_size,
                                 max(0,zc-dilation_size):zc+dilation_size] = True

        valid_points_coord = np.argwhere(mask).astype(np.float32)
        valid_points = valid_points_coord * step + bmin_pad

        z = []
        near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float, device=device)
        for pnts in tqdm(torch.split(near_surface_samples_torch,num_pts,dim=0), ncols=100, disable=True):

            latent["pos_non_manifold"] = pnts.unsqueeze(0)
            occ_hat = network.from_latent(latent)

            class_dim = 1
            occ_hat = torch.stack([occ_hat[:, class_dim] , occ_hat[:,[i for i in range(occ_hat.shape[1]) if i!=class_dim]].max(dim=1)[0]], dim=1)
            occ_hat = F.softmax(occ_hat, dim=1)
            occ_hat[:, 0] = occ_hat[:, 0] * (-1)
            if class_dim == 0:
                occ_hat = occ_hat * (-1)
            occ_hat = occ_hat.sum(dim=1)
            outputs = occ_hat.squeeze(0)
            z.append(outputs.detach().cpu().numpy())

        z = np.concatenate(z,axis=0)
        z  = z.astype(np.float64)
        volume[mask] = z
        mask_pos = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), False, dtype=bool)
        mask_neg = np.full((resolutionX+2*padding, resolutionY+2*padding, resolutionZ+2*padding), False, dtype=bool)

        
        for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
            xc = int(pts_ids[i,0])
            yc = int(pts_ids[i,1])
            zc = int(pts_ids[i,2])
            mask_to_see[xc,yc,zc] = False
            if volume[xc,yc,zc] <= 0:
                mask_neg[max(0,xc-dilation_size):xc+dilation_size, 
                                 max(0,yc-dilation_size):yc+dilation_size,
                                 max(0,zc-dilation_size):zc+dilation_size] = True
            if volume[xc,yc,zc] >= 0:
                mask_pos[max(0,xc-dilation_size):xc+dilation_size, 
                                 max(0,yc-dilation_size):yc+dilation_size,
                                 max(0,zc-dilation_size):zc+dilation_size] = True
        
        new_mask = (mask_neg & (volume>=0) & mask_to_see) | (mask_pos & (volume<=0) & mask_to_see)
        pts_ids = np.argwhere(new_mask).astype(int)

    volume[0:padding, :, :] = out_value
    volume[-padding:, :, :] = out_value
    volume[:, 0:padding, :] = out_value
    volume[:, -padding:, :] = out_value
    volume[:, :, 0:padding] = out_value
    volume[:, :, -padding:] = out_value
    maxi = volume[~np.isnan(volume)].max()
    mini = volume[~np.isnan(volume)].min()

    if not (maxi > mc_value and mini < mc_value):
        return None

    if return_volume:
        return volume

    verts, faces, _, _ = measure.marching_cubes(
            volume=volume.copy(),
            level=mc_value,
            )

    values = verts.sum(axis=1)
    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
    mesh.remove_vertices_by_mask(np.isnan(values))
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)


    if refine_iter > 0:

        dirs = verts - np.floor(verts)
        dirs = (dirs>0).astype(dirs.dtype)

        mask = np.logical_and(dirs.sum(axis=1)>0, dirs.sum(axis=1)<2)
        v = verts[mask]
        dirs = dirs[mask]
        v1 = np.floor(v)
        v2 = v1 + dirs

        v1 = v1.astype(int)
        v2 = v2.astype(int)
        preds1 = volume[v1[:,0], v1[:,1], v1[:,2]]
        preds2 = volume[v2[:,0], v2[:,1], v2[:,2]]

        v1 = v1.astype(np.float32)*step + bmin_pad
        v2 = v2.astype(np.float32)*step + bmin_pad

        # tmp mask
        mask_tmp = np.logical_and(
                        np.logical_not(np.isnan(preds1)),
                        np.logical_not(np.isnan(preds2))
                        )
        v = v[mask_tmp]
        dirs = dirs[mask_tmp]
        v1 = v1[mask_tmp]
        v2 = v2[mask_tmp]
        mask[mask] = mask_tmp

        verts = verts * step + bmin_pad
        v = v * step + bmin_pad

        for iter_id in tqdm(range(refine_iter), ncols=50, disable=True):

            # print(f"iter {iter_id}")

            preds = []
            pnts_all = torch.tensor(v, dtype=torch.float, device=device)
            for pnts in tqdm(torch.split(pnts_all,num_pts,dim=0), ncols=100, disable=True):

                
                latent["pos_non_manifold"] = pnts.unsqueeze(0)
                occ_hat = network.from_latent(latent)

                class_dim = 1
                occ_hat = torch.stack([occ_hat[:, class_dim] , occ_hat[:,[i for i in range(occ_hat.shape[1]) if i!=class_dim]].max(dim=1)[0]], dim=1)
                occ_hat = F.softmax(occ_hat, dim=1)
                occ_hat[:, 0] = occ_hat[:, 0] * (-1)
                if class_dim == 0:
                    occ_hat = occ_hat * (-1)
                # occ_hat = -occ_hat.sum(dim=1)
                occ_hat = occ_hat.sum(dim=1)
                outputs = occ_hat.squeeze(0)
                # outputs = network.predict_from_latent(latent, pnts.unsqueeze(0), with_sigmoid=True)
                # outputs = outputs.squeeze(0)
                preds.append(outputs.detach().cpu().numpy())
            preds = np.concatenate(preds,axis=0)

            mask1 = (preds*preds1)>0
            v1[mask1] = v[mask1]
            preds1[mask1] = preds[mask1]

            mask2 = (preds*preds2)>0
            v2[mask2] = v[mask2]
            preds2[mask2] = preds[mask2]

            v = (v2 + v1)/2

            verts[mask] = v

            if refine_threshold is not None:
                mask_vertices = (np.linalg.norm(v2 - v1, axis=1) > refine_threshold)
                # print("V", mask_vertices.sum() , "/", v.shape[0])
                v = v[mask_vertices]
                preds1 = preds1[mask_vertices]
                preds2 = preds2[mask_vertices]
                v1 = v1[mask_vertices]
                v2 = v2[mask_vertices]
                mask[mask] = mask_vertices

                if v.shape[0] == 0:
                    break

    else:
        verts = verts * step + bmin_pad


    o3d_verts = o3d.utility.Vector3dVector(verts)
    o3d_faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

    if simplification_target is not None and simplification_target > 0:
        mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, simplification_target)

    return mesh

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(config):    
    
    config = eval(str(config))  
    logging.getLogger().setLevel(config["logging"])
    disable_log = (config["log_mode"] != "interactive")
    device = torch.device("cuda")
    savedir_root = config["save_dir"]


    N_LABELS = config["network_n_labels"]
    latent_size = config["network_latent_size"]
    backbone = "FKAConv"
    decoder = {'name': "InterpAttentionKHeadsNet", 'k': 64}
    

    logging.info("Creating the network")
    def network_function():
        return networks.Network(3, latent_size, N_LABELS, backbone, decoder)
    net = network_function()
    checkpoint = torch.load(config["checkpoint_path"])
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device)
    net.eval()
    logging.info(f"Network -- Number of parameters {count_parameters(net)}")

    
    logging.info("Getting the dataset")
    DatasetClass = get_dataset(Cryosat_test)
    test_transform = []

    if config["manifold_points"] is not None and config["manifold_points"] > 0:
        test_transform.append(lcp_T.FixedPoints(config["manifold_points"], item_list=["x", "pos"]))
    test_transform.append(lcp_T.FixedPoints(1, item_list=["pos_non_manifold", "occupancies"]))

    test_transform = test_transform + [
                                            lcp_T.Permutation("pos", [1,0]),
                                            lcp_T.Permutation("pos_non_manifold", [1,0]),
                                            lcp_T.Permutation("x", [1,0]),
                                            lcp_T.ToDict(),]
    test_transform = T.Compose(test_transform)

    gen_dataset = DatasetClass(config["dataset_root"],
                split=config["test_split"], 
                transform=test_transform, 
                network_function=network_function, 
                filter_name=config["filter_name"], 
                num_non_manifold_points=config["non_manifold_points"],
                dataset_size=config.get("num_mesh")
                )

    gen_loader = torch.utils.data.DataLoader(
        gen_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    
    with torch.no_grad():

        gen_dir = f"output_{config['dataset_name']}"
     #   savedir_root = os.path.join(savedir_root, gen_dir)
        savedir_root = os.path.join(config["save_dir"], gen_dir)

        os.makedirs(savedir_root, exist_ok=True)

        log_file_path = os.path.join(savedir_root, "generation_info.txt")
        log_file = open(log_file_path, "w")
        log_file.write("Network output info Log\n")
        log_file.write(f"Network: Backbone = {backbone}, Decoder = {decoder['name']}, Parameters = {count_parameters(net)}\n\n")
        log_file.write("PatchID | InputFile | #InputPoints | OutputFile | #OutputPoints\n")


        for data in tqdm(gen_loader, ncols=100):

            shape_id = data["shape_id"].item()
            category_name = gen_dataset.get_category(shape_id)
            object_name = gen_dataset.get_object_name(shape_id)
            print(f"{shape_id} | {category_name} - {object_name} - {data['pos'].shape}")
            data = dict_to_device(data, device)
            input_pts = data["pos"][0].transpose(1, 0).cpu().numpy()


            scale = 1
            data["pos"] = data["pos"] * scale
            latent = net.get_latent(data, with_correction=False)
            step = None
            resolution = config["gen_resolution_global"]


            print("POS", data["pos"].shape)
            mesh = export_mesh_and_refine_vertices_region_growing_v2(
                net, latent,
                resolution=resolution,
                padding=1,
                mc_value=0,
                device=device,
                input_points=data["pos"][0].cpu().numpy().transpose(1,0),
                refine_iter=config["gen_refine_iter"],
                out_value=1,
                step=step
            )

            
            if mesh is not None:

                vertices = np.asarray(mesh.vertices)
                vertices = vertices / scale
                save_path_npy = os.path.join(savedir_root, object_name + ".npy")
                np.save(save_path_npy, vertices.astype(np.float32))
                print(f"Saved {vertices.shape[0]} points to {save_path_npy}")
              #  log_file.write(f"{shape_id} | {object_name}.xyz | {pts.shape[0]} | {object_name}.npy | {vertices.shape[0]}\n")
                log_file.write(f"{shape_id} | {object_name}.xyz | {input_pts.shape[0]} | {object_name}.npy | {vertices.shape[0]}\n")

            else:
                logging.warning("Patch is None")

        log_file.close()
        print(f"Log saved to {log_file_path}")


def replace_values_of_config(config, config_update):

    for key, value in config_update.items():
        if key not in config:
            print(f"replace warning unknown key '{key}'")
            continue
        if isinstance(value, dict):
            config[key] = replace_values_of_config(config[key], value)
        else:
            config[key] = value
    return config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(CONFIG_GEN)