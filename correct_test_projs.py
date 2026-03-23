from tigre.utilities.io import VarianDataLoader
from tigre.algorithms.single_pass_algorithms import FDK
from tigre.utilities.crop_CBCT import cropCBCT
import pathlib
import numpy as np

pt_num = 4105896
mydir = "C:/CBCT_Scatter_Removal_Project/data/clinical/4105896/2021-08-09_121614/77f2251f-5813-4611-bcbd-4d52864d18ee/"

LOG_DIRS = []


def main():
    log_dir = pathlib.Path("C:/dev/scatter-unet/runs/20251212-070941")
    keras_model = log_dir / "ckpt" / "model.keras"
    for log_dir in LOG_DIRS:
        log_projs, geo, angles = VarianDataLoader(
            mydir, acdc=1, dps=1, fasks=0, keras_model=keras_model
        )
        fname = str(log_dir).split("\\")[-1] + f"_pt{pt_num}_cnn_projs.npy"
        np.savez(log_dir / fname, log_projs=log_projs, geo=geo, angles=angles)

        recon = FDK(log_projs, geo, angles, filter="shepp_logan")
        rec = cropCBCT(recon)
        rec[rec < 0] = 0

        fname_rec = str(log_dir).split("\\")[-1] + f"_pt{pt_num}_cnn_recon_sl.npy"
        np.save(log_dir / fname_rec, rec)
