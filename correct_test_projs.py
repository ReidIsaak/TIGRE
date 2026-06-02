from tigre.utilities.io import VarianDataLoader
from tigre.algorithms.single_pass_algorithms import FDK
from tigre.utilities.crop_CBCT import cropCBCT
import pathlib
import numpy as np
import pickle

PT_ID = 4104266
PT_DIR = "C:/CBCT_Scatter_Removal_Project/data/clinical/4104266/2021-08-09_112508/578acb28-0f5d-4676-9a0e-8647d3138bfe/"


BASE_DIR = pathlib.Path("C:/dev/scatter-unet/runs/")
LOG_DIRS = [
    "20251210-024442",
    "20251210-141034",
    "20251210-204557",
    "20251211-081647",
    "20251211-194500",
    "20251212-070941",
    "20251212-134851",
    "20251212-202544",
]
FILTER = "ram_lak"
FILTER_STR = "ram"


def main():
    # cnn correction
    for log_dir in LOG_DIRS:
        print(f"\nCNN CORRECTION: {log_dir}\n")
        keras_model = BASE_DIR / log_dir / "ckpt" / "model.keras"
        log_projs, geo, angles = VarianDataLoader(
            PT_DIR, acdc=1, dps=1, fasks=0, keras_model=keras_model
        )
        fname = str(log_dir).split("\\")[-1] + f"_pt{PT_ID}_cnn_projs"
        np.savez(BASE_DIR / log_dir / fname, log_projs=log_projs, angles=angles)
        fname_geo = str(log_dir).split("\\")[-1] + f"_pt{PT_ID}_geo.pkl"
        with open(BASE_DIR / log_dir / fname_geo, "wb") as f:
            pickle.dump(geo, f)

        recon = FDK(log_projs, geo, angles, filter=FILTER)
        rec = cropCBCT(recon)
        rec[rec < 0] = 0

        fname_rec = str(log_dir).split("\\")[-1] + f"_pt{PT_ID}_cnn_recon_{FILTER_STR}.npy"
        np.save(BASE_DIR / log_dir / fname_rec, rec)

        log_projs = None
        geo = None
        angles = None
        recon = None
        rec = None

    print("\nNO CORRECTION\n")
    log_projs, geo, angles = VarianDataLoader(
        PT_DIR,
        acdc=1,
        dps=1,
        fasks=0,
    )
    fname = f"pt{PT_ID}_dps_projs"
    np.savez(BASE_DIR / fname, log_projs=log_projs, angles=angles)
    fname_geo = f"pt{PT_ID}_geo.pkl"
    with open(BASE_DIR / fname_geo, "wb") as f:
        pickle.dump(geo, f)

    recon = FDK(log_projs, geo, angles, filter=FILTER)
    rec = cropCBCT(recon)
    rec[rec < 0] = 0

    fname_rec = f"pt{PT_ID}_dps_recon_{FILTER_STR}.npy"
    np.save(BASE_DIR / fname_rec, rec)

    log_projs = None
    geo = None
    angles = None
    recon = None
    rec = None

    print("\nFASKS CORRECTION\n")
    log_projs, geo, angles = VarianDataLoader(
        PT_DIR,
        acdc=1,
        dps=1,
        fasks=1,
    )
    fname = f"pt{PT_ID}_fasks_projs"
    np.savez(BASE_DIR / fname, log_projs=log_projs, angles=angles)
    fname_geo = f"pt{PT_ID}_geo.pkl"
    with open(BASE_DIR / fname_geo, "wb") as f:
        pickle.dump(geo, f)

    recon = FDK(log_projs, geo, angles, filter=FILTER)
    rec = cropCBCT(recon)
    rec[rec < 0] = 0

    fname_rec = f"pt{PT_ID}_fasks_recon_{FILTER_STR}.npy"
    np.save(BASE_DIR / fname_rec, rec)

    log_projs = None
    geo = None
    angles = None
    recon = None
    rec = None


if __name__ == "__main__":
    main()
