# """chaos_dataset dataset."""

from pathlib import Path
import random
import keras
import tensorflow_datasets as tfds
import pydicom
import tensorflow as tf
import numpy as np
from PIL import Image

import cv2

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for chaos_dataset dataset."""

    VERSION = tfds.core.Version('1.0.12')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    base_path: Path = Path(__file__).parent / "data"


    def load_dicom(self, path: Path):

        mri_img = pydicom.dcmread(path).pixel_array

        equalized_mri_img = cv2.createCLAHE (clipLimit = 2.0, tileGridSize = (4, 4)).apply(mri_img)

        img_o = np.float32(equalized_mri_img.copy())
        m = np.mean(img_o)
        s = np.std(img_o)
        normalized_img = np.divide((img_o - m), s)

        return np.expand_dims(normalized_img, 2)


    # Preprocessing
    def normalize_mask(self, path: Path):
        with Image.open(path) as im:
            a = np.array(im)
            
            a[ a == 63 ] = 1
            a[ a == 126 ] = 2
            a[ a == 189 ] = 3
            a[ a == 252 ] = 4

            return np.expand_dims(a, 2)

    def _info(self):
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(
                    shape=(None, None, 1), dtype=tf.float32
                ),
                "segmentation_mask": tfds.features.Image(
                    shape=(None, None, 1)
                ),
            }),
            supervised_keys=("image", "segmentation_mask"),
            homepage="https://zenodo.org/records/3431873#.ZSKsIuxBy3I",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns splits."""

        if not self.base_path.exists():
            dataset_dir = dl_manager.download_and_extract(
            'https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1'
            )

            self.base_path = Path(dataset_dir) / "Train_Sets"

        # 60/20/20 split
        train_list = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22]
        validate_list = [31, 32, 33, 34]
        test_list = [36, 37, 38, 39]

        # Setup train and test splits
        train_split = tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "patients_list": train_list,
            },
        )
        validate_split = tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "patients_list": validate_list,
            },
        )
        test_split = tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "patients_list": test_list,
            },
        )

        return [train_split, validate_split, test_split]
    
    def _generate_examples(self, patients_list: list[int]):
        # path_base = "MR/1/T1DUAL/DICOM_anon/InPhase"    # pari      [2,4...]
        # path_base = "MR/1/T1DUAL/DICOM_anon/OutPhase"   # dispari   [1,3...]
        # path_base = "MR/1/T1DUAL/Ground"                # pari      [2,4...]

        # path_base = "MR/1/T2SPIR/DICOM_anon"
        # path_base = "MR/1/T2SPIR/Ground"


        # extra = "" if "T2SPIR" else "InPhase" + "OutPhase"
        # dcm_base  = "MR/{p_num}/{dicom_type}/DICOM_anon{extra}/{dicom_name}.dcm"   # IMG-0004-00004.dcm
        # mask_base = "MR/{p_num}/{dicom_type}/Ground/{dicom_name}.png"              # IMG-0004-00004.png

        mr_base = self.base_path / "MR"

        for patient in patients_list:
            base_folder = mr_base / str(patient)

            t2spir_folder = base_folder / "T2SPIR" / "DICOM_anon"
            for dcm in t2spir_folder.iterdir():
                yield random.getrandbits(256), {
                    "image": self.load_dicom(dcm),
                    "segmentation_mask": self.normalize_mask(dcm.parent.parent / "Ground" / f"{dcm.stem}.png")
                }

            inPhase_folder =  base_folder / "T1DUAL" / "DICOM_anon" / "InPhase"
            for dcm in inPhase_folder.iterdir():
                yield random.getrandbits(256), {
                    "image": self.load_dicom(dcm),
                    "segmentation_mask": self.normalize_mask(dcm.parent.parent.parent / "Ground" / f"{dcm.stem}.png")
                }

            outPhase_folder = base_folder / "T1DUAL" / "DICOM_anon" / "OutPhase"
            for dcm in outPhase_folder.iterdir():
                mask_name = "-".join(dcm.stem.split("-")[:-1]) + "-" + str(1+int(dcm.stem.split("-")[-1])).zfill(5)

                yield random.getrandbits(256), {
                    "image": self.load_dicom(dcm),
                    "segmentation_mask": self.normalize_mask(dcm.parent.parent.parent / "Ground" / f"{mask_name}.png")
                }