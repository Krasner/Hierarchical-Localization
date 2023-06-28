from pathlib import Path
import yaml
import numpy as np
import h5py
import pycolmap
import cv2
from itertools import combinations, permutations
from collections import defaultdict
from tqdm import tqdm
from hloc import reconstruction

from third_party.SuperRetina.predictor import Predictor

class SuperRetinaSFM:
    def __init__(
        self,
        image_dir: Path,
        output_dir: Path,
        super_retina_cfg_path: str,
        super_retina_model_path: str,
        extension: str = "JPG",
    ):
        self.image_dir = image_dir
        self.image_files = list(image_dir.glob(f"*.{extension}"))

        cfg = yaml.safe_load(
            open(Path(super_retina_cfg_path),"rb")
        )

        cfg["PREDICT"]["model_save_path"] = super_retina_model_path
        cfg["PREDICT"]["knn_thresh"] = 0.7
        self.sr_model = Predictor(cfg)

        if not output_dir.exists():
            output_dir.mkdir()

        self.feature_path = output_dir / "super_retina_features.h5"
        self.match_path = output_dir / "super_retina_matches.h5"

        self.sfm_pairs = output_dir / "pairs.txt"
        self.sfm_dir = output_dir / "colmap"

        self.mask_dir = output_dir / "masks"
        if not self.mask_dir.exists():
            self.mask_dir.mkdir()

        self.misc_dir = output_dir / "misc"
        if not self.misc_dir.exists():
            self.misc_dir.mkdir()

    def extract_features(self):

        self._images = dict()
        # extract keypoints
        keypoints = dict()
        descriptors = dict()
        preds = dict()
        for img_file in tqdm(self.image_files, desc="Keypoint Extraction"):
            kp, desc, score, cv_kp, _img, _mask = self.sr_model.model_run_one_image(str(img_file), compute_mask=True)

            filename = str(img_file).split('/')[-1]

            keypoints[filename] = kp
            descriptors[filename] = desc

            preds[filename] = dict()
            # preds[filename]["keypoints"] = kp.numpy().astype(np.float16)
            preds[filename]["keypoints"] = np.array([k.pt for k in cv_kp]).astype(np.float16)
            preds[filename]["descriptors"] = desc.numpy().astype(np.float16)
            preds[filename]["image_size"] = np.array([self.sr_model.image_width, self.sr_model.image_height]).astype(np.int16)
            preds[filename]["score"] = score.numpy().astype(np.float16)

            self._images[filename] = _img.copy()

            cv2.imwrite(str(self.mask_dir / f"{filename}.png"), _mask)

        # write to h5
        with h5py.File(str(self.feature_path), 'a', libver='latest') as fd:
            for name in preds:
                # print(name)
                try:
                    if name in fd:
                        del fd[name]
                    grp = fd.create_group(name)
                    print(f"{name} - number of kp: {len(preds[name]['keypoints'])}")
                    for k, v in preds[name].items():
                        # print(k)
                        grp.create_dataset(k, data=v)
                    # if 'keypoints' in preds:
                    #     grp['keypoints'].attrs['uncertainty'] = uncertainty
                except OSError as error:
                    if 'No space left on device' in error.args[0]:
                        print(
                            'Out of disk space: storing features on disk can take '
                            'significant space, did you enable the as_half flag?')
                        del grp, fd[name]
                    raise error

        return keypoints, descriptors

    def match_pairs(self, keypoints, descriptors):
        # n(n-1)//2 paired images
        pairs = permutations(list(keypoints.keys()), 2)

        # find matches and force into pycolmap format
        matches = dict()
        pair_ids = []
        for k0, k1 in tqdm(pairs, desc="Keypoint Matching"):
            
            goodMatch, inliers_num_rate, *_ = self.sr_model.match_pair(
                keypoints[k1], keypoints[k0], 
                descriptors[k1], descriptors[k0], 
                # query_image=self._images[k1], refer_image=self._images[k0],
                # save_path=str(self.misc_dir),
                # save_name="_".join([k1,k0,"matches"]) + ".png"
            )

            print(f"{k0} -> {k1} : {inliers_num_rate=}")

            # need at least 8 points for fundamental matrix estimation
            if len(goodMatch) >= 4 and (inliers_num_rate >= 0.67):

                if k0 not in matches:
                    matches[k0] = defaultdict(dict)

                pair_ids.append((k0, k1))

                match_idx = np.array([[m.trainIdx, m.queryIdx] for m in goodMatch])
                # match_idx = np.array([[m.queryIdx, m.trainIdx] for m in goodMatch])

                no_matches = []
                for k in range(len(keypoints[k0])):
                    if k not in list(match_idx[:,0]):
                        no_matches.append([k, -1])

                match_idx = np.concatenate((match_idx, np.array(no_matches)), 0)
                match_idx = match_idx[match_idx[:,0].argsort()]

                matches[k0][k1] = {"matches0": match_idx[:,1].astype(np.int32), "matching_scores0": np.zeros_like(match_idx[:,1]).astype(np.float16)}
        
        with h5py.File(str(self.match_path), 'a', libver='latest') as fd:
            for name in matches:
                print(name)
                try:
                    if name in fd:
                        del fd[name]
                    grp = fd.create_group(name)
                    for k, v in matches[name].items():
                        inner_grp = grp.create_group(k)
                        for _k, _v in v.items():
                            # print(k)
                            inner_grp.create_dataset(_k, data=_v)
                    # if 'keypoints' in preds:
                    #     grp['keypoints'].attrs['uncertainty'] = uncertainty
                except OSError as error:
                    if 'No space left on device' in error.args[0]:
                        print(
                            'Out of disk space: storing features on disk can take '
                            'significant space, did you enable the as_half flag?')
                        del grp, fd[name]
                    raise error
        
        _pair_ids = [" ".join(p) + "\n" for p in pair_ids]
        with open(str(self.sfm_pairs),"w") as f:
            f.writelines(_pair_ids)

    def reconstruct(self, image_opts, mapper_opts, undistort_opts, patch_match_opts, stereo_fusion_opts):
        kp, desc = self.extract_features()
        self.match_pairs(kp, desc)
        # run pycolmap from created h5 features and matches
        image_opts.update({"mask_path": str(self.mask_dir)})
        stereo_fusion_opts.update({"mask_path": str(self.mask_dir)})

        model = reconstruction.main(
            self.sfm_dir, 
            self.image_dir, 
            self.sfm_pairs, 
            self.feature_path, 
            self.match_path,
            camera_mode = pycolmap.CameraMode.PER_IMAGE,
            verbose=True,
            dense_reconstruction=True,
            image_options=image_opts,
            mapper_options=mapper_opts,
            undistort_options=undistort_opts,
            patch_match_options=patch_match_opts,
            stereo_fusion_options=stereo_fusion_opts,
        )

        return model

if __name__ == "__main__":
    sfm = SuperRetinaSFM(
        image_dir=Path("/home/ubuntu/fundus/images/"), # losses EXIF data after masking
        output_dir=Path("/home/ubuntu/fundus/sfm3/"),
        super_retina_cfg_path="./third_party/SuperRetina/config/test.yaml",
        super_retina_model_path="/home/ubuntu/Hierarchical-Localization/third_party/SuperRetina/save/SuperRetina.pth",
        extension="JPG",
    )

    image_opts = {
        "camera_model": "SIMPLE_RADIAL",
    }
    # image_opts = {}
    mapper_opts = {
        "min_num_matches": 4,
        "ignore_watermarks": True,
    }

    undistort_opts = {}
    patch_match_opts = {
        "window_radius": 11,
        "num_samples": 100,
    }
    stereo_fusion_opts = {}

    _ = sfm.reconstruct(image_opts, mapper_opts,undistort_opts, patch_match_opts, stereo_fusion_opts)