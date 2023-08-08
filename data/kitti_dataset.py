import os

from data.base_dataset import BaseDataset
from data.folder_dataset import make_dataset

class KittiDataset(BaseDataset):

    def get_data(self, opt, phase="train", from_vid=False):
        assert not from_vid
        root = opt.dataroot
        name = "all_vid" if opt.load_all else "vid"
        if opt.true_dim != 375:
            self.frame_folder = os.path.join(root, f'{name}_{opt.true_dim}')
            self.layout_folder = os.path.join(root, f'{name}_{opt.lyt_model}_{opt.true_dim}')
            self.flow_folder = os.path.join(root, f'{name}_{opt.flow_model}_{opt.true_dim}')
        else:
            self.frame_folder = os.path.join(root, f'{name}')
            self.layout_folder = os.path.join(root, f'{name}_{opt.lyt_model}')
            self.flow_folder = os.path.join(root, f'{name}_{opt.flow_model}')
        if opt.flow_dim != 0:
            # load specific dim for flow
            self.flow_folder = os.path.join(root, f'{name}_{opt.flow_model}_{opt.flow_dim}')

        if phase == 'train' or phase == 'valid':
            frame_dir = os.path.join(self.frame_folder, 'train')
            frame_paths = make_dataset(frame_dir, recursive=True)
        else:
            frame_dir = os.path.join(self.frame_folder, 'test')
            frame_paths = make_dataset(frame_dir, recursive=True)

        frame_dic = {}
        for path in sorted(frame_paths):
            seq = path.split("/")[-4]
            if seq in frame_dic:
                frame_dic[seq].append(path)
            else:
                frame_dic[seq] = [path]

        vid_frame_paths = [sorted(paths) for paths in list(frame_dic.values())]

        if phase == 'train' or phase == 'valid':
            split = int(0.1 * len(vid_frame_paths))
            vid_frame_paths = vid_frame_paths[split:] if phase == 'train' else vid_frame_paths[:split]
        frame_paths = [p for vid in vid_frame_paths for p in vid]

        # split videos into smaller clips
        new_vid_frame_paths = []
        if phase == 'train' or phase == 'valid':
            n = 20
            for paths in vid_frame_paths:
                chunks = len(paths) // n
                for k in range(chunks):
                    start = k * n
                    end = start + n
                    if k < chunks - 1:
                        new_vid_frame_paths.append(paths[start:end])
                    else:
                        new_vid_frame_paths.append(paths[start:])
        else:
            for paths in vid_frame_paths:
                for k in range(1, len(paths) - opt.vid_len):
                    new_vid_frame_paths.append(paths[k: k + opt.vid_len])
        vid_frame_paths = new_vid_frame_paths
        if "demo" in root:
            vid_frame_paths = vid_frame_paths[:1]

        return {"frame_paths": frame_paths, "vid_frame_paths": vid_frame_paths}
