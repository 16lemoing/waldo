import os

from data.base_dataset import BaseDataset
from data.folder_dataset import make_dataset

class CityscapesDataset(BaseDataset):

    def get_data(self, opt, phase="train", from_vid=False):
        assert not from_vid
        root = opt.dataroot
        if opt.true_dim != 1024:
            self.frame_folder = os.path.join(root, f'leftImg8bit_sequence_{opt.true_dim}')
            self.layout_folder = os.path.join(root, f'leftImg8bit_sequence_{opt.lyt_model}_{opt.true_dim}')
            self.flow_folder = os.path.join(root, f'leftImg8bit_sequence_{opt.flow_model}_{opt.true_dim}')
        else:
            self.frame_folder = os.path.join(root, f'leftImg8bit_sequence')
            self.layout_folder = os.path.join(root, f'leftImg8bit_sequence_{opt.lyt_model}')
            self.flow_folder = os.path.join(root, f'leftImg8bit_sequence_{opt.flow_model}')
        if opt.flow_dim != 0:
            # load specific dim for flow
            self.flow_folder = os.path.join(root, f'leftImg8bit_sequence_{opt.flow_model}_{opt.flow_dim}')

        if phase == 'train' or phase == 'valid':
            frame_dir = os.path.join(self.frame_folder, 'train')
            frame_paths = make_dataset(frame_dir, recursive=True)
        else:
            frame_dir = os.path.join(self.frame_folder, 'val')
            frame_paths = make_dataset(frame_dir, recursive=True)

        frame_dic = {}
        for path in sorted(frame_paths):
            seq ='_'.join(os.path.basename(path).split('_')[:2])
            if seq in frame_dic:
                frame_dic[seq].append(path)
            else:
                frame_dic[seq] = [path]

        vid_frame_paths = list(frame_dic.values())
        new_paths = []
        vid_len = opt.vid_len if opt.load_vid_len is None else opt.load_vid_len
        for l in vid_frame_paths:
            if len(l) == 30 or len(l) == 29:
                new_paths.append(l)
            else:
                seq = [l[0]]
                curr = int(os.path.basename(l[0]).split('_')[2])
                for i in range(len(l) - 1):
                    next = int(os.path.basename(l[i + 1]).split('_')[2])
                    if next == curr + 1:
                        seq.append(l[i + 1])
                    else:
                        if len(seq) >= vid_len:
                            new_paths.append(seq)
                        seq = [l[i + 1]]
                    curr = next
        vid_frame_paths = new_paths

        if phase == 'train' or phase == 'valid':
            split = int(0.9 * len(vid_frame_paths))
            vid_frame_paths = vid_frame_paths[:split] if phase == 'train' else vid_frame_paths[split:]
        frame_paths = [p for vid in vid_frame_paths for p in vid]
        return {"frame_paths": frame_paths, "vid_frame_paths": vid_frame_paths}
