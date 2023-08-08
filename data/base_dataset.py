import random
import PIL
import os
import math
import numpy as np
from copy import deepcopy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets.video_utils import VideoClips

from tools.utils import get_vprint, serialize, deserialize

class BaseDataset(data.Dataset):
    def __init__(self, opt, phase='train', fold=None, load_vid=False, load_rgb=True, verbose=True):
        super().__init__()
        self.opt = opt
        self.phase = phase
        self.from_vid = opt.from_vid
        self.from_animation = opt.from_animation
        self.load_vid = load_vid
        self.load_rgb = load_rgb

        vprint = get_vprint(verbose)

        data_path = self.get_path_to_serialized("data", phase, fold)

        if opt.load_data and os.path.exists(data_path):
            self.data = deserialize(data_path)
        else:
            self.data = self.get_data(opt, phase=phase, from_vid=opt.from_vid)
        if opt.save_data:
            vprint(f"Saving dataset paths and labels to {data_path}")
            serialize(data_path, self.data)

        if "vid_labels" in self.data and opt.categories is not None:
            cat_idx, n_cat = np.unique(self.data["vid_labels"], return_counts=True)
            summary = [f"{n_cat[idx]} '{opt.categories[i]}'" for idx, i in enumerate(cat_idx)]
            vprint("Found " + ", ".join(summary) + " video files")

        if opt.from_vid:
            metadata_path = self.get_path_to_serialized("metadata", phase, fold)
            if os.path.exists(metadata_path) and not opt.force_compute_metadata:
                vprint(f"Loading dataset metadata from {metadata_path}")
                metadata = deserialize(metadata_path)
                vprint(f"Metadata loaded")
                if metadata["video_paths"] != self.data["vid_paths"]:
                    vprint(f"Video paths have changed: initially {len(metadata['video_paths'])} files and now {len(self.data['vid_paths'])}")
                    vprint(f"Recomputing metadata")
                    metadata = None
            else:
                metadata = None

            if load_vid:
                frames_per_clip = self.opt.load_vid_len if (self.opt.load_vid_len is not None and self.phase == "train") else self.opt.vid_len
            else:
                frames_per_clip = 1

            self.vid_clips = self.get_clips("vid", frames_per_clip, metadata=metadata)

            if metadata is None:
                vprint(f"Saving dataset metadata to {metadata_path}")
                serialize(metadata_path, self.vid_clips.metadata)

            self.dataset_size = self.vid_clips.num_clips()
        else:
            self.dataset_size = len(self.data['vid_frame_paths']) if load_vid else len(self.data['frame_paths'])

        self.dim = opt.dim if opt.load_dim == 0 else opt.load_dim
        self.true_dim = opt.true_dim

    def get_path_to_serialized(self, data_str, phase, fold):
        if self.opt.data_specs is None:
            path = os.path.join(self.opt.dataroot, f"{phase}_{data_str}.pkl")
        else:
            if fold is None:
                path = os.path.join(self.opt.dataroot, f"{self.opt.data_specs}_{phase}_{data_str}.pkl")
            else:
                path = os.path.join(self.opt.dataroot, f"folds/{self.opt.data_specs}_{fold}_{phase}_{data_str}.pkl")
        if fold is not None:
            assert os.path.exists(path)
        return path

    def get_data(self, opt, phase, from_vid):
        assert False, "A subclass of BaseDataset must override self.get_data(opt, phase1, from_vid)"

    def path_matches(self, vid_paths, vid2_paths):
        assert len(vid_paths) == len(vid2_paths)
        matches = [os.path.basename(vp) == os.path.basename(v2p) for vp, v2p in zip(vid_paths, vid2_paths)]
        if not all(matches):
            print(f"{np.sum(matches)}/{len(matches)} path matches")
        return all(matches)

    def get_clips(self, data_type, frames_per_clip, metadata=None):
        data = None
        if metadata is None:
            try:
                metadata = deepcopy(self.vid_clips.metadata)
                assert self.path_matches(metadata["video_paths"], self.data[f"{data_type}_paths"])
                metadata["video_paths"] = self.data[f"{data_type}_paths"]
            except:
                metadata = None
                data = self.data[f"{data_type}_paths"]
        return VideoClips(data,
                          clip_length_in_frames=frames_per_clip,
                          frames_between_clips=self.opt.vid_skip,
                          num_workers=self.opt.num_workers,
                          _precomputed_metadata=metadata)

    def get_augmentation_parameters(self):
        v_flip = random.random() > 0.5 if self.phase == 'train' and not self.opt.no_v_flip else False
        h_flip = random.random() > 0.5 if self.phase == 'train' and not self.opt.no_h_flip else False
        h = int(self.opt.true_dim)
        w = int(self.opt.true_dim * self.opt.true_ratio)
        zoom = None
        if self.opt.fixed_top_centered_zoom:
            h_crop = int(h / self.opt.fixed_top_centered_zoom)
            w_crop = int(h_crop * self.opt.aspect_ratio)
            top_crop = 0
            assert w >= w_crop
            left_crop = int((w - w_crop) / 2)
            scale = None
        elif self.opt.fixed_crop:
            h_crop = self.opt.fixed_crop[0] # if self.phase1 == 'train' else h
            w_crop = self.opt.fixed_crop[1] # if self.phase1 == 'train' else w
            zoom = self.opt.min_zoom + random.random() * (self.opt.max_zoom - self.opt.min_zoom) if self.phase == 'train' else 1.
            h_scaled = int(h * zoom)
            w_scaled = int(w * zoom)
            scale = (h_scaled, w_scaled)
            assert h_scaled - h_crop >= 0
            assert w_scaled - w_crop >= 0
            h_p, w_p = (random.random(), random.random())
            if self.opt.centered_crop:
                h_p, w_p = (0.5, 0.5)
            if self.opt.horizontal_centered_crop:
                w_p = 0.5
            top_crop = int(h_p * (h_scaled - h_crop)) # if self.phase1 == 'train' else 0
            left_crop = int(w_p * (w_scaled - w_crop)) # if self.phase1 == 'train' else 0
        else:
            min_zoom = max(1., self.opt.aspect_ratio / self.opt.true_ratio)
            max_zoom = max(self.opt.max_zoom, min_zoom)
            zoom = min_zoom + random.random() * (max_zoom - min_zoom) if self.phase == 'train' else min_zoom
            h_crop = int(h / zoom)
            w_crop = int(h_crop * self.opt.aspect_ratio)
            assert h >= h_crop
            assert w >= w_crop
            top_crop = int(random.random() * (h - h_crop)) if self.phase == 'train' else 0
            left_crop = int(random.random() * (w - w_crop)) if self.phase == 'train' else 0
            scale = None
        if self.opt.colorjitter is not None and self.phase == 'train':
            brightness = (random.random() * 2 - 1) * self.opt.colorjitter
            contrast = 0 if self.opt.colorjitter_no_contrast else (random.random() * 2 - 1) * self.opt.colorjitter
            saturation = (random.random() * 2 - 1) * self.opt.colorjitter
            hue = 0.5 * (random.random() * 2 - 1) * self.opt.colorjitter
            brightness = max(0, 1 + brightness)
            contrast = max(0, 1 + contrast)
            saturation = max(0, 1 + saturation)
            colorjitter = [[brightness, brightness], [contrast, contrast], [saturation, saturation], [hue, hue]]
        else:
            colorjitter = None
        rot = (random.random() * 2 - 1) * self.opt.rotate * 180 % 360 if self.phase == 'train' else 0
        return v_flip, h_flip, top_crop, left_crop, h_crop, w_crop, scale, colorjitter, rot, zoom

    def load_rgb_path(self, p, transform=None):
        img = PIL.Image.open(p).convert('RGB')
        if transform is not None:
            img = transform(img)
        return img

    def load_layout_path(self, p, transform=None):
        layout = PIL.Image.open(p)
        layout = (transforms.ToTensor()(layout) * 255).long()
        for i in range(len(self.opt.remap_lyt) // 2):
            src_idx, tgt_idx = self.opt.remap_lyt[2 * i], self.opt.remap_lyt[2 * i + 1]
            layout[layout == src_idx] = tgt_idx
        layout = torch.zeros(self.opt.num_lyt, *layout.shape[-2:]).scatter_(0, layout, 1)
        if transform is not None:
            layout = transform(layout)
        layout = 5 * (layout * 2 - 1)
        return layout

    def load_flow_path(self, p, transform=None, zoom=None, v_flip=False, h_flip=False, rot=0, scale=None):
        with open(p, 'rb') as f:
            header = f.read(4)
            assert header.decode('utf-8') == 'PIEH'
            width = np.fromfile(f, np.int32, 1).squeeze()
            height = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
        flow = torch.from_numpy(flow).permute(2, 0, 1)
        if rot != 0:
            rot = rot / 180 * math.pi
            flow[0], flow[1] = flow[0] * math.cos(rot) - flow[1] * math.sin(rot), flow[0] * math.sin(rot) + flow[1] * math.cos(rot)
        if zoom is not None:
            flow = flow * zoom
        if v_flip:
            flow[0] = -flow[0]
        if h_flip:
            flow[1] = -flow[1]
        flow[0] = 2.0 * flow[0] / width
        flow[1] = 2.0 * flow[1] / height
        if self.opt.flow_dim > 0:
            transform = transforms.Compose([transforms.Resize(self.true_dim, PIL.Image.BILINEAR), transform])
        if transform is not None:
            flow = transform(flow)
        return flow

    def __getitem__(self, index):
        input_dict = {}
        v_flip, h_flip, top_crop, left_crop, h_crop, w_crop, scale, colorjitter, rot, zoom = self.get_augmentation_parameters()
        transform_rgb = get_transform(self.dim, rot=rot, v_flip=v_flip, h_flip=h_flip, top_crop=top_crop, left_crop=left_crop,
                                      h_crop=h_crop, w_crop=w_crop, resize=self.opt.resize_img, scale=scale,
                                      imagenet=self.opt.imagenet_norm, colorjitter=colorjitter,
                                      is_PIL=not self.from_vid, resize_center_crop=self.opt.resize_center_crop_img, aspect_ratio=self.opt.aspect_ratio)
        transform_layout = get_transform(self.dim, rot=rot, v_flip=v_flip, h_flip=h_flip, top_crop=top_crop, left_crop=left_crop,
                                         h_crop=h_crop, w_crop=w_crop, resize=self.opt.resize_img, scale=scale,
                                         is_PIL=False, resize_center_crop=self.opt.resize_center_crop_img,
                                         normalize=False, aspect_ratio=self.opt.aspect_ratio) # method=PIL.Image.NEAREST,
        transform_flow = transform_layout

        if self.from_animation:
            if self.load_vid:
                vid, obj = self.make_vid()
                input_dict['vid'] = vid
                input_dict['obj'] = obj
            else:
                img, obj = self.make_img()
                input_dict['img'] = img
                input_dict['obj'] = obj
        elif self.from_vid:
            vid, _, _, video_index = self.vid_clips.get_clip(index)
            vid = (vid.float() / 255).permute(0, 3, 1, 2)
            if 'vid_labels' in self.data:
                input_dict['vid_lbl'] = self.data['vid_labels'][video_index]
            if 'vid_id' in self.data:
                input_dict['vid_id'] = self.data['vid_id'][video_index]
            if self.load_vid:
                if self.opt.load_vid_len is not None:
                    vid_len = self.opt.vid_len
                    step = min(max(1, int(random.random() * (self.opt.load_vid_len - 1) / (vid_len - 1))), self.opt.max_vid_step)
                    start = int(random.random() * (self.opt.load_vid_len - (vid_len - 1) * step)) if self.phase == "train" else 0
                    end = start + step * (vid_len - 1) + 1
                    vid = vid[start:end:step]
                input_dict['vid'] = transform_rgb(vid)
            else:
                img = transform_rgb(vid)
                input_dict['img'] = img.squeeze(0)
        else:
            if self.load_vid:
                frame_paths = self.data['vid_frame_paths'][index]
                # if self.opt.skip_first:
                #     frame_paths = frame_paths[1:]
                frames_per_clip = self.opt.load_vid_len if self.opt.load_vid_len is not None else self.opt.vid_len
                assert len(frame_paths) >= frames_per_clip, f"{frame_paths}, {frames_per_clip}"
                idx = random.randrange(len(frame_paths) - ((frames_per_clip - 1) * self.opt.one_every_n) - 1) if self.phase == "train" else 0
                frame_paths = frame_paths[idx:idx + frames_per_clip * self.opt.one_every_n:self.opt.one_every_n]
                if self.opt.load_vid_len is not None:
                    if self.opt.load_n_plus_1:
                        start = int(random.random() * (self.opt.load_vid_len - (self.opt.vid_len - 1)))
                        end = start + self.opt.vid_len - 1
                        last = int(random.random() * (self.opt.load_vid_len - end))
                        frame_paths = frame_paths[start:end] + [frame_paths[end + last]]
                    elif self.opt.load_n_rd:
                        rd_idx = list(range(self.opt.load_vid_len))
                        random.shuffle(rd_idx)
                        frame_paths = [frame_paths[i] for i in rd_idx[:self.opt.vid_len]]
                    elif self.opt.load_2_apart:
                        idx1 = int(0.25 * random.random() * (self.opt.load_vid_len - 1))
                        idx2 = int((1 - 0.25 * random.random()) * (self.opt.load_vid_len - 1))
                        rd_idx = [idx1, idx2]
                        random.shuffle(rd_idx)
                        frame_paths = [frame_paths[i] for i in rd_idx]
                    else:
                        vid_len = self.opt.vid_len if self.opt.p2p_len is None else self.opt.p2p_len
                        step = max(1, int(random.random() * (self.opt.load_vid_len - 1) / (vid_len - 1)))
                        start = int(random.random() * (self.opt.load_vid_len - (vid_len - 1) * step))
                        end = start + step * (vid_len - 1) + 1
                        frame_paths = frame_paths[start:end:step]
                vid = torch.zeros(self.opt.vid_len, 3, self.dim, int(self.dim * self.opt.aspect_ratio))
                for k, frame_path in enumerate(frame_paths):
                    try:
                        vid[k] = self.load_rgb_path(frame_path, transform_rgb)
                    except:
                        print("Error with file", frame_path)
                        raise ValueError
                input_dict['vid'] = vid
                input_dict['path'] = frame_paths[0]
                if self.opt.load_lyt:
                    lyt = torch.zeros(self.opt.vid_len, self.opt.num_lyt, self.dim, int(self.dim * self.opt.aspect_ratio))
                    lyt_paths = [p.replace(self.frame_folder, self.layout_folder) for p in frame_paths]
                    for k, lyt_path in enumerate(lyt_paths):
                        lyt[k] = self.load_layout_path(lyt_path, transform_layout)
                    input_dict['lyt'] = lyt
                if self.opt.load_flow:
                    flow = torch.zeros(self.opt.vid_len, 2, self.dim, int(self.dim * self.opt.aspect_ratio))
                    flow_paths = [p.replace(self.frame_folder, self.flow_folder).replace(".png", ".flo") for p in frame_paths]
                    for k, flow_path in enumerate(flow_paths):
                        flow[k] = self.load_flow_path(flow_path, transform_flow, zoom, v_flip, h_flip, rot)
                    input_dict['flow'] = flow
                if 'vid_labels' in self.data:
                    vid_lbl = self.data['vid_labels'][index]
                    input_dict['vid_lbl'] = vid_lbl
                if 'vid_cap' in self.data:
                    input_dict['cap'] = self.data['vid_cap'][index]
            else:
                frame_path = self.data['frame_paths'][index]
                if 'frame_labels' in self.data:
                    frame_lbl = self.data['frame_labels'][index]
                    input_dict['vid_lbl'] = frame_lbl
                input_dict['img'] = self.load_rgb_path(frame_path, transform_rgb)
                if self.opt.load_lyt:
                    lyt_path = frame_path.replace(self.frame_folder, self.layout_folder)
                    lyt = self.load_layout_path(lyt_path, transform_layout)
                    input_dict['lyt'] = lyt
                if self.opt.load_flow:
                    flow_path = frame_path.replace(self.frame_folder, self.flow_folder).replace(".png", ".flo")
                    flow = self.load_flow_path(flow_path, transform_flow, zoom, v_flip, h_flip, rot)
                    input_dict['flow'] = flow
                if 'frame_cap' in self.data:
                    input_dict['cap'] = self.data['frame_cap'][index]

        return input_dict

    def __len__(self):
        return self.dataset_size


def get_transform(dim, rot=0, v_flip=False, h_flip=False, method=PIL.Image.BILINEAR, normalize=True, imagenet=False, top_crop=None, left_crop=None, h_crop=None,
                  w_crop=None, resize=None, resize_center_crop=None, scale=None, colorjitter=None, is_PIL=True, blur=None, aspect_ratio=None):
    transform_list = []
    if resize is not None:
        transform_list.append(transforms.Resize(resize, method))
    if resize_center_crop is not None:
        transform_list.append(transforms.Resize(resize_center_crop, method))
        transform_list.append(transforms.CenterCrop(resize_center_crop))
    if scale is not None:
        transform_list.append(transforms.Resize(scale, method))
    if rot > 0:
        transform_list.append(transforms.Lambda(lambda img: transforms.functional.rotate(img, rot, method)))
    if top_crop is not None:
        transform_list.append(transforms.Lambda(lambda img: transforms.functional.crop(img, top_crop, left_crop, h_crop, w_crop)))
    size = [dim, int(dim * aspect_ratio)]
    transform_list.append(transforms.Resize(size, method))
    if v_flip:
        if is_PIL:
            transform_list.append(transforms.Lambda(lambda img: img.transpose(PIL.Image.FLIP_LEFT_RIGHT)))
        else:
            transform_list.append(transforms.Lambda(lambda img: img.flip(-1)))
    if h_flip:
        if is_PIL:
            transform_list.append(transforms.Lambda(lambda img: img.transpose(PIL.Image.FLIP_TOP_BOTTOM)))
        else:
            transform_list.append(transforms.Lambda(lambda img: img.flip(-2)))
    if colorjitter is not None:
        transform_list.append(transforms.ColorJitter(brightness=colorjitter[0], contrast=colorjitter[1], saturation=colorjitter[2], hue=colorjitter[3]))
    if blur is not None:
        s1, s2 = blur
        s = s1 + (s2 - s1) * random.random()
        k = int(3 * s) + 1 if int(3 * s) % 2 == 0 else int(3 * s)
        transform_list.append(transforms.GaussianBlur(kernel_size=max(3, min(k, 13)), sigma=s))
    if is_PIL:
        transform_list.append(transforms.ToTensor())
    if normalize:
        if imagenet:
            transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        else:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)
