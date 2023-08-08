import torch
import os
from glob import glob

def save_network(net, label, which_iter, opt, name):
    if net is None:
        return

    if name is not None:
        save_path = os.path.join(opt.checkpoint_path, f'{label}_{name}_net_{which_iter}.pth')
        old_paths = glob(os.path.join(opt.checkpoint_path, f"{label}_{name}_net_*.pth"))
    else:
        save_path = os.path.join(opt.checkpoint_path, f'{label}_net_{which_iter}.pth')
        old_paths = None

    torch.save(net.state_dict(), save_path)

    if old_paths is not None:
        for old_path in old_paths:
            open(old_path, 'w').close()
            os.unlink(old_path)

def load_state_dict(net, state_dict, strict=True, from_multi=False):
    if from_multi:
        rm_prefix = lambda key: key[len("module."):] if key.startswith("module.") else key
        state_dict = {rm_prefix(key): value for key, value in state_dict.items()}
    if not strict:
        model_dict = net.state_dict()
        pop_list = []
        # remove the keys which don't match in size
        for key in state_dict:
            if key in model_dict:
                if model_dict[key].shape != state_dict[key].shape:
                    pop_list.append(key)
                    print(f"Size mismatch for {key}")
            else:
                pop_list.append(key)
                print(f"Key missing in model for {key}")
        for key in model_dict:
            if key not in state_dict:
                print(f"Key missing in ckpt for {key}")
        for key in pop_list:
            state_dict.pop(key)
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)
    else:
        net.load_state_dict(state_dict)

def load_network(net, label, opt, iter=None, load_path=None, required=True, map_location=None, verbose=False):
    if net is None:
        return None

    which_iter = iter if iter is not None else opt.which_iter
    load_path = load_path if load_path is not None else opt.load_path
    try:
        int(which_iter)
        name = None
        load_from_name = False
    except:
        name = which_iter
        load_from_name = True

    if load_path is not None and which_iter is not None:
        if load_from_name:
            load_paths = glob(os.path.join(load_path, f"{label}_{name}_net_*.pth"))
            if len(load_paths) == 0:
                if required:
                    raise ValueError(f"No checkpoint for {label} net with name {name} and path {load_path}")
                else:
                    if verbose:
                        print(f"No checkpoint for {label} net with name {name} and path {load_path}")
                        print(f"Loading untrained {label} net")
                    return net
            # assert len(load_paths) > 0, f"Did not find any checkpoint for {label} net with name {name} and path {load_path}"
            assert len(load_paths) == 1, f"Too many checkpoint candidates for {label} net with name {name}:\n{load_paths}"
            load_path = load_paths[0]
        else:
            load_paths = glob(os.path.join(load_path, f"{label}_*net_{which_iter}.pth"))
            if len(load_paths) == 0:
                if required:
                    raise ValueError(f"No checkpoint for {label} net at iter {which_iter} and path {load_path}")
                else:
                    if verbose:
                        print(f"No checkpoint for {label} net at iter {which_iter} and path {load_path}")
                        print(f"Loading untrained {label} net")
                    return net
            else:
                load_path = load_paths[0]
        state_dict = torch.load(load_path, map_location=map_location)
        load_state_dict(net, state_dict, strict=not opt.not_strict, from_multi=opt.from_multi)
        if verbose:
            print(f"Loading checkpoint for {label} net from {load_path}")

    elif opt.cont_train and which_iter is not None:
        assert not load_from_name, "If load_path is not specified, which_iter should be an int"
        load_paths = glob(os.path.join(opt.save_path, "checkpoints", f"*-{opt.name}", f"{label}_*net_{which_iter}.pth"))
        assert len(load_paths) > 0, f"Did not find any checkpoint for {label} net at iter {which_iter} for {opt.name}"
        assert len(load_paths) == 1, f"Too many checkpoint candidates for {label} net at iter {which_iter} for {opt.name}:\n{load_paths}"
        load_path = load_paths[0]
        load_state_dict(net, torch.load(load_path, map_location=map_location), strict=not opt.not_strict, from_multi=opt.from_multi)
        if verbose:
            print(f"Loading checkpoint for {label} net from {load_path}")

    else:
        if verbose:
            print(f"Loading untrained {label} net")

    return net

def print_network(net):
    if net is not None:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)