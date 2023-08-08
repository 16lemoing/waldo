class Helper:
    def __init__(self):
        pass

    def next_batch(self, data_info):
        assert hasattr(self, 'engine')
        try:
            return next(data_info["loader_iter"])
        except StopIteration:
            if data_info["num_folds"] is not None:
                num_folds = data_info["num_folds"]
                epoch = data_info["epoch"] + 1 / num_folds
                fold = (data_info["fold"] + 1) % num_folds
                phase, data_type = data_info["phase1"], data_info["data_type"]
                # free memory from previous fold before loading the next
                for k in data_info:
                    data_info[k] = None
                new_data_info = self.get_data_info(phase, data_type, fold, num_folds)
                new_data_info["epoch"] = epoch
                for k in data_info:
                    data_info[k] = new_data_info[k]
            else:
                data_info["epoch"] += 1
                if self.engine.distributed:
                    data_info["datasampler"].set_epoch(data_info["epoch"])
                data_info["loader_iter"] = iter(data_info["dataloader"])
            return next(data_info["loader_iter"])

    def reinit_batches(self, data_info):
        data_info["loader_iter"] = iter(data_info["dataloader"])

    def get_data_info(self, phase, data_type, fold=None, num_folds=None):
        assert hasattr(self, 'engine')
        assert hasattr(self, 'opt')
        fold = 0 if num_folds is not None and fold is None else fold
        dataset = self.engine.create_dataset(self.opt,
                                             phase=phase,
                                             fold=fold,
                                             load_vid=data_type == "vid")
        if data_type == "img":
            batch_size = self.opt.batch_size_img // (self.opt.load_n_from_tar if self.opt.is_tar else 1)
            persist = self.opt.is_tar
        else:
            batch_size = self.opt.batch_size_vid
            persist = False
        num_workers = self.opt.num_workers_eval if phase != "train" and self.opt.num_workers_eval is not None else self.opt.num_workers
        loader_info = self.engine.create_dataloader(dataset, batch_size, num_workers, is_train=phase == "train", persist=persist)
        dataloader, datasampler, batch_size_per_gpu = loader_info
        loader_iter = iter(dataloader)
        return {"dataloader": dataloader, "datasampler": datasampler, "epoch": 0, "phase1": phase, "data_type": data_type,
                "batch_size_per_gpu": batch_size_per_gpu, "loader_iter": loader_iter, "fold": fold,
                "num_folds": num_folds}

    def run(self):
        assert False, "A subclass of Trainer must override self.run()"