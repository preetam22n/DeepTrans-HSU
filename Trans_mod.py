import os
import pickle
import time

import scipy.io as sio
import torch
import torch.nn as nn
from torchsummary import summary

import datasets
import plots
import transformer
import utils


class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim*P)//patch**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((dim*P)//patch**2, momentum=0.5),
        )

        self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=(dim*P), depth=2,
                                      heads=8, mlp_dim=12, pool='cls')
        
        self.upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),
        )
        
        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        abu_est = self.encoder(x)
        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.view(1, self.P, -1)
        abu_est = self.upscale(cls_emb).view(1, self.P, self.size, self.size)
        abu_est = self.smooth(abu_est)
        re_result = self.decoder(abu_est)
        return abu_est, re_result


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)


class Train_test:
    def __init__(self, dataset, device, skip_train=False, save=False):
        super(Train_test, self).__init__()
        self.skip_train = skip_train
        self.device = device
        self.dataset = dataset
        self.save = save
        self.save_dir = "trans_mod_" + dataset + "/"
        os.makedirs(self.save_dir, exist_ok=True)
        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95
            self.LR, self.EPOCH = 6e-3, 200
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 5e3, 3e-2
            self.weight_decay_param = 4e-5
            self.order_abd, self.order_endmem = (0, 1, 2), (0, 1, 2)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        elif dataset == 'apex':
            self.P, self.L, self.col = 4, 285, 110
            self.LR, self.EPOCH = 9e-3, 200
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 5e3, 5e-2
            self.weight_decay_param = 4e-5
            self.order_abd, self.order_endmem = (3, 1, 2, 0), (3, 1, 2, 0)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        elif dataset == 'dc':
            self.P, self.L, self.col = 6, 191, 290
            self.LR, self.EPOCH = 6e-3, 150
            self.patch, self.dim = 10, 400
            self.beta, self.gamma = 5e3, 1e-4
            self.weight_decay_param = 3e-5
            self.order_abd, self.order_endmem = (0, 2, 1, 5, 4, 3), (0, 2, 1, 5, 4, 3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        else:
            raise ValueError("Unknown dataset")

    def run(self, smry):
        net = AutoEncoder(P=self.P, L=self.L, size=self.col,
                          patch=self.patch, dim=self.dim).to(self.device)
        if smry:
            summary(net, (1, self.L, self.col, self.col), batch_dim=None)
            return

        net.apply(net.weights_init)

        model_dict = net.state_dict()
        model_dict['decoder.0.weight'] = self.init_weight
        net.load_state_dict(model_dict)

        loss_func = nn.MSELoss(reduction='mean')
        loss_func2 = utils.SAD(self.L)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
        apply_clamp_inst1 = NonZeroClipper()
        
        if not self.skip_train:
            time_start = time.time()
            net.train()
            epo_vs_los = []
            for epoch in range(self.EPOCH):
                for i, (x, _) in enumerate(self.loader):

                    x = x.transpose(1, 0).view(1, -1, self.col, self.col)
                    abu_est, re_result = net(x)

                    loss_re = self.beta * loss_func(re_result, x)
                    loss_sad = loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                          x.view(1, self.L, -1).transpose(1, 2))
                    loss_sad = self.gamma * torch.sum(loss_sad).float()

                    total_loss = loss_re + loss_sad

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()

                    net.decoder.apply(apply_clamp_inst1)
                    
                    if epoch % 10 == 0:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data)
                    epo_vs_los.append(float(total_loss.data))

                scheduler.step()
            time_end = time.time()
            
            if self.save:
                with open(self.save_dir + 'weights_new.pickle', 'wb') as handle:
                    pickle.dump(net.state_dict(), handle)
                sio.savemat(self.save_dir + f"{self.dataset}_losses.mat", {"losses": epo_vs_los})
            
            print('Total computational cost:', time_end - time_start)

        else:
            with open(self.save_dir + 'weights.pickle', 'rb') as handle:
                net.load_state_dict(pickle.load(handle))

        # Testing ================

        net.eval()
        x = self.data.get("hs_img").transpose(1, 0).view(1, -1, self.col, self.col)
        abu_est, re_result = net(x)
        abu_est = abu_est / (torch.sum(abu_est, dim=1))
        abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
        true_endmem = self.data.get("end_mem").numpy()
        est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
        est_endmem = est_endmem.reshape((self.L, self.P))

        abu_est = abu_est[:, :, self.order_abd]
        est_endmem = est_endmem[:, self.order_endmem]

        sio.savemat(self.save_dir + f"{self.dataset}_abd_map.mat", {"A_est": abu_est})
        sio.savemat(self.save_dir + f"{self.dataset}_endmem.mat", {"E_est": est_endmem})

        x = x.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
        re_result = re_result.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
        re = utils.compute_re(x, re_result)
        print("RE:", re)

        rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
        print("Class-wise RMSE value:")
        for i in range(self.P):
            print("Class", i + 1, ":", rmse_cls[i])
        print("Mean RMSE:", mean_rmse)

        sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
        print("Class-wise SAD value:")
        for i in range(self.P):
            print("Class", i + 1, ":", sad_cls[i])
        print("Mean SAD:", mean_sad)

        with open(self.save_dir + "log1.csv", 'a') as file:
            file.write(f"LR: {self.LR}, ")
            file.write(f"WD: {self.weight_decay_param}, ")
            file.write(f"RE: {re:.4f}, ")
            file.write(f"SAD: {mean_sad:.4f}, ")
            file.write(f"RMSE: {mean_rmse:.4f}\n")

        plots.plot_abundance(target, abu_est, self.P, self.save_dir)
        plots.plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir)
        
# =================================================================

if __name__ == '__main__':
    pass
