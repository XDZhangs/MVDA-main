import torch.nn.functional as F
import torch
import torch.nn as nn
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE", DEVICE)


class MMDLoss(nn.Module):

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        losses = []

        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class WalkerLoss(nn.Module):

    def forward(self, Psts, y):
        equality_matrix = torch.eq(y.clone().view(-1,1), y).float()
        p_target = equality_matrix / equality_matrix.sum(dim=1, keepdim=True)
        p_target.requires_grad = False

        L_walker = F.kl_div(torch.log(1e-7 + Psts), p_target, size_average=False)
        L_walker /= p_target.size()[0]

        return L_walker

class VisitLoss(nn.Module):

    def forward(self, Pt):
        p_visit = torch.ones([1, Pt.size()[1]]) / float(Pt.size()[1])
        p_visit.requires_grad = False
        if Pt.is_cuda: p_visit = p_visit.cuda()
        L_visit = F.kl_div(torch.log(1e-7 + Pt), p_visit, size_average=False)
        L_visit /= p_visit.size()[0]

        return L_visit

class AssociationMatrix(nn.Module):

    def __init__(self):
        super(AssociationMatrix, self).__init__()

    def forward(self, xs, xt):
        """
        xs: (Ns, K, ...)
        xt: (Nt, K, ...)
        """

        # TODO not sure why clone is needed here
        Bs = xs.size()[0]
        Bt = xt.size()[0]

        xs = xs.clone().view(Bs, -1)
        xt = xt.clone().view(Bt, -1)

        W = torch.mm(xs, xt.transpose(1,0))

        # p(xt | xs) as softmax, normalize over xt axis
        Pst = F.softmax(W, dim=1) # Ns x Nt
        # p(xs | xt) as softmax, normalize over xs axis
        Pts = F.softmax(W.transpose(1,0), dim=1) # Nt x Ns

        # p(xs | xs)
        Psts = Pst.mm(Pts) # Ns x Ns

        # p(xt)
        Pt = torch.mean(Pst, dim=0, keepdim=True) # Nt

        return Psts, Pt

class AssociativeLoss(nn.Module):

    def __init__(self, walker_weight = 0.6, visit_weight = 1.):
        super(AssociativeLoss, self).__init__()

        self.matrix = AssociationMatrix()
        self.walker = WalkerLoss()
        self.visit  = VisitLoss()

        self.walker_weight = walker_weight
        self.visit_weight  = visit_weight

    def forward(self, xs, xt, y):

        Psts, Pt = self.matrix(xs, xt)
        L_walker = self.walker(Psts, y)
        L_visit  = self.visit(Pt)

        return self.visit_weight*L_visit + self.walker_weight*L_walker