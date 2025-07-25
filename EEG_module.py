import torch.nn.functional as F
import torch
import torch.nn as nn
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Function
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

#####GRL
class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


######Base module of DANN and MLDA model
class DANN_dwt(nn.Module):
    def __init__(self, input_size = 64*5, num_classes=20, alpha = 0.3):
        super(DANN_dwt, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, 512)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(p=0.6)
        self.alpha = alpha

        self.fc2 = nn.Linear(512, 256)
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(p=0.6)

        self.fc3 = nn.Linear(256, 128)
        self.tanh3 = nn.Tanh()
        self.dropout3 = nn.Dropout(p=0.6)

        self.fc4 = nn.Linear(128, num_classes)

        self.domain_discriminator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.dropout2(x)
        

        x = self.fc3(x)
        x = self.tanh3(x)
        x = self.dropout3(x)
        feature = x
        #feature = feature.unsqueeze(2)
        feature_rev = ReverseLayer.apply(feature, self.alpha)
        domain_pred = self.domain_discriminator(feature_rev)

        x = self.fc4(x)
        #print("fc4 x shape",x.shape)

        return feature, x, domain_pred
    
#### The MVDA model(ABP,EEGNet,Transformer,AL,MMD,ADL)
# The ABP module
class ABPlayer(nn.Module):
    def __init__(self, cuda=True, input_dim=310):
        super(ABPlayer, self).__init__()
        self.input_dim = input_dim
        if cuda:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim).cuda())
            self.u_linear = nn.Parameter(torch.randn(input_dim).cuda())
        else:
            self.w_linear = nn.Parameter(torch.randn(input_dim, input_dim))
            self.u_linear = nn.Parameter(torch.randn(input_dim))

    def forward(self, x, batch_size):
        x_reshape = torch.Tensor.reshape(x, [-1, self.input_dim])
        attn_softmax = F.softmax(torch.mm(x_reshape, self.w_linear)+ self.u_linear,1)
        res = torch.mul(attn_softmax, x_reshape)
        res = torch.Tensor.reshape(res, [batch_size, x.shape[1], x.shape[2]])
        return res


####EEGConformer

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim_head * heads, dim),
            nn.Dropout(dropout)
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Multi-Head Self-Attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # Add & Norm
        x = self.norm1(x + out)

        # Feed-Forward Network
        x = self.norm2(x + self.mlp(x))

        return x

class Conformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(ConformerBlock(dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EEGConformer(nn.Module):
    def __init__(self, input_size, num_classes, model_dim, depth, heads, dim_head, mlp_dim, dropout=0.1, alpha = 0.3):
        super().__init__()
        self.input_size = input_size
        self.model_dim = model_dim
        self.channels = input_size[0]
        self.num_classes = num_classes
        self.alpha = alpha

        # ABP Block
        self.ABPlayer = ABPlayer(cuda=True, input_dim=310)


        # EEGNet-like Convolutional Block
        self.conv1 = nn.Conv2d(1, 40, kernel_size=(20, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv2 = nn.Conv2d(40, 40, kernel_size=(43, 1), stride=(1, 1), padding=(0, 0), groups=40, bias=True)#SEEDVIIkernel_size=(43, 1)
        self.conv3 = nn.Conv2d(40, 40, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True)

        self.bn1 = nn.BatchNorm2d(40, eps=1e-5, momentum=0.1)
        self.bn2 = nn.BatchNorm2d(40, eps=1e-5, momentum=0.1)
        self.bn3 = nn.BatchNorm2d(40, eps=1e-5, momentum=0.1)

        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.dropout = nn.Dropout(dropout)

        self.layer_Norm = nn.LayerNorm([40, 62, 5])
        self.layer_Norm2 = nn.LayerNorm(56)

        # Projection to model_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),  # Reshape to (batch_size, 5*62, 40)
            nn.Linear(40, model_dim)           # Project to (batch_size, 5*62, model_dim)
        )

        # Conformer Encoder
        self.conformer = Conformer(model_dim, depth, heads, dim_head, mlp_dim, dropout)

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes)
        )

        #Domain adversarial training
        self.domain_discriminator = nn.Sequential(
            nn.Linear(56, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):

        # ABP layer
        x_ABP = self.ABPlayer(x, x.shape[0])
        x_ABP = x_ABP.unsqueeze(1)

        # Add channel dimension
        x1 = x.unsqueeze(1)  # (batch_size, 1, 62, 5)

        # EEGNet-like Convolutional Block
        x = self.conv1(x1)  # (batch_size, 40, 62, 5)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.conv2(x)  # (batch_size, 40, 1, 5)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.conv3(x)  # (batch_size, 40, 1, 5)
        x = self.bn3(x)
        x = self.elu(x)
        
        x = self.dropout(x)
        #print("feature1 shape", x.shape)
        #feature1 = x[:,:,0,0]###（batch,40)
        # print("x shape", x.shape)
        # print("feature1 shape", feature1.shape)

        #x = self.avg_pool(x)  # (batch_size, 40, 1, 1)
        x = x.expand(-1, -1, 62, 5)  # SEEDVIIx = x.expand(-1, -1, 62, 5)

        #x2 = x1.expand(-1, 40, -1, -1)  # (batch_size, 40, 62, 5)

        x2 = x_ABP.expand(-1, 40, -1, -1)
        x = x + x2   ###(B,40,62,5)

        x = self.layer_Norm(x)


        feature1 = rearrange(x, 'b m c h -> b m (h c)')
        feature1 = feature1.mean(dim=2)  # (batch_size, model_dim)


        # EEG_feature = x.mean(dim=3)  # (batch_size, model_dim)
        # EEG_feature = EEG_feature
        # feature_rev = ReverseLayer.apply(EEG_feature, self.alpha)
        # domain_pred = self.domain_discriminator(feature_rev)


        # x = x.squeeze(-1).squeeze(-1)  # (batch_size, 40)
        # print(x.shape)

        # # Repeat channels to match model_dim
        # x = x.unsqueeze(1).repeat(1, self.input_size[0], 1)  # (batch_size, 62, 40)
        # x = rearrange(x, 'b c h -> b h c')  # (batch_size, 62, 40) -> (batch_size, 62, 40)
        # print(x.shape)

        # Projection to model_dim
        x = self.to_patch_embedding(x)  # (batch_size, 62*5, model_dim)

        # Conformer Encoder
        x = self.conformer(x)  # (batch_size, 62*5, model_dim)


        x = x.mean(dim=1)  # (batch_size, model_dim)
        #print("feature2 shape", x.shape)


        feature2 = torch.cat((x, feature1), dim=1)
        feature2 = self.layer_Norm2(feature2)


        feature_rev = ReverseLayer.apply(feature2, self.alpha)
        domain_pred = self.domain_discriminator(feature_rev)

        # MLP Head
        out = self.mlp_head(x)  # (batch_size, num_classes)

        return feature1, feature2, out, domain_pred

####EEGNet
class my_EEGNet(nn.Module):
    def __init__(self):
        super(my_EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 20), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(64, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=32, out_features=20, bias=True)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        feature = x.view(x.size(0), -1)  # Flatten for linear layer
        x = self.classify(feature)
        return feature, x


class TransformerModel(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=8, num_layers=1, dim_feedforward=256, num_classes=20):
        super(TransformerModel, self).__init__()


        self.embedding = nn.Linear(input_dim, d_model)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.FC = nn.Linear(d_model * 62, 256)


        self.classifier = nn.Sequential(

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        batch_size = x.size(0)


        x = self.embedding(x)  # [batch_size, seq_len, d_model]


        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)


        x = x.reshape(batch_size, -1)  # [batch_size, d_model * seq_len]

        feature = self.FC(x)


        output = self.classifier(feature)

        return feature, output