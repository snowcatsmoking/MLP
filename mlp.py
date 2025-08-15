import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # 批标准化，加快收敛
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ===== 测试代码 =====
if __name__ == "__main__":
    model = MLP(input_dim=20, output_dim=3)  # 比如20个特征，3分类
    x = torch.randn(5, 20)  # batch_size=5
    print(model(x).shape)   # 输出: torch.Size([5, 3])
