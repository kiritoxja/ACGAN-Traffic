import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk


#参数定义
cuda = True if torch.cuda.is_available() else False
n_class = 9
batch_size = 64
lr = 0.0002
n_epochs = 2000
save_dir = r'.\save'
data_dir = r'..\trainTestData\500_augmentTrain.pkl'

#生成器参数
#生成器噪声输入维度 生成器输入向量维度latent_dim + n_class
latent_dim = 100
# 生成器网络模型参数
G_hidden_size_1 = 256
G_hidden_size_2 = 512

#判别器参数
session_size = 500
D_hidden_size_1 = 512
D_hidden_size_2 = 256
D_hidden_size_3 = 128


#筛选需要进行扩充的类别
# def filter(data:pd.DataFrame,numClass:int,threshold:int):
#     result = pd.DataFrame()
#     labels = [i for i in range(numClass)]
#     augmentLables = []
#     for i in labels:
#         if(sum(data['label'] == i))< threshold:
#             augmentLables.append(i)
#     choosen = data['label'].isin(augmentLables)
#     return data[choosen]


# Configure data loader
class MydataSet(tud.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        session = self.data.iloc[idx, 1:]



# 模型初始化函数
# def weights_init_normal(m):
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#         session = self.data.iloc[idx, 1:].tolist()
#         return label,session

# 模型初始化函数
def weights_init_normal(m):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 保存了每个标签的向量  每个标签转化为一个正太分布的向量
        self.label_emb = nn.Embedding(n_class,latent_dim)

        #生成器的网络结构
        self.model = nn.Sequential(
            nn.Linear(latent_dim , G_hidden_size_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(G_hidden_size_1, G_hidden_size_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(G_hidden_size_2, session_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, noise, labels):
        '''
        :param noise: 输入的噪声  batchsize * latent_dim
        :param labels: 标签输入   batchsize * n_class
        :return: 产生 batchsize * session_size的输出
        '''
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.model(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(session_size,D_hidden_size_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(D_hidden_size_1, D_hidden_size_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(D_hidden_size_2, D_hidden_size_3),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(D_hidden_size_3, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(D_hidden_size_3, n_class), nn.Softmax())

    def forward(self, session):
        out = self.model(session)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

#DataSet
f = open(data_dir,'rb')
dataset = MydataSet(pk.load(f))
dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# ----------
#  Training
# ----------
G_losses = []
D_losses = []
ACC = []

for epoch in range(n_epochs):
    for i, (labels,sessions) in enumerate(dataloader):

        batch_size = sessions.shape[0]

        # Adversarial ground truths
        valid = FloatTensor(batch_size, 1).fill_(1.0)
        fake = FloatTensor(batch_size, 1).fill_(0.0)

        # Configure input
        real_sessions = sessions.type(FloatTensor)
        labels = labels.type(LongTensor)

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise and labels as generator input
        z = FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))
        gen_labels = LongTensor(np.random.randint(0, n_class, batch_size))

        # Generate a batch of sessions
        gen_sessions = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_sessions)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_sessions)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_sessions.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        # save loss
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        ACC.append(d_acc)

    if epoch %20 == 0:
        print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
        % (epoch, n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
         )

torch.save(generator,save_dir+r'\generator.pkl')
torch.save(G_losses,save_dir+r'\gloss.pkl')
torch.save(D_losses,save_dir+r'\dloss.pkl')
torch.save(ACC,save_dir+r'\acc.pkl')
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()