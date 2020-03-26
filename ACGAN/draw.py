import matplotlib.pyplot as plt
import torch

save_dir = r'.\save'
G_losses = torch.load(save_dir+r'\gloss.pkl')
D_losses = torch.load(save_dir+r'\dloss.pkl')

plt.figure()
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G_Loss")
plt.plot(D_losses,label="D_Loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()