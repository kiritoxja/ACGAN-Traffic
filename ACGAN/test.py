import torch
save_dir = './save'

# torch.save(a,save_dir + r'\a.pkl')
print(torch.load(save_dir + '/a.pkl'))