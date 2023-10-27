# import torch

# from torch import nn

# class PredictionLoss(nn.Module):
#     def __init__(self):
#         super(PredictionLoss, self).__init__()

#     def forward(self, output, target):
#          criterion = nn.MSELoss()
#          loss = criterion(output, target)
#          return loss









#         loss = 0
#         for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors):
#             λ = eigenvalue
#             g = eigenvector.reshape(3,1)
#             g_H = torch.conj(torch.transpose(g, 0, 1))
#             A_H = torch.conj(torch.transpose(A, 0, 1))

#             res = torch.sqrt(
#                 (g_H @ (L - λ*A_H - torch.conj(λ)*A + (torch.abs(λ)**2)*G) @ g) /
#                 (g_H @ G.to(torch.complex64) @ g)
#             )
            
#             if torch.isnan(res):
#                 pass
#             else:
#                 loss = loss + (res**2)

#         return loss